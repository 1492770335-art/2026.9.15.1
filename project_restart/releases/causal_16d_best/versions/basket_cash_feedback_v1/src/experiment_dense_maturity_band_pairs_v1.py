"""Learn a paired 60-tick payoff with neighboring expiries as noisy repeated training outcomes."""
import numpy as np
import torch
from torch.nn import functional as F

import experiment_dense_pair_payoff_sizes_v1 as payoff
from prepare_dense_activity_reference_v1 import OUT as REFERENCE

pairs = payoff.pairs
rel = pairs.net.rel
OUT = REFERENCE.parent/"dense_maturity_band_pairs_v1"
ARMS = ("neighborhood_payoff", "specific_expiry", "conditional_endpoint")
USES_FORECAST = True


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4,16):
        def load(k):
            return np.load(rel.ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        grid=load("time").astype(int)
        y=load("y").astype(float)
        y-=y.mean(1,keepdims=True)
        p=np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else pairs.net.future.forecast(d)
        x=pairs.net.features(load("x"),grid,code,p)
        q=x[:,:,-1]
        scale=np.clip(np.sum(q*y[:,:,2])/np.sum(q.astype(float)**2),.2,4)
        inputs=[];targets=[]
        for a,b in rel.fields.parent.sessions(grid):
            t=np.arange(a+40+d%5,b-76,6)
            if not len(t):
                continue
            center=y[t,:,2]/scale-q[t]
            cash=[None]*7
            cash[3]=center
            for j in range(3):
                # Each neighboring target differs only by the five-tick payoff at its right boundary.
                cash[2-j]=cash[3-j]-rel.ref.base.UNITS[0]*y[t+55-5*j,:,0]/scale
                cash[4+j]=cash[3+j]+rel.ref.base.UNITS[0]*y[t+60+5*j,:,0]/scale
            targets.append(np.stack(cash,-1));inputs.append(x[t])
        np.savez(OUT/f"day_{d:02d}.npz",x=np.concatenate(inputs).astype(np.float16),
                 y=np.concatenate(targets).astype(np.float32))


def expected(e,c):
    magnitude=.08+F.softplus(c)
    return magnitude*(2*e[...,0].sigmoid()-1+.85*e[...,1].tanh())


def train(end):
    torch.manual_seed(45269)
    records=[]
    for day in range(4,end):
        with np.load(OUT/f"day_{day:02d}.npz") as f:
            records.append((f["x"],f["y"]))
    device="cuda" if torch.cuda.is_available() else "cpu"
    x,y=[torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for v in zip(*records)]
    unit=x.square().mean((0,1)).sqrt().clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    y/=y[...,3].square().mean().sqrt().clamp_min(.01)
    model=pairs.Comparison(x.shape[-1],effects=5,contexts=2).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0011,weight_decay=.045)
    for _ in range(5):
        for t in torch.randperm(len(x),device=device).split(32):
            e,c,peer,w=model(x[t])
            yy=y[t]
            gap=yy[:,:,None]-yy[torch.arange(len(t),device=device)[:,None,None],peer]
            losses=[]
            for j,values in enumerate((gap,gap[...,3:4])):
                sign=(values>0).float()
                logits=e[...,2*j,None]
                size=(.08+F.softplus(c[...,j,None]))*(1+.85*e[...,2*j+1,None].tanh()*(2*sign-1))
                # Equal weight to the seven expiries in the auxiliary head; no expiry jitter at inference.
                likelihood=F.softplus(logits)-sign*logits+.5*(size.log()+values.abs().clamp(max=8)/size)
                losses.append(likelihood.mean(-1))
            local=F.smooth_l1_loss(e[...,4],gap[...,3]-gap.mean(-1),beta=.35,reduction="none")
            loss=(w*(.55*losses[0]+.45*losses[1]+.30*local)).sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    torch.save(dict(unit=unit.cpu(),params=model.cpu().state_dict()),OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    data=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    model=pairs.Comparison(len(data["unit"]),effects=5,contexts=2).double().eval()
    model.load_state_dict(data["params"])
    x=pairs.net.features(raw,grid,code,p)
    result=[]
    for start in range(0,len(x),32):
        u=(torch.as_tensor(x[start:start+32]).double()/data["unit"]).clamp(-5,5)
        e,c,peer,w=model(u)
        near=expected(e[...,:2],c[...,0])
        exact=expected(e[...,2:4],c[...,1])
        empty=u.new_zeros(len(u),50,50)
        edges=empty.scatter_add(-1,peer,w)
        adj=(edges+edges.transpose(1,2))*.5
        lap=torch.diag_embed(adj.sum(-1)+.4)-adj
        flows=[]
        for val in (near,exact,near+e[...,4]):
            msg=empty.scatter_add(-1,peer,w*val)
            flows.append(.5*(msg-msg.transpose(1,2)).sum(-1))
        result.append(torch.linalg.solve(lap,torch.stack(flows,-1)).numpy())
    z=np.concatenate(result)
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
