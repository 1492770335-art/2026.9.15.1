"""Share magnitude-dependent direction across forecast horizons while retaining target-specific departures."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_settled_error_gru_v1 as settled
from experiment_dense_rank_risk_calibration_v1 import main as rolling

rel, REFERENCE = settled.rel, settled.REFERENCE
OUT = REFERENCE.parent/"dense_shared_rank_deviations_v1"
ARMS=("shared_direction_shape","specific_direction_shape","complete_direction_shape")


class HorizonRanks(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,28),nn.SiLU())
        self.mass=nn.Linear(28,15)
        self.sign=nn.Linear(28,3)
        self.shape=nn.Linear(28,5)
        self.departure=nn.Linear(28,15)

    def forward(self,x):
        f=self.encoder(x)
        mass=self.mass(f).reshape(*x.shape[:2],3,5).softmax(-1)
        for _ in range(4):
            mass=mass*10/mass.sum(1,keepdim=True)
            mass=mass/mass.sum(-1,keepdim=True)
        shape=self.shape(f).unsqueeze(2)
        diff=self.departure(f).reshape(*x.shape[:2],3,5)
        return mass,self.sign(f).unsqueeze(-1),shape,.6*(diff-diff.mean(2,keepdim=True))


def train(end):
    torch.manual_seed(15649)
    device="cuda" if torch.cuda.is_available() else "cpu"
    rows=[]
    for day in range(4,end):
        with np.load(settled.OUT/f"day_{day:02d}.npz") as f:
            inputs=np.concatenate((f["x"],f["seq"][:,:,-1]),-1)
        p=np.load(REFERENCE/f"rolling_{day:02d}.npy") if day>=6 else settled.head.ranks.band.pairs.net.future.forecast(day)
        p=p.astype(float)-p.mean(1,keepdims=True,dtype=float)
        y=settled.load(day,"y").astype(float)
        y-=y.mean(1,keepdims=True)
        slope=(np.sum(p*y,axis=(0,1))/np.sum(p*p,axis=(0,1))).clip(.2,4)
        grid=settled.load(day,"time")
        t=np.concatenate([np.arange(a+40+day%5,b-61,6) for a,b in rel.fields.parent.sessions(grid)])
        rows.append((inputs,y[t]/slope-p[t]))
    x,y=[torch.as_tensor(np.concatenate(a),device=device,dtype=torch.float32) for a in zip(*rows)]
    unit=x.square().mean((0,1)).sqrt().clamp_min(.04)
    x=(x/unit).clamp(-5,5)
    y=y/y.square().mean((0,1)).sqrt().clamp_min(.05)
    rank=torch.stack([settled.head.ranks.rank_targets(v) for v in y.unbind(-1)],2)
    levels=(rank*y.unsqueeze(-1)).sum((0,1))/rank.sum((0,1))
    span=(levels[:,5:].flip(-1)-levels[:,:5])*.5
    pos,neg=rank[...,5:].flip(-1),rank[...,:5]
    net=HorizonRanks(len(unit)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.045)
    for _ in range(6):
        for ids in torch.randperm(len(x),device=device).split(40):
            mass,base,shared,specific=net(x[ids])
            def cost(v):
                return ((pos[ids]*F.softplus(-v)+neg[ids]*F.softplus(v))*span).sum(-1).mean()
            loss=cost(base+shared+specific)+.4*cost(base+shared)+.25*cost(base)
            loss-=.6*((pos[ids]+neg[ids])*mass.log()).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
    torch.save(dict(unit=unit.cpu(),state=net.cpu().state_dict(),span=span.cpu()),OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def design(raw,grid,code,p,bank):
    pack=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    net=HorizonRanks(len(pack["unit"])).double().eval()
    net.load_state_dict(pack["state"])
    book=settled.head.ranks.band.pairs.net.features(raw,grid,code,p)
    errors,_=settled.observations(raw,grid,p)
    x=np.concatenate((book,errors),-1)
    results=[]
    for v in np.array_split(x,(len(x)+63)//64):
        u=(torch.as_tensor(v).double()/pack["unit"]).clamp(-5,5)
        m,b,s,d=net(u)
        def expected(logits):
            prob=settled.head.upper_probability(m,logits)
            return (m*pack["span"]*prob).sum(-1)
        origin=expected(b)
        pooled=expected(b+s)
        full=expected(b+s+d)
        results.append(torch.stack((pooled-origin,full-pooled,full-origin),2).numpy())
    z=np.concatenate(results)
    return z-z.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    z=design(raw,grid,params["code"],p,params["bank"])
    q=p.copy()
    for h in range(3):
        q[:,:,h]+=(rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    rolling(OUT,ARMS,train,design,REFERENCE)
