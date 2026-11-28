"""Price three future twenty-tick residuals using their joint sign path and fitted conditional sizes."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_return_unit_quantiles_v1 as units

onset, rel, REFERENCE = units.onset, units.rel, units.REFERENCE
basket = units.curves.basket
OUT = REFERENCE.parent/"dense_joint_block_paths_v1"
ARMS = ("independent_block_signs","joint_path_payoff","dependence_increment")
BITS = ((np.arange(8)[:,None] >> np.arange(3)) & 1).astype(np.float32)


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    with np.load(onset.settled.GEO) as geom:
        k,coef,missing = float(geom["k"]),geom["coeff"],geom["row"]
    report = []
    for d in range(4,16):
        raw,grid,labels = [onset.settled.load(d,n) for n in ("x","time","y")]
        p = np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d)
        x = basket.features(raw,grid,code,p,basket.front.direction.table(d))
        rows = []
        for a,b in rel.fields.parent.sessions(grid):
            t = a+np.arange(32+d%4,b-a-60,4)
            y = np.stack([labels[t+lag,:,1] for lag in (0,20,40)],axis=-1).astype(float)
            moves = np.log1p(k*(y-coef[1,0])/coef[1,1])*coef[2,1]/k
            true = labels[t,:,2].astype(float)
            recon = coef[2,0]+np.expm1(moves.sum(-1)*k/coef[2,1])*coef[2,1]/k
            gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
            valid = gaps[t-a+61]==gaps[t-a]
            def relative(z):
                return (z-(z*valid[...,None]).sum(1,keepdims=True)/valid.sum(1)[:,None,None].clip(1))*valid[...,None]
            rows.append((x[t],relative(p[t].astype(float)),relative(moves),valid,abs(recon-true)[valid]))
        x,q,y,mask,discrepancy = [np.concatenate(v) for v in zip(*rows)]
        flat = q[mask]
        gram = flat.T@flat/len(flat)
        w = np.linalg.solve(gram+.015*np.diag(gram.diagonal().clip(.001)),flat.T@y[mask]/len(flat))
        err = y-q@w
        corr = np.corrcoef(np.column_stack((flat[:,2],y[mask])).T)[0,1:]
        report.append(dict(day=d+1,valid_rows=int(mask.sum()),block_ic=corr.tolist(),
                           median_and_p99_reconstruction_difference=np.quantile(discrepancy,[.5,.99]).tolist()))
        np.savez(OUT/f"day_{d:02d}.npz",x=x.astype(np.float16),target=err.astype(np.float32),valid=mask)
    (OUT/"block_audit.json").write_text(json.dumps(report,indent=2))


class PathHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.names = nn.Embedding(50,5)
        self.encoder = nn.Sequential(nn.Linear(width*2+5,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.signs = nn.Linear(24,8)
        self.size = nn.Linear(24,6)
        self.interaction = nn.Linear(24,24)
        self.register_buffer("bit",torch.from_numpy(BITS))

    def forward(self,x):
        names = self.names.weight.expand(len(x),-1,-1)
        f = self.encoder(torch.cat((x,x.mean(1,keepdim=True).expand_as(x),names),dim=-1))
        prob = self.signs(f).softmax(-1)
        signed = 2*self.bit-1
        base = self.size(f).reshape(len(x),50,1,3,2)
        size = base[...,0]*(1-self.bit)+base[...,1]*self.bit
        extra = self.interaction(f).reshape(len(x),50,8,3)
        # Within a given sign of block j the average path interaction is fixed to zero.
        extra = extra-torch.stack([torch.where(self.bit[:,j]>0,extra[...,self.bit[:,j]>0,j].mean(-1,keepdim=True),
                                              extra[...,self.bit[:,j]==0,j].mean(-1,keepdim=True)) for j in range(3)],-1)
        value = (.02+F.softplus(size+.5*extra))*signed
        marginal = prob@self.bit
        independent = torch.where(self.bit.bool(),marginal.unsqueeze(-2),1-marginal.unsqueeze(-2)).prod(-1)
        return prob,independent,value


def train(end):
    torch.manual_seed(90281)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[n] for n in ("x","target","valid")))
    x,y,valid = [torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for v in zip(*rows)]
    scale = x.square().mean((0,1)).sqrt().clamp_min(.04)
    unit = (y.square().sum((0,1))/valid.sum()).sqrt().clamp_min(.05)
    x = (x/scale).clamp(-6,6)
    y = y/unit
    path = ((y>0).long()*torch.tensor([1,2,4],device=device)).sum(-1)
    net = PathHead(len(scale)).to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.06)
    for _ in range(7):
        for t in torch.randperm(len(x),device=device).split(64):
            pr,ind,val = net(x[t])
            which = path[t]
            choice = val.gather(2,which[...,None,None].expand(-1,-1,1,3)).squeeze(2)
            partial = (pr.unsqueeze(-1)*val).sum(2)
            loss = -pr.gather(-1,which.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8).log()
            loss += .55*((choice-y[t].clamp(-7,7)).square()).mean(-1)
            loss += .25*((partial-y[t].clamp(-7,7)).square()).mean(-1)
            loss = (loss*valid[t]).sum()/valid[t].sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
    torch.save(dict(state=net.cpu().state_dict(),scale=scale.cpu(),unit=unit.cpu(),
                    rate=torch.from_numpy(basket.front.direction.table(end))),OUT/f"fit_{end}.pt")
    return np.array([end])


def load(end):
    pack = torch.load(OUT/f"fit_{end}.pt",map_location="cpu",weights_only=True)
    net = PathHead(len(pack["scale"])).double().eval()
    net.load_state_dict(pack["state"])
    return net,pack


@torch.inference_mode()
def estimate(x,net,pack):
    full,independent,parts = [],[],[]
    for chunk in np.array_split(x,max(1,(len(x)+95)//96)):
        q = (torch.from_numpy(chunk).double()/pack["scale"]).clamp(-6,6)
        prob,ind,value = net(q)
        value = value*pack["unit"]
        full.append((prob[...,None]*value).sum(2).numpy())
        independent.append((ind[...,None]*value).sum(2).numpy())
        parts.append(torch.stack((prob,ind),dim=-1).numpy())
    return np.concatenate(full),np.concatenate(independent),np.concatenate(parts)


def design(raw,grid,code,p,bank):
    net,pack = load(int(bank[0]))
    x = basket.features(raw,grid,code,p,pack["rate"].numpy())
    full,ind,_ = estimate(x,net,pack)
    def read(z):
        return np.stack((z[...,0],z[...,0],z.sum(-1)),axis=-1)
    z = np.stack((read(ind),read(full),read(full-ind)),axis=2)
    return z-z.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


def audit():
    report = {}
    for end in (8,12):
        net,pack = load(end)
        rows = []
        for d in range(end,end+4):
            with np.load(OUT/f"day_{d:02d}.npz") as f:
                x,y,m = f["x"],f["target"],f["valid"]
            full,ind,prob = estimate(x,net,pack)
            idx = ((y[m]>0)*np.array([1,2,4])).sum(-1)
            nll = -np.log(prob[m][np.arange(len(idx)),idx].clip(1e-10)).mean(0)
            corr = [np.corrcoef(full[...,j][m],y[...,j][m])[0,1] for j in range(3)]
            rows.append([*nll,*corr])
        report[str(end+1)] = dict(metrics=["joint_nll","independent_nll","near_ic","middle_ic","late_ic"],
                                   mean=np.mean(rows,axis=0).tolist(),daily=rows)
    (OUT/"path_diagnostics.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
    audit()
