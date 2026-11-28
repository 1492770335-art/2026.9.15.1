"""Predict first-reversal timing and marginalize its conditional return value across five pressure channels."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_switch_landmark_response_v1 as panel

clock,onset,rel,REFERENCE = panel.clock,panel.onset,panel.rel,panel.REFERENCE
OUT = REFERENCE.parent/"dense_switch_timing_scenarios_v1"
ARMS = ("first_five_tick_timing", "later_timing", "whole_timing_distribution")
DIMS = (5,2,4,3)


def chunks(raw,grid,code,p,rates,training_day=None):
    with np.load(onset.settled.GEO) as f:
        missing = f["row"]
    for a,b,v,s,start,age,prior,use,moves,cash,q in panel.sessions(raw,grid,code,p,rates):
        _,keys = clock.keys(v,s,start)
        chance = rates[keys,0]
        change = np.concatenate((np.zeros_like(v[:1]),np.diff(v,axis=0)))
        if training_day is not None:
            t = np.arange(22+training_day%6,b-a-60,6)
            idx = np.arange(b-a)[:,None,None]
            switch = np.concatenate((s[1:]!=s[:-1],np.zeros_like(s[:1],bool)))
            nxt = np.minimum.accumulate(np.where(switch,idx+1,b-a+61)[::-1],axis=0)[::-1]
            gaps = np.vstack((np.zeros((1,50),int),np.all(raw[a:b]==missing,axis=2).cumsum(0)))
        else:
            t = np.arange(b-a)
        for ix in np.array_split(t,max(1,(len(t)+63)//64)):
            cell,side,_,trajectory = panel.features(ix,v,s,start,age,prior,use,moves,cash,q)
            valid = (s[ix]!=0)&(start[ix]>0)
            f = np.concatenate((trajectory, (np.log1p(age[ix])/4)[...,None],side[...,None],
                      np.log(chance[ix].clip(.0001))[...,None]/4, (side*change[ix])[...,None],
                      (side*s[ix].mean(2,keepdims=True))[...,None]),axis=-1)
            groups = np.ravel_multi_index(np.broadcast_arrays(np.arange(5)[None,None,:],(side>0).astype(int),
                               np.digitize(age[ix],[4,15,40]),np.digitize(chance[ix],[.025,.15])),DIMS)
            if training_day is None:
                yield a+ix,f,groups,side,valid,None
            else:
                timing = np.digitize(nxt[ix]-ix[:,None,None],[5,20,60],right=True)
                valid &= (gaps[ix+61]==gaps[ix])[:,:,None]
                yield a+ix,f,groups,side,valid,timing


def collect():
    OUT.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=(np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        q=p-p.mean(1,keepdims=True,dtype=float)
        truth=y-y.mean(1,keepdims=True,dtype=float)
        slope=((q*truth).sum((0,1))/(q*q).sum((0,1))).clip(.2,4)
        err=truth/slope-q
        rows=[]
        for t,x,group,side,valid,observed in chunks(raw,grid,code,p,clock.rates(d),d):
            rows.append((x.astype(np.float16),group.astype(np.uint8),side.astype(np.int8),valid,observed.astype(np.uint8),
                         err[t].astype(np.float32)))
        np.savez(OUT/f"day_{d:02d}.npz",**{k:np.concatenate(v) for k,v in zip(
                    ("x","group","side","valid","category","error"),zip(*rows))})


class TimedValue(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.field = nn.Embedding(5,5)
        self.encode = nn.Sequential(nn.Linear(width+5,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,20),nn.SiLU())
        self.time = nn.Linear(20,4)
        self.base = nn.Linear(20,3)
        self.value = nn.Linear(20,9)

    def forward(self,x,prior):
        field=self.field.weight.expand(*x.shape[:-2],5,5)
        f=self.encode(torch.cat((x,field),axis=-1))
        probs=(prior.log()+self.time(f)).softmax(-1)
        return probs,self.base(f),self.value(f).reshape(*x.shape[:-1],3,3)


def train(end):
    torch.manual_seed(12797)
    device="cuda" if torch.cuda.is_available() else "cpu"
    rows=[]
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("x","group","side","valid","category","error")))
    x,g,s,ok,c,y=[torch.from_numpy(np.concatenate(col)).to(device) for col in zip(*rows)]
    x,s=x.float(),s.float()
    g,c=g.long(),c.long()
    prior=torch.bincount((4*g[ok]+c[ok]),minlength=int(np.prod(DIMS))*4).reshape(-1,4).float()+1
    prior/=prior.sum(-1,keepdim=True)
    unit=x.square().mean((0,1,2)).sqrt().clamp_min(.03)
    x=(x/unit).clamp(-5,5)
    y=y/y.square().mean((0,1)).sqrt().clamp_min(.04)
    model=TimedValue(x.shape[-1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0011,weight_decay=.055)
    for _ in range(5):
        for ix in torch.randperm(len(x),device=device).split(48):
            local=prior[g[ix]]
            probs,base,effect=model(x[ix],local)
            realized=F.one_hot(c[ix],4).cumsum(-1)[...,:3]-local.cumsum(-1)[...,:3]
            pred=base+(realized[...,None]*effect).sum(-2)
            valid=ok[ix].float()
            loss=F.smooth_l1_loss(pred,(s[ix][...,None]*y[ix][:,:,None,:]),beta=1.25,reduction="none").mean(-1)
            ce=-probs.gather(-1,c[ix][...,None]).squeeze(-1).log()
            loss=((loss+.60*ce)*valid).sum()/valid.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
    torch.save(dict(state=model.cpu().state_dict(),unit=unit.cpu(),prior=prior.cpu(),
                    rates=torch.from_numpy(clock.rates(end))),OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def design(raw,grid,code,p,bank):
    pack=torch.load(OUT/f"net_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    model=TimedValue(len(pack["unit"])).double().eval()
    model.load_state_dict(pack["state"])
    out=np.zeros((*p.shape[:2],3,3))
    for t,x,g,s,ok,_ in chunks(raw,grid,code,p,pack["rates"].numpy()):
        u=(torch.from_numpy(x).double()/pack["unit"]).clamp(-5,5)
        prior=pack["prior"].double()[g]
        probs,_,effect=model(u,prior)
        shift=(probs-prior).cumsum(-1)[...,:3]
        value=(shift[...,None]*effect*torch.from_numpy(s*ok)[...,None,None]).sum(2)
        early,late=value[:,:,0],value[:,:,1:].sum(2)
        out[t]=torch.stack((early,late,early+late),2).numpy()
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    z=design(raw,grid,params["code"],p,params["bank"])
    q=p.copy()
    for h in range(3):
        q[:,:,h]+=(rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
