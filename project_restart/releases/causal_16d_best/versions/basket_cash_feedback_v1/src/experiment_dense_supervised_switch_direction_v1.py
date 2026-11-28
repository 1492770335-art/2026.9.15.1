"""Train matched book-only and timing-augmented predictors directly on the sign and size of Y3 errors."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_switch_timing_scenarios_v1 as timing

onset,clock,rel,REFERENCE=timing.onset,timing.clock,timing.rel,timing.REFERENCE
OUT=REFERENCE.parent/"dense_supervised_switch_direction_v1"
ARMS=("book_only","book_and_timing","increment_from_timing")
USES_FORECAST=True


def stage_one():
    OUT.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    with np.load(onset.settled.GEO) as f:
        missing=f["row"]
    for d in range(16):
        raw,grid=[onset.settled.load(d,k) for k in ("x","time")]
        counts=np.zeros((clock.N,4))
        for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
            _,keys=clock.keys(v,s,start)
            visible=~np.all(raw[a:b]==missing,axis=-1)
            gaps=np.vstack((np.zeros((1,50),int),(~visible).cumsum(0)))
            ix=np.arange(max(0,b-a-60))
            ticks=np.arange(b-a)[:,None,None]
            changed=np.concatenate((s[1:]!=s[:-1],np.zeros_like(s[:1],bool)))
            nxt=np.minimum.accumulate(np.where(changed,ticks+1,b-a+61)[::-1],axis=0)[::-1]
            cat=np.digitize(nxt[ix]-ix[:,None,None],[5,20,60],right=True)
            use=(start[ix]>0)&(s[ix]!=0)&(gaps[ix+61]==gaps[ix])[:,:,None]
            counts+=np.bincount((keys[ix]*4+cat)[use],minlength=clock.N*4).reshape(-1,4)
        np.save(OUT/f"counts_{d:02d}.npy",counts)


def table(end):
    c=sum(np.load(OUT/f"counts_{d:02d}.npy") for d in range(end)).reshape(*clock.DIMS,4)
    def probs(x):
        return (x+.5)/(x.sum(-1,keepdims=True)+2)
    pooled=probs(c.sum(0,keepdims=True))
    usual=probs(c.sum((4,5),keepdims=True))
    smoothed=c+48*pooled+12*usual
    return (smoothed/smoothed.sum(-1,keepdims=True)).reshape(-1,4)


def inputs(raw,grid,code,p,end,training_day=None):
    rates=clock.rates(end)
    # The first-stage model is estimated only on dates preceding the feature observation.
    rates4=table(end)
    keys=np.empty((*raw.shape[:2],5),dtype=np.int32)
    for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
        keys[a:b]=clock.keys(v,s,start)[1]
    for t,x,g,side,_,cat in timing.chunks(raw,grid,code,p,rates,training_day):
        probs=rates4[keys[t]]
        # Current validity must be identical at train and test. Future visibility is used for
        # the first-stage training labels only; it never gates the supervised trading signal.
        valid=(x[...,7]>0)&(side!=0)
        yield t,(x*valid[...,None]).astype(np.float32),((probs[...,:3].cumsum(-1)-.5)*valid[...,None]).astype(np.float32)


def collect():
    code=rel.fields.parent.codec()
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=(np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        q=p[:,:,2].astype(float)-p[:,:,2].mean(1,keepdims=True,dtype=float)
        actual=y[:,:,2].astype(float)-y[:,:,2].mean(1,keepdims=True,dtype=float)
        scale=((q*actual).sum()/(q*q).sum()).clip(.2,4)
        rows=[(x.astype(np.float16),pr.astype(np.float16),(actual[t]/scale-q[t]).astype(np.float32))
              for t,x,pr in inputs(raw,grid,code,p,d,d)]
        np.savez(OUT/f"sample_{d:02d}.npz",**{k:np.concatenate(v) for k,v in zip(("x","pr","y"),zip(*rows))})


class Direction(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.fields=nn.Sequential(nn.Linear(width,24),nn.LayerNorm(24),nn.SiLU(),nn.Linear(24,10),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(50,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,2))

    def forward(self,x):
        h=self.fields(x).flatten(-2)
        logit,radius=self.head(h).unbind(-1)
        return logit,F.softplus(radius)+.02


def train(end):
    device="cuda" if torch.cuda.is_available() else "cpu"
    rows=[]
    for d in range(4,end):
        with np.load(OUT/f"sample_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("x","pr","y")))
    x,pr,y=[torch.from_numpy(np.concatenate(v)).to(device).float() for v in zip(*rows)]
    feature=torch.cat((x,pr),-1)
    unit=feature.square().mean((0,1,2)).sqrt().clamp_min(.025)
    feature=(feature/unit).clamp(-5,5)
    y=y/y.square().mean().sqrt().clamp_min(.04)
    pack=dict(unit=unit.cpu(),end=end)
    for variant in ("book","timing"):
        torch.manual_seed(53191)
        net=Direction(len(unit)).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.06)
        for _ in range(5):
            for ids in torch.randperm(len(x),device=device).split(48):
                u=feature[ids]
                if variant=="book":
                    u[...,-3:]=0
                yy=y[ids]
                logit,rad=net(u)
                loss=F.binary_cross_entropy_with_logits(logit,(yy>0).float(),weight=yy.abs().clamp(max=4))
                loss+=.3*F.smooth_l1_loss(rad, yy.abs(),beta=.7)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1)
                opt.step()
        pack[variant]=net.cpu().state_dict()
    torch.save(pack,OUT/f"net_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def views(raw,grid,code,bank,p):
    end=int(bank[0])
    pack=torch.load(OUT/f"net_{end}.pt",weights_only=True,map_location="cpu")
    nets=[]
    for k in ("book","timing"):
        model=Direction(len(pack["unit"])).double().eval()
        model.load_state_dict(pack[k])
        nets.append(model)
    z=np.zeros((*p.shape[:2],3))
    for t,x,pr in inputs(raw,grid,code,p,end):
        u=(torch.from_numpy(np.concatenate((x,pr),-1)).double()/pack["unit"]).clamp(-5,5)
        aug,ar=nets[1](u)
        u[...,-3:]=0
        plain,br=nets[0](u)
        full=(2*aug.sigmoid()-1)*ar
        basic=(2*plain.sigmoid()-1)*br
        z[t]=torch.stack((basic,full,full-basic),dim=-1).numpy()
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    stage_one()
    collect()
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
