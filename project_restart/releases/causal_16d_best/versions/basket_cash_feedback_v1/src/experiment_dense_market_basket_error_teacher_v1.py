"""Train a pooled market head on complementary stock baskets with a cross-fitted common-return estimator."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_market_switch_front_v1 as front

onset,clock,rel,REFERENCE=front.onset,front.clock,front.rel,front.REFERENCE
OUT=REFERENCE.parent/"dense_market_basket_error_teacher_v1"
ARMS=("other_basket","other_above_same","precision_weighted_basket")


def features(raw,grid,code,p,rate):
    parts=[]
    for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
        age=np.arange(b-a)[:,None,None]-start
        _,key=clock.keys(v,s,start)
        prob=rate[key]
        valid=(start>0)&(s!=0)
        new=s*(age<6)*valid
        stable=s*prob[...,3]*valid
        excess=new-onset.book.smooth(new,24)
        slow=onset.book.smooth(v,40)
        q=p[a:b].copy()
        q[:,:,0]-=q[:,:,0].mean(1,keepdims=True)
        parts.append(np.concatenate((q,v,s*valid,excess,stable,v-slow,slow),axis=-1).astype(np.float32))
    return np.concatenate(parts)


def collect():
    OUT.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=(np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        q=p.astype(float)-p.mean((0,1),keepdims=True,dtype=float)
        y=y.astype(float)-y.mean((0,1),keepdims=True,dtype=float)
        slope=((q*y).sum((0,1))/(q*q).sum((0,1))).clip(.2,4)
        err=y/slope-q
        t=np.concatenate([np.arange(a+30+d%4,b-60,4) for a,b in rel.fields.parent.sessions(grid)])
        x=features(raw,grid,code,p,front.direction.table(d))[t]
        # Keep all common Y2/Y3 drift. Center Y1 across names to match the rolling reference convention.
        err[...,0]-=err[...,0].mean(1,keepdims=True)
        np.savez(OUT/f"day_{d:02d}.npz",x=x.astype(np.float16),y=err[t].astype(np.float32))


def reliability(y):
    y=y-y.mean(0)
    # Estimate loadings against a basket that contains none of the name being scored.
    left,right=y[:,::2].mean(1),y[:,1::2].mean(1)
    var=(left*right).mean(0).clip(.006)
    load=np.ones((50,3))
    for k,other in enumerate((right,left)):
        load[k::2]=(y[:,k::2]*other[:,None]).mean(0)/var
    load=(.6*load+.4).clip(.25,2.75)
    noise=(np.square(y).mean(0)-load*load*var).clip(.025)
    weight=load/(.4*noise.mean(0)+.6*noise)
    return np.stack((weight,load*weight),-1).astype(np.float32)


class MarketHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(width,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,20),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(60,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,9))
        self.vote=nn.Linear(20,20)

    def forward(self,x,src):
        h=self.encode(x)
        w=src[...,None]/src.sum(1,keepdim=True)[...,None]
        mean=(h*w).sum(1)
        spread=((h-mean[:,None]).square()*w).sum(1).clamp_min(1e-6).sqrt()
        vote=(self.vote(h).tanh()*w).sum(1)
        return self.value(torch.cat((mean,spread,vote),-1)).reshape(-1,3,3)


def train(end):
    torch.manual_seed(23813)
    device="cuda" if torch.cuda.is_available() else "cpu"
    xs=[];ys=[];dates=[]
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            xs.append(f["x"]);ys.append(f["y"]);dates.append(np.full(len(f["y"]),d%2))
    day=np.concatenate(dates)
    yy=np.concatenate(ys)
    risks=torch.as_tensor(np.stack([reliability(yy[day!=k]) for k in (0,1)]),device=device)
    day=torch.from_numpy(day).to(device)
    x=torch.from_numpy(np.concatenate(xs)).to(device).float()
    y=torch.from_numpy(yy).to(device)
    scale=x.square().mean((0,1)).sqrt().clamp_min(.03)
    x=(x/scale).clamp(-5,5)
    unit=y.square().mean((0,1)).sqrt().clamp_min(.05)
    y=y/unit
    model=MarketHead(x.shape[-1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.065)
    for _ in range(7):
        for ids in torch.randperm(len(x),device=device).split(80):
            src=torch.rand((len(ids),50),device=device).argsort(1)<25
            hidden=(~src)[...,None]
            donor=risks[day[ids]]
            denom=(donor[...,1]*hidden).sum(1)
            proxy=(y[ids]*donor[...,0]*hidden).sum(1)/denom.clamp_min(1)
            target=torch.stack(((y[ids]*src[...,None]).sum(1)/25,(y[ids]*hidden).sum(1)/25,proxy),axis=1)
            pred=model(x[ids],src.float())
            loss=F.smooth_l1_loss(pred,target,beta=.6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
    torch.save(dict(weights=model.cpu().state_dict(),unit=unit.cpu(),scale=scale.cpu(),
               rate=torch.from_numpy(front.direction.table(end))),OUT/f"fit_{end}.pt")
    return np.array([end])


@torch.inference_mode()
def design(raw,grid,code,p,bank):
    pack=torch.load(OUT/f"fit_{int(bank[0])}.pt",map_location="cpu",weights_only=True)
    net=MarketHead(len(pack["scale"])).double().eval()
    net.load_state_dict(pack["weights"])
    x=features(raw,grid,code,p,pack["rate"].numpy())
    masks=np.stack([np.random.default_rng(107+j).permutation(50)<25 for j in range(3)])
    masks=torch.from_numpy(np.vstack((masks,~masks))).double()
    rows=[]
    for chunk in np.array_split(x,(len(x)+95)//96):
        u=(torch.from_numpy(chunk).double()/pack["scale"]).clamp(-5,5)
        pred=torch.stack([net(u,m.expand(len(u),-1)) for m in masks]).mean(0)*pack["unit"]
        same,other,precision=pred[:,:,2].unbind(1)
        rows.append(torch.stack((other,other-same,precision-other),-1).numpy())
    return np.concatenate(rows)


def scale(hist):
    mats=np.stack([m/m[:,0:1,0:1] for m in hist])
    c=mats-mats[:,:,:,0,None]*mats[:,:,None,0,:]
    slope=(c[:,:,1,2]/c[:,:,1,1]).clip(.2,4)
    grad=c[:,:,3,2]/slope-c[:,:,3,1]
    mu=grad.mean(0)
    var=c[:,:,3,3].mean(0).clip(1e-8)
    gain=mu*mu**2/(mu**2+grad.var(0,ddof=1)/len(c)+1e-10)/var
    cap=.05*np.sqrt(c[:,:,1,1].mean(0)/var)
    return np.clip(.6*gain,-cap,cap)


def rolling(out,arms,train,design,reference=REFERENCE):
    out.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    hist=[];rows={k:[] for k in arms}
    for d in range(6,16):
        if d in (6,8,12):bank=train(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=np.load(reference/f"rolling_{d:02d}.npy")
        z=design(raw,grid,code,p,bank)
        if d==8:
            np.testing.assert_allclose(z[:512],design(raw[:512],grid[:512],code,p[:512],bank),atol=1e-7,rtol=0)
        mats=np.stack([rel.ref.market.metric(p[...,2],y[...,2],np.broadcast_to(z[:,j,None],p.shape[:2])) for j in range(len(arms))])
        if d in (8,12):coef=scale(hist)
        if d>=8:
            for j,k in enumerate(arms):
                tr=np.diag([1.,1.,1.,coef[j]])
                rows[k].append(tr@mats[j]@tr)
        hist.append(mats)
    report={k:front.score(m) for k,m in rows.items()}
    good=[k for k,v in report.items() if min(v["folds"]+v["omit_one"])>0 and sum(z>0 for z in v["days"])>=5]
    best=max(good,key=lambda k:min(report[k]["folds"])) if good else None
    (out/"rolling.json").write_text(json.dumps(dict(selected={"2":best} if best else {},results=report),indent=2))
    if best:
        w=np.zeros(len(arms));j=arms.index(best);w[j]=scale(hist)[j]
        np.savez(out/"weights.npz",code=code,bank=train(16),w=w)
    print(best,{k:v["folds"] for k,v in report.items()},flush=True)


def predict(raw,grid,p,params):
    z=design(raw,grid,params["code"],p,params["bank"])
    q=p.copy()
    q[...,2]+=(z@params["w"])[:,None].astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    rolling(OUT,ARMS,train,design)
