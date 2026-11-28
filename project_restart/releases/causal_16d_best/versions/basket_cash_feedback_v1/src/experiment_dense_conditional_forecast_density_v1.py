"""Estimate a correction from the local density of issued forecast curves in market and book-conditioned histories."""
import json
import sys

import numpy as np
import torch

import experiment_dense_cross_section_transport_v1 as base

OUT=base.REFERENCE.parent/"dense_conditional_forecast_density_v1"
REFERENCE,rel,onset=base.REFERENCE,base.rel,base.onset
ARMS=("same_tick_density","book_conditioned_density","same_stock_history_density")
TARGETS=(1,2)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LAGS=(4,12,28,56,112)
WIDTH=3


def center(x):
    return x-x.mean(1,keepdims=True)


def geometry(end):
    days=[]
    for d in range(4,end):
        q=center(base.read_p(d)[::8].astype(float)).reshape(-1,3)
        days.append(q.T@q/len(q))
    g=np.mean(days,axis=0)
    sd=np.sqrt(g.diagonal().clip(.0001))
    val,v=np.linalg.eigh(g/np.outer(sd,sd))
    return v/np.sqrt(val.clip(0)+.35)/sd[:,None]


@torch.inference_mode()
def design(raw,grid,p,axes):
    axes=torch.as_tensor(axes,device=DEVICE,dtype=torch.float64)
    identity=torch.eye(50,device=DEVICE,dtype=torch.bool)
    for a,b in rel.fields.parent.sessions(grid):
        q=torch.as_tensor(center(p[a:b].astype(float)),device=DEVICE)
        u=q@axes
        v=np.tanh(.65*np.arcsinh(raw[a:b,:,[41,42,54,55,11,0]].astype(np.float32)))
        book=np.stack((v[...,0]-v[...,1],v[...,2]-v[...,3],v[...,4],v[...,5]),axis=-1)
        book=base.panel.smooth(book,12)
        context=torch.as_tensor(book,dtype=u.dtype,device=DEVICE)
        for start in range(0,b-a,64):
            t=torch.arange(start,min(start+64,b-a),device=DEVICE)
            f=u[t]
            diff=f[:,None,:,:]-f[:,:,None,:]
            logits=-.5*diff.square().sum(-1)/.8**2
            logits=logits.masked_fill(identity,-1e5)
            w=logits.softmax(-1)
            covariate=(context[t,None,:,:]-context[t,:,None,:]).square().sum(-1)
            wb=(logits-3*covariate).softmax(-1)
            shift=w@q[t]-q[t]
            conditioned=(wb-w)@q[t]
            # Pool only previously issued curves of the same stock and stay inside the session.
            old=t[:,None]-torch.as_tensor(LAGS,device=DEVICE)
            past=u[old.clamp_min(0)].transpose(1,2)
            l=-(past-f[:,:,None,:]).square().sum(-1)/1.28
            history=l.masked_fill(old[:,None,:]<0,-1e5).softmax(-1)
            previous=q[old.clamp_min(0)].transpose(1,2)
            drift=(history[:,:,:,None]*(previous-q[t,:,None,:])).sum(2)
            drift=torch.where((old>=0).any(1)[:,None,None],drift,0.)
            z=torch.stack((shift,conditioned,drift),axis=2)
            z=z-z.mean(1,keepdim=True)
            yield a+t.cpu().numpy(),q[t].cpu().numpy(),z.cpu().numpy()


def train(end, designer=design):
    axes=geometry(end)
    grams=[]
    for d in range(4,end):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=base.read_p(d)
        b3,b2=base.panel.calibration([base.panel.phase_stats(p,y)])
        target=center(y[...,1:].astype(float)/np.array([b2,b3])-p[...,1:])
        daily=np.zeros((len(ARMS),8,8))
        n=0
        for t,q,z in designer(raw,grid,p,axes):
            take=t%6==d%6
            if not take.any():
                continue
            for j in range(len(ARMS)):
                ex=np.concatenate((q[take],z[take,:,j],target[t[take]]),axis=-1).reshape(-1,8)
                daily[j]+=ex.T@ex
            n+=take.sum()*50
        grams.append(daily/n)
    rows=np.stack(grams)
    coeff=[]
    for j in range(len(ARMS)):
        day=rows[:,j]
        g=day.mean(0)
        transform=np.eye(6)[:,3:]
        transform[:3]=-np.linalg.solve(g[:3,:3]+1e-5*np.eye(3),g[:3,3:6])
        cov=transform.T@g[:6,:6]@transform
        aw=base.previous.whitening(cov,.9)
        v=transform@aw
        pay=v.T@day[:,:6,6:]
        mean=pay.mean(0)
        stable=mean*mean**2/(mean**2+pay.var(0)/len(pay)+1e-8)
        coeff.append(v@stable)
    return np.r_[axes.ravel(),np.stack(coeff).ravel()]


def signals(raw,grid,p,bank,designer=design):
    axes=bank[:9].reshape(3,3)
    co=bank[9:].reshape(3,6,2)
    out=np.zeros((*p.shape[:2],3,2))
    for t,q,z in designer(raw,grid,p,axes):
        for j in range(3):
            out[t,:,j]=np.concatenate((q,z[:,:,j]),axis=-1)@co[j]
    return out


def scales(history):
    strengths=(0.,.1,.25,.5,.8,1.2)
    result=np.zeros((3,2))
    for j in range(3):
        for h in range(2):
            merits=[]
            for s in strengths:
                diag=np.diag([1.,1.,1.,s])
                changes=np.array([rel.ref.market.gain(diag@g[j,h]@diag) for g in history])
                merits.append(changes.mean()-.5*changes.std()/np.sqrt(len(changes)))
            result[j,h]=strengths[np.argmax(merits)]
    return result


def rolling(model):
    model.OUT.mkdir(exist_ok=True)
    read_forecast=getattr(model,"read_p",base.read_p)
    history=[];stats={(j,h):[] for j in range(3) for h in range(2)}
    for d in range(6,16):
        if d in (6,8,12):
            bank=model.train(d)
            if history:
                scale=scales(history)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=read_forecast(d)
        z=model.signals(raw,grid,p,bank)
        if d==8:
            for n in (240,2451):
                np.testing.assert_allclose(z[:n],model.signals(raw[:n],grid[:n],p[:n],bank),atol=2e-7,rtol=0)
        rec=np.stack([[rel.ref.market.metric(p[...,h+1],y[...,h+1],z[...,j,h]) for h in range(2)] for j in range(3)])
        if d>=8:
            for key,rows in stats.items():
                diag=np.diag([1.,1.,1.,scale[key]])
                rows.append(diag@rec[key]@diag)
        history.append(rec)
    rep={};sel={}
    final=scales(history)
    for h in range(2):
        rep[str(h+1)]={name:base.panel.score(stats[j,h]) for j,name in enumerate(model.ARMS)}
        good=[]
        for j,name in enumerate(model.ARMS):
            v=rep[str(h+1)][name]
            if min(v["folds"]+v["omit_one"])>1e-12 and sum(g>1e-12 for g in v["days"])>=5 and final[j,h]>0:
                good.append(name)
        if good:
            sel[str(h+1)]=max(good,key=lambda name:min(rep[str(h+1)][name]["folds"]))
    (model.OUT/"rolling.json").write_text(json.dumps(dict(selected=sel,results=rep,strengths=final.tolist()),indent=2))
    if sel:
        choices=np.full(2,-1)
        for k,name in sel.items():
            choices[int(k)-1]=model.ARMS.index(name)
        np.savez(model.OUT/"weights.npz",bank=model.train(16),choices=choices,strengths=final)
    print(sel,{h:{k:v["folds"] for k,v in r.items()} for h,r in rep.items()},flush=True)


def apply(model,raw,grid,p,params):
    q=p.copy()
    z=model.signals(raw,grid,p,params["bank"])
    for h,j in enumerate(params["choices"]):
        if j>=0:
            q[...,h+1]+=(params["strengths"][j,h]*z[...,j,h]).astype(q.dtype)
    return q


def predict(raw,grid,p,params):
    return apply(sys.modules[__name__],raw,grid,p,params)


if __name__=="__main__":
    torch.set_num_threads(1)
    rolling(sys.modules[__name__])
