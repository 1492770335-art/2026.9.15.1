"""Update twelve forward price knots after observing five ticks of cash and a new three-horizon forecast."""
import sys

import numpy as np

import experiment_dense_conditional_forecast_density_v1 as evaluation

base,rel,onset,REFERENCE=evaluation.base,evaluation.rel,evaluation.onset,evaluation.REFERENCE
OUT=REFERENCE.parent/"dense_forward_strip_observer_v1"
ARMS=("one_revision","recursive_revisions","subsequent_revision_feedback")
H=np.arange(5,61,5)
ANCHORS=(0,5,20,60)
INTERP=np.stack([np.interp(H,ANCHORS,np.eye(4)[j]) for j in (1,2,3)])
D=15


def center(x):
    return x-x.mean(axis=1,keepdims=True)


def physical():
    with np.load(onset.settled.GEO) as f:
        return float(f["k"]),f["coeff"][:,1].astype(float),f["row"]


def daily_scale(p,y):
    q=center(p.astype(float));v=center(y.astype(float))
    return (q*v).mean((0,1))/(q*q).mean((0,1)).clip(1e-8)


def session(raw,p,scale):
    k,coef,bad=physical()
    cash=np.log1p(k*(raw[...,0].astype(float)-.00136))*coef[2]/k
    ok=~np.all(raw==bad,axis=2)
    # Express every name in the same market-relative price coordinate.
    cash=(cash-(cash*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1))*ok
    price=np.cumsum(cash,axis=0)
    curve=center(p.astype(float))*(scale*coef[2]/coef)
    curve=curve@INTERP
    imbalance=np.tanh(.6*np.arcsinh(raw[...,41].astype(float)))-np.tanh(.6*np.arcsinh(raw[...,42].astype(float)))
    gaps=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    return price,curve,imbalance,gaps


def innovations(curve,prior,settled,pressure,valid):
    # Today's 5 and 20 tick expiries coincided with the old 10 and 25 tick knots.
    extension=prior[...,-1]+prior[...,-1]-prior[...,-2]
    new=np.stack((settled-prior[...,0],settled+curve[...,0]-prior[...,1],
                  settled+curve[...,3]-prior[...,4],settled+curve[...,11]-extension),axis=-1)
    new=np.clip(new,-3,3)*valid[...,None]
    fast=new[...,0:1]
    late=new[...,3:4]
    aligned=np.tanh(3*pressure[...,None]*late)
    return np.concatenate((new,new*np.tanh(4*abs(fast)),new*aligned,curve[...,[0,3,11]]),axis=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        p=base.read_p(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        scale=daily_scale(p,y)
        gram=np.zeros((D+len(H),D+len(H)));count=0
        for a,b in rel.fields.parent.sessions(grid):
            path,curve,press,gaps=session(raw[a:b],p[a:b],scale)
            t=np.arange(5+d%5,b-a-60,5)
            valid=gaps[t+1]==gaps[t-5]
            x=innovations(curve[t],curve[t-5],path[t]-path[t-5],press[t],valid)
            r=np.stack([path[t+h]-path[t] for h in H],axis=-1)
            future_ok=gaps[t+61]==gaps[t]
            u=np.concatenate((x,r-curve[t]),axis=-1)[valid&future_ok]
            gram+=u.T@u
            count+=len(u)
        np.savez(OUT/f"day_{d:02d}.npz",gram=gram/max(count,1),scale=scale)


def train(end):
    grams=[];scales=[]
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            grams.append(f["gram"]);scales.append(f["scale"])
    grams=np.stack(grams)
    g=grams.mean(0)
    # Fit only the part of price and forecast news outside the new forecast itself.
    tr=np.eye(D)[:,:-3]
    tr[-3:]=-np.linalg.solve(g[D-3:D,D-3:D]+1e-6*np.eye(3),g[D-3:D,:D-3])
    cov=tr.T@g[:D,:D]@tr
    v=tr@base.previous.whitening(cov,.7)
    rhs=v.T@grams[:,:D,D:]
    m=rhs.mean(0);m*=m*m/(m*m+rhs.var(0)/len(grams)+1e-8)
    return np.r_[np.mean(scales,axis=0),(v@m).ravel()]


def signals(raw,grid,p,bank):
    _,coef,_=physical()
    scale=bank[:3]
    w=bank[3:].reshape(D,len(H))
    out=np.zeros((*p.shape[:2],3,2))
    unit=scale[1:]*coef[2]/coef[1:]
    for a,b in rel.fields.parent.sessions(grid):
        path,curve,press,gaps=session(raw[a:b],p[a:b],scale)
        t=np.arange(5,len(curve))
        if not len(t):
            continue
        ok=gaps[t+1]==gaps[t-5]
        moves=path[t]-path[t-5]
        first=innovations(curve[t],curve[t-5],moves,press[t],ok)@w
        first=center(first*ok[...,None])
        revised=curve.copy()
        # Each of the five phases inherits just its own previous posterior curve.
        for i in range(0,len(t),5):
            s=slice(i,i+5);now=t[s]
            z=innovations(curve[now],revised[now-5],moves[s],press[now],ok[s])@w
            revised[now]+=center((.65*np.tanh(z/.65))*ok[s,...,None])
        diff=revised[t]-curve[t]
        out[a+t]=np.stack((first[..., [3,11]],diff[..., [3,11]],(diff-first)[..., [3,11]]),axis=2)/unit
    return out


def predict(raw,grid,p,params):
    return evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=="__main__":
    collect()
    evaluation.rolling(sys.modules[__name__])
