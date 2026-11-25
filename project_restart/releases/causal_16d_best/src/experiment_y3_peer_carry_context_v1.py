"""Predict continuation of a short-horizon peer adjustment from its current context."""
import json
import sys

import numpy as np
import torch

import experiment_y3_carried_peer_market_legs_v1 as carry

history, parent, stack, trial = carry.history, carry.parent, carry.stack, carry.trial
OUT = trial.clock.ART/'y3_peer_carry_context_v1'
ARMS = ('unchanged','unconditional carry','alignment with return curve','alignment with market move',
        'persistence across recent revisions','all conditions','future 40 steps only','rotated context',
        'conditional part only','agreement between curve and price')


def smooth(a, width):
    path=np.concatenate((np.zeros_like(a[:1],dtype=float),a.cumsum(0,dtype=float)))
    stop=np.arange(len(a))+1
    start=np.maximum(stop-width,np.where(stop>2401,2401,0))
    return (path[stop]-path[start])/(stop-start)[:,None]


def features(raw,p,z):
    ratio=carry.observed.historical_geometry()[1][2,1]/carry.observed.historical_geometry()[1][1,1]
    direction=np.tanh(z/(.005+np.sqrt(smooth(z*z,64))))
    curve=p[:,:,2]-ratio*p[:,:,1]
    ret=raw[:,:,0].astype(float)-.00136
    price=smooth(ret,24)
    spread=np.sqrt(smooth(ret*ret,64))+.05
    past=smooth(z,24)
    contexts=np.stack((np.tanh(curve/.18),np.tanh(price/(.16*spread)),
                       np.tanh(past/(np.sqrt(smooth(z*z,64))+.005))),axis=2)
    def products(state):
        gates=state*direction[:,:,None]
        agree=np.minimum(abs(gates[:,:,0]),abs(gates[:,:,1]))*np.sign(gates[:,:,0])*(gates[:,:,0]*gates[:,:,1]>0)
        return np.concatenate((z[:,:,None],z[:,:,None]*gates,(z*agree)[:,:,None]),axis=2)
    return products(contexts),products(np.roll(contexts,17,axis=1))


def record(raw,p,z,y,mod):
    a,b=mod.features(raw,p,z)
    ratio=carry.observed.historical_geometry()[1][2,1]/carry.observed.historical_geometry()[1][1,1]
    tail=y[:,:,2]-ratio*y[:,:,1]
    v=np.concatenate((np.ones_like(y[:,:,:1]),p[:,:,1:],a,b,tail[:,:,None],y[:,:,2:3]),axis=2)
    v=v.reshape(-1,v.shape[-1]).astype(float)
    return v.T@v


def weights(records):
    days=np.asarray(records).mean(1)
    def components(m):
        c=stack.covariance(m)
        beta=np.clip(c[1,-1]/c[1,1],.35,3)
        idx=np.arange(2,7)
        g=c[np.ix_(idx,idx)]-np.outer(c[idx,1],c[idx,1])/c[1,1]
        # For a target component, subtract its share of the normalization derivative.
        rhs=c[idx,-1]/beta-c[idx,1]
        far=(c[idx,-2]-c[idx,1]*c[1,-2]/c[1,1])/beta
        return g,rhs,far
    g,rhs,far=components(days.sum(0))
    ds=np.stack([components(m)[1]/m[0,0] for m in days])
    conf=abs(ds.mean(0))/(abs(ds.mean(0))+ds.std(0)/np.sqrt(len(ds))+1.e-8)
    def fit(ids,grad):
        w=np.zeros(5)
        gram=g[np.ix_(ids,ids)]
        w[ids]=(.65*np.linalg.solve(gram+np.diag(gram.diagonal().clip(1.e-8)),grad[ids])).clip(-2,2)
        return w
    w=[np.zeros(5),fit([0],rhs),fit([0,1],rhs),fit([0,2],rhs),fit([0,3],rhs),
       fit([0,1,2,3],rhs*conf),fit([0,1,2,3],far*conf),fit([0,1,2,3],rhs*conf),
       fit([1,2,3],rhs*conf),fit([0,4],rhs)]
    # Refit the shuffled control with the same regularization and historical dates.
    ix=np.r_[0,1,2,8:13,3:8,13,14]
    gs,rs,_=components(days.sum(0)[np.ix_(ix,ix)])
    d=np.stack([components(m[np.ix_(ix,ix)])[1]/m[0,0] for m in days])
    scale=abs(d.mean(0))/(abs(d.mean(0))+d.std(0)/np.sqrt(len(d))+1.e-8)
    g=gs
    w[7]=fit([0,1,2,3],rs*scale)
    return np.array(w)


def gains(m,heads):
    c=stack.covariance(m)
    result=[]
    for j,w in enumerate(heads):
        v=np.zeros(len(c)-2)
        v[1]=1
        start=7 if j==7 else 2
        v[start:start+5]=w
        result.append((v@c[:-2,-2:])/np.sqrt((v@c[:-2,:-2]@v)*c[-1,-1]))
    return np.array(result)-result[0]


def train(mod):
    mod.OUT.mkdir(exist_ok=True)
    basis=carry.peer.U()[0].astype(float)
    stats,folds,daily=[],[],[]
    warm=parent.readouts(carry.warmup(),1)[6]
    for day in range(8,16):
        raw=trial.clock.book.rawday(day)
        y=trial.clock.labels(day).reshape(-1,50,3)
        p=history.references(day,raw)
        old=history.previous.references(day,raw)
        rows=[]
        for a,b in zip(old,p):
            z=b[:,:,1]-a[:,:,1] if day>=10 else carry.legs(a,warm,basis).sum(2)
            if day==10:
                full=mod.features(raw,b,z)[0]
                for n in (311,2477):
                    np.testing.assert_allclose(full[:n],mod.features(raw[:n],b[:n],z[:n])[0],atol=1e-6,rtol=0)
            rows.append(record(raw,b,z,y,mod))
        if day>=10:
            if day%2==0:
                heads=weights(stats)
                group=[]
            group.append(rows)
            daily.append([gains(m,heads) for m in rows])
            if day%2:
                folds.append([gains(m,heads) for m in np.sum(group,0)])
        stats.append(rows)
    f,d=np.array(folds),np.array(daily)
    mean=f.mean(0)
    ids=[j for j in (2,3,4,5,6,8,9) if mean[:,j].min()>0 and (mean[:,j,1]>mean[:,1,1]).all()
         and (mean[:,j,1]>mean[:,7,1]).all() and (f[-1,:,j,1]>0).all()
         and (d[:,:,j,1]>0).sum(0).min()>=4 and (f[:,:,j,0]>0).sum(0).min()>=2]
    pick=max(ids,key=lambda j:mean[:,j,1].min()) if ids else 0
    report=dict(selected=pick,arms=mod.ARMS,metrics=['tail contribution','Y3 gain'],folds=f.tolist(),daily=d.tolist(),mean=mean.tolist())
    (mod.OUT/'results.json').write_text(json.dumps(report))
    print(mod.ARMS[pick],mean.round(6),flush=True)
    if pick:
        np.savez(mod.OUT/'model.npz',coef=weights(stats)[pick],pick=pick)
    return bool(pick)


def adjustment(mod):
    with np.load(mod.OUT/'model.npz') as f:
        w=f['coef']
    def apply(raw,old,p):
        q=p.copy()
        x=mod.features(raw,p,p[:,:,1]-old[:,:,1])[0]
        q[:,:,2]+=(x@w).astype('float32')
        return q
    return apply


def layer(mod=None):
    mod=mod or sys.modules[__name__]
    prev=history.previous.incumbent.layer()
    endpoint=parent.adjustment()
    adjust=adjustment(mod)
    def apply(raw,base):
        old=prev(raw,base)
        return adjust(raw,old,endpoint(old))
    return apply


def replay_with(mod=None):
    adjust=adjustment(mod or sys.modules[__name__])
    def read(d,raw):
        old,short,_=history.previous.read(d,raw)
        p,h,n=history.read(d,raw)
        return adjust(raw,old,p),adjust(raw[:n],short,h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    mod=sys.modules[__name__]
    if sys.argv[1:]==['replay'] or train(mod):
        carry.peer.replay.replay(mod)
