"""Predict the common Y3 return from forecast errors accumulated in two disjoint stock portfolios."""
import json
import sys
import numpy as np
import torch

import y3_unfulfilled_references as history

incumbent=history.incumbent
prior,trial=incumbent.prev,incumbent.trial
OUT=trial.clock.ART/'y3_market_reversal_flow_v1'
ARMS=('recent signed prices','opposing moves','unfulfilled expectations','overshooting moves','all surprise types',
      'stock-permuted forecasts','stale forecasts','opposing moves across timescales','half all types')


def pools(raw,w):
    r=raw[:,:,0].astype('float32')-.00136
    expected=prior.predicted(raw,w)
    stale=np.zeros_like(expected);stale[16:]=expected[:-16]
    outputs=[]
    for sl in (slice(None,None,2),slice(1,None,2)):
        move=r[:,sl]
        pred=expected[:,sl]
        same=pred*move>0
        fulfilled=np.where(same,np.sign(pred)*np.minimum(abs(pred),abs(move)),0.)
        streams=[move,np.where(same,0,move),fulfilled-pred,np.where(same,move-fulfilled,0),
                 np.where(np.roll(expected,17,axis=1)[:,sl]*move<0,move,0),
                 np.where(stale[:,sl]*move<0,move,0)]
        parts=[]
        for x in streams:
            signed=np.column_stack((np.tanh(x).mean(1),np.tanh(4*x).mean(1))).astype('float32')
            cs=np.vstack((np.zeros(2),signed.cumsum(0,dtype=float)))
            end=np.arange(len(raw))+1
            pools=[]
            for width in (4,16,48):
                start=(end-width).clip(0)
                pools.append((cs[end]-cs[start])/(end-start)[:,None])
            parts.extend(pools)
        outputs.append(np.column_stack(parts).astype('float32'))
    return outputs


def record(z,p,y):
    p=p.astype(float)
    y=y.astype(float)
    q=p[:,:,2].ravel()
    obs=y[:,:,2].ravel()
    beta=np.cov(q,obs,bias=True)[0,1]/q.var().clip(1e-6)
    moments=[]
    for j in range(2):
        side=slice(1-j,None,2)
        x=np.column_stack((p[:,side].mean(1),z[j]))
        err=(y[:,side,2]/beta.clip(.5,2.)-p[:,side,2]).mean(1)
        x=x-x.mean(0);err=err-err.mean()
        moments.append((x.T@x/len(x),x.T@err/len(x)))
    return tuple(np.mean(v,axis=0) for v in zip(*moments))


def fit(rows):
    grams,rhs=[np.stack(a) for a in zip(*rows)]
    g=grams.mean(0)
    def read(cols,n):
        b=np.eye(len(g))[:,cols]
        b[:n]=-trial.curve.solve(g[:n,:n],g[:n,cols],.15)
        daily=rhs@b;mu=daily.mean(0)
        mu*=abs(mu)/(abs(mu)+1.7*daily.std(0)/np.sqrt(len(rows))+1e-6)
        return b@trial.curve.solve(b.T@g@b,mu,1.7)
    groups=[np.arange(3+6*j,9+6*j) for j in range(6)]
    total,reverse,miss,extra,wrong,stale=groups
    joint=read(np.r_[reverse,miss,extra],9)
    bank=[read(total,3),read(reverse,9),read(miss,9),read(extra,9),joint,read(wrong,9),
          read(stale,9),read(reverse[::2],9),.5*joint]
    return dict(w=(.40*np.column_stack(bank)).astype('float32'),teacher=prior.teacher())


def changes(z,p,m):
    forecasts=np.mean([np.column_stack((p[:,1-j::2].mean(1),x))@m['w'] for j,x in enumerate(z)],axis=0)
    return [np.broadcast_to(v[:,None],p.shape[:2]) for v in forecasts.T]


def stats(z,p,y,m):
    changes_=changes(z,p,m)
    market=p.mean(1,keepdims=True)
    target=y.mean(1,keepdims=True)
    bs=trial.scoring.audit.blockstats
    return [bs(p,y,changes_).sum(0),bs(market,target,[c[:,:1] for c in changes_]).sum(0)]


def run():
    OUT.mkdir(exist_ok=True)
    hist=[];folds=[];daily=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        z=pools(raw,prior.teacher())
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=[prior.path.early.load(d,raw)] if d<8 else history.references(d,raw)
        if d>=8 and d%2==0:
            frozen=fit(hist);collected=[[],[]]
        entries=[]
        for j,p in enumerate(refs):
            entries.append(record(z,p,y))
            if d>=8:
                if d==8 and j==0:
                    a,b=changes(z,p,frozen),changes(pools(raw[:311],frozen['teacher']),p[:311],frozen)
                    for u,v in zip(a,b): np.testing.assert_allclose(u[:311],v,rtol=0,atol=1e-6)
                g=stats(z,p,y,frozen)
                collected[j].append(g)
                if j==0: daily.append(trial.scoring.audit.pooled(g[0]))
        hist.append([np.mean(a,axis=0) for a in zip(*entries)])
        if d>=8 and d%2:
            folds.append([[trial.scoring.audit.pooled(x) for x in np.sum(c,axis=0)] for c in collected])
    f=np.array(folds)
    good=[]
    for j in (1,2,3,4,7,8):
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0
            and (np.array(daily)[:,j]>0).sum()>=5 and f[:,:,1,j].mean(0).min()>0
            and all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,5,6))):
            good.append(j)
    pick=max(good,key=lambda j:f[:,0,0,j].mean()) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),daily=np.array(daily).tolist()),indent=2))
    print('selected',pick,flush=True)
    if pick is not None:
        np.savez(OUT/'model.npz',pick=pick,**fit(hist))
        history.outer(sys.modules[__name__])


def correction():
    with np.load(OUT/'model.npz') as f:m=dict(f)
    return lambda raw,p:changes(pools(raw,m['teacher']),p,m)[int(m['pick'])]


def layer():
    predict,update=incumbent.layer(),correction()
    def apply(raw,p):
        q=predict(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
