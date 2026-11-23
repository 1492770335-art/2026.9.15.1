"""Estimate the reliability of the far return after partialling out the near return."""
import json
import numpy as np

import experiment_conditional_expert_subspace_v1 as incumbent
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

roll = incumbent.rolling
book = roll.gls.book
OUT = ART/'maturing_band_slopes_v1'
S20 = .272164
S60 = .160943
RATIO = S60/S20


def split(a):
    market = a.mean(1, keepdims=True, dtype=float)
    return market, a-market


def correlations(p, y):
    def audit(forecast, truth):
        forecasts, targets = split(forecast), split(truth)
        return dict(corr=[float(np.corrcoef(x.ravel(), z.ravel())[0, 1]) for x,z in zip(forecasts, targets)],
                    market_truth_share=float(targets[0].var()/truth.var()),
                    market_forecast_share=float(forecasts[0].var()/forecast.var()))
    return dict(y3=audit(p[:,:,2], y[:,:,2]),
                far=audit(p[:,:,2]-RATIO*p[:,:,1], y[:,:,2]-RATIO*y[:,:,1]))


def states(p):
    return tuple(np.stack(pair, axis=-1) for pair in zip(*[split(a) for a in
                 (RATIO*p[:,:,1], p[:,:,2]-RATIO*p[:,:,1])]))


def statistics(x,y):
    return np.concatenate(((x[:,:,:,None]*x[:,:,None,:]).mean(1).reshape(len(x),4),
                            (x*y[:,:,None]).mean(1)),axis=1)


def elapsed(raw,p):
    # At t, price[t]-price[t-40] is the far interval for a forecast made at t-60.
    # A second measurement becomes available earlier: keep the first 20 realized
    # steps and use the newly available Y2 forecast for the final 20.
    r=raw[:,:,0].astype(float)-.00136
    price=r.cumsum(0)
    t=np.arange(len(p))
    z=states(p)
    result=[]
    for h in (60,40):
        a=(t-h).clip(0)
        realized=S60*(price-price[np.minimum(a+20,t)])
        if h==40:
            realized+=RATIO*p[:,:,1]
        realized[:h]=0
        moments=[]
        for x,truth in zip(z,split(realized)):
            stat=statistics(x[a],truth)
            stat[:h]=0
            moments.append(stat)
        result.append(np.stack(moments,1))
    return result


def truth_stats(p,y):
    target=y[:,:,2]-RATIO*y[:,:,1]
    return np.stack([statistics(x,z).mean(0) for x,z in zip(states(p),split(target))])


def coefficient(s,prior):
    gram=s[...,:4].reshape(*s.shape[:-1],2,2)
    reference=prior[...,:4].reshape(2,2,2)
    regularizer=.32*np.eye(2)[None]*reference.diagonal(axis1=1,axis2=2).clip(.0001)[:,None,:]
    anchor=np.array([0.,1.])
    return np.linalg.solve(gram+regularizer,(s[...,4:]+regularizer@anchor)[...,None])[...,0]


def features(raw,p,prior):
    z=states(p)
    observed=elapsed(raw,p)
    prelim=coefficient(prior,prior)
    control=np.broadcast_to(prelim,(len(p),2,2)).copy()
    # Only use information from the same day's prefix. Shrink the highly
    # overlapping windows toward the previous dates' moment estimates.
    weights=[control]
    for stat,lag in zip(observed,(60,40)):
        acc=np.concatenate((np.zeros_like(stat[:1]),stat.cumsum(0)))
        t=np.arange(len(p))+1
        estimates=[]
        for window in (480,1800):
            l=(t-window).clip(0)
            n=(t-np.maximum(l,lag)).clip(0)
            posterior=(acc[t]-acc[l]+750*prior)/(n[:,None,None]+750)
            estimates.append(coefficient(posterior,prior))
        weights.append(np.mean(estimates,axis=0))
    result=[]
    for mode,w in enumerate(weights):
        diff=(w-np.array([0,1]) if mode==0 else w-control).clip(-1.3,1.3)
        result.append([np.einsum('tsj,tj->ts',x,diff[:,j]) for j,x in enumerate(z)])
    return result


def record(feat,p,y):
    q=p[:,:,2].astype(float)
    target=y[:,:,2].astype(float)
    a=q-q.mean()
    target-=target.mean()
    slope=np.clip((target*a).mean()/(a*a).mean(),.35,3.)
    residual=split(target/slope-a)
    stats=[]
    for k,e in enumerate(residual):
        x=np.stack([f[k] for f in feat],axis=-1).reshape(-1,3)
        x-=x.mean(0)
        stats.append((x.T@x/len(x),x.T@e.ravel()/len(x)))
    return stats


def fit(rows):
    out=[]
    for days in zip(*rows):
        gs,bs=map(np.stack,zip(*days))
        reg=.8*np.diag(gs.mean(0).diagonal().clip(1e-6))
        beta=np.linalg.solve(gs.mean(0)+reg,bs.mean(0))
        daily=np.linalg.solve(gs+reg,bs[:,:,None])[:,:,0]
        out.append((.45*(.5*beta+.5*np.median(daily,0))*abs(np.sign(daily).mean(0))).clip(-.6,.6))
    return np.stack(out)


def delta(feats,w,part,dynamic):
    modes=range(3) if dynamic else (0,)
    return sum(feats[k][part]*w[part,k] for k in modes)


def load():
    parent=incumbent.load()
    with np.load(OUT/'model.npz') as st:
        prior,w,keep=st['prior'],st['w'],st['keep']
    def apply(raw,p):
        base=parent(raw,p);q=base.copy()
        f=features(raw,base,prior)
        for j in np.flatnonzero(keep):
            q[:,:,2]+=delta(f,w,j,True)
        return q
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    mixing=[];priors=[];history=[];folds=[];days=[];diagnostics=[]
    for end in (8,10,12,14):
        mix=roll.fit(mixing)
        # For the first training block, use its first day's moments to fit
        # controls on the second. Both precede every reported validation date.
        prior=np.mean(priors,0) if priors else None
        w=fit(history) if history else None
        blocks=[[[],[]] for _ in range(2)]
        dy=[];dp=[]
        for d in (end,end+1):
            fs=roll.sources(d);raw=book.rawday(d)
            t=np.arange(len(raw));p=roll.predict(fs,t,mix);y=labels(d).reshape(p.shape)
            mixing.append(roll.moments(fs,y,t[d%5::5]))
            dp.append(p);dy.append(y)
            current=prior if prior is not None else (priors[0] if priors else None)
            priors.append(truth_stats(p,y))
            if current is None:
                continue
            f=features(raw,p,current)
            history.append(record(f,p,y))
            if w is None:
                continue
            if d==end:
                prefix=features(raw[:213],p[:213],current)
                for full,short in zip(f,prefix):
                    for a,b in zip(full,short):
                        np.testing.assert_allclose(a[:213],b,rtol=0,atol=2e-7)
            row_gains=[]
            for j,block in enumerate(blocks):
                for k,rs in enumerate(block):
                    q=p.copy()
                    q[:,:,2]+=delta(f,w,j,k==1)
                    row=tuple(a[::3].reshape(-1,3) for a in (p,q,y))
                    rs.append(row)
                    if k:
                        s=measure([row]);row_gains.append(s['candidate'][2]-s['base'][2])
            days.append(row_gains)
        diagnostics.append(correlations(np.concatenate(dp),np.concatenate(dy)))
        if w is not None:
            folds.append([[measure(rs) for rs in block] for block in blocks])
            print(end+1,[[(s['candidate'][2]-s['base'][2]) for s in row] for row in folds[-1]],flush=True)
    g=np.array([[[s['candidate'][2]-s['base'][2] for s in pair] for pair in f] for f in folds])
    keep=(g[:,:,1].min(0)>0)&((g[:,:,1]-g[:,:,0]).mean(0)>0)&((np.array(days)>0).sum(0)>=4)
    report=dict(components=['market','relative'],arms=['prior','maturing'],fold_start_days=[11,13,15],
                diagnostic_start_days=[9,11,13,15],diagnostic=diagnostics,folds=folds,daily_gains=days,retained=keep.tolist())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('retained',keep,flush=True)
    if keep.any():
        np.savez(OUT/'model.npz',prior=np.mean(priors,0),w=fit(history),keep=keep)
        replay=roll.gls.best.walk.replay
        res=replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        old=json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
        print('official',res['pooled']['score'],res['pooled']['score']-old,flush=True)
        if res['pooled']['score']>old:
            check=replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],res['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
