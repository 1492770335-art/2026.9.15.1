"""Estimate a stock's long forecast from the other 49 stocks using training-day conditional covariances."""
import json
import numpy as np
import experiment_tail_portfolio_filter_v1 as base
from experiment_causal_cross_horizon_ridge_v1 import ART,CACHE,labels,measure

OUT=ART/'peer_implied_tail_v1'


def peer_weights():
    daily=[]
    for d in range(8):
        raw=np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')
        r=raw[:,:,0].astype(float)-.00136
        z=r[:len(r)//40*40].reshape(-1,40,50).sum(1)
        daily.append(np.cov(z.T))
    g=np.median(daily,axis=0)
    # Shrink each daily covariance before inversion so collinear peers cannot dominate.
    weights=[]
    for cov in daily:
        c=.65*np.mean(daily,axis=0)+.35*cov
        precision=np.linalg.inv(c+.25*np.diag(g.diagonal()))
        w=-precision/np.diag(precision)[:,None]
        np.fill_diagonal(w,0.)
        weights.append(w)
    return np.median(weights,axis=0)


def inputs(raw,p,weights):
    ret=raw[:,:,0].astype(float)-.00136
    t=np.arange(len(p))[:,None]
    last=np.maximum.accumulate(np.where(abs(ret)>.015,t,0),axis=0)
    stale=np.tanh((t-last)/8)
    tail=p[:,:,2]-base.RATIO*p[:,:,1]
    prior=p[:,:,1]*base.RATIO
    predictors=[]
    for a in (tail,prior):
        peers=a@weights.T
        gap=peers-a
        strength=(abs(peers)/(abs(peers)+abs(a)+.03)).clip(0,1)
        same=np.tanh(peers*a/.025)
        predictors.extend((gap,gap*stale,gap*strength,gap*same))
    z=np.stack(predictors,axis=2)
    m=z.mean(1)
    return m,z-m[:,None]


def moments(raw,p,y,weights):
    m,r=inputs(raw,p,weights)
    e=base.target_residual(p,y);em=e.mean(1)
    hist=[]
    for z,b in ((m,em),(r.reshape(-1,r.shape[2]),(e-em[:,None]).ravel())):
        z=z-z.mean(0)
        hist.append((z.T@z/len(z),z.T@b/len(z)))
    return hist


def fit(hist):
    result=[]
    for days in zip(*hist):
        g,b=(np.stack(a) for a in zip(*days))
        ridge=.85*np.diag(g.mean(0).diagonal().clip(.00004,None))
        dw=np.linalg.solve(g+ridge,b[:,:,None])[:,:,0]
        w=np.linalg.solve(g.mean(0)+ridge,b.mean(0))
        result.append((.65*(w+np.median(dw,axis=0))*.5*abs(np.sign(dw).mean(0))).clip(-.4,.4))
    return np.stack(result)


def changes(raw,p,weights,w):
    m,r=inputs(raw,p,weights)
    return (m@w[0])[:,None],r@w[1]


def main():
    OUT.mkdir(exist_ok=True);weights=peer_weights();history=[];gains=[]
    for end in (8,10,12,14):
        w=fit(history) if history else None
        for d in range(end,end+2):
            p=base.early(d);y=labels(d).reshape(p.shape)
            raw=np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')
            if w is not None:
                zero=base.scores(p,y,0)
                gains.append([base.scores(p,y,z)-zero for z in changes(raw,p,weights,w)])
            history.append(moments(raw,p,y,weights))
    folds=np.array(gains).reshape(3,2,2,2).mean(1)
    gate=(((folds>0).sum(0)>=2)&(folds.mean(0)>0)&(folds[-1]>0)).all(1)
    report={'validation_date_ids':list(range(10,16)),'inner_fold_y3_and_tail_gain':folds.tolist(),'retained':gate.tolist()}
    save=OUT/'results.json';save.write_text(json.dumps(report,indent=2))
    print('fold gains',folds.tolist(),'gate',gate,flush=True)
    if not gate.any():
        return
    w=fit(history)*gate[:,None]
    np.savez(OUT/'model.npz',peers=weights,readout=w)
    import replay_session_experts_16d as original
    import polars as pl
    ref=json.loads(base.REFERENCE.read_text())
    def advance(raw,p):
        q=p.copy()
        q[:,:,2]+=sum(changes(raw,p,weights,w))
        return q
    rows=[];daily={};bands=[]
    for d in range(16,20):
        raw=original.features(original.DATA,d);p,head,n=base.replay_parent(d,raw)
        q=advance(raw,p);err=float(np.max(abs(advance(raw[:n],head)-q[:n])))
        if not np.isfinite(q).all() or err>=1.e-4:
            raise ValueError('prefix replay failed')
        y=pl.scan_parquet(original.DATA).filter(pl.col('date')==d).sort('time','code').select('y0','y1','y2').collect().to_numpy().reshape(p.shape)
        row=tuple(a.reshape(-1,3) for a in (p,q,y));rows.append(row);ds=measure([row])
        if abs(ds['baseline_score']-ref['daily'][str(d+1)]['score'])>2e-6:
            raise ValueError('base score does not match raw replay')
        daily[d+1]=dict(score=ds['score'],gain=ds['gain'],prefix_error=err)
        bands.append((p,q,y))
    p,q,y=(np.concatenate(a) for a in zip(*bands))
    report.update(pooled=measure(rows),daily=daily,base_ic=base.scores(p,y,0).tolist(),candidate_ic=base.scores(q,y,0).tolist())
    save.write_text(json.dumps(report,indent=2))
    print(report['pooled'],report['candidate_ic'],flush=True)


if __name__=='__main__':
    main()
