"""Compare static peer forecasts with conditional weights adjusted for recent observations of each peer."""
import json
import numpy as np
import experiment_peer_implied_tail_v1 as peer
import experiment_tail_portfolio_filter_v1 as base
from experiment_causal_cross_horizon_ridge_v1 import ART,CACHE,labels,measure

OUT=ART/'peer_quote_reliability_v1'
STRIDE=8


def covariance():
    matrices=[]
    for d in range(8):
        r=np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')[:,:,0].astype(float)
        z=r[:len(r)//40*40].reshape(-1,40,50).sum(1)
        matrices.append(np.cov(z.T))
    return np.mean(matrices,axis=0)


def state(raw,p,cov):
    t=np.arange(len(p))
    r=raw[:,:,0].astype(float)-.00136
    active=abs(r)>.015
    last=np.maximum.accumulate(np.where(active,t[:,None],0),axis=0)
    def trailing(a):
        cs=np.vstack((np.zeros(50),a.cumsum(0,dtype=float)))
        right=t+1;left=(right-96).clip(0)
        return (cs[right]-cs[left])/(right-left)[:,None]
    rate=trailing(active)
    age=(t[:,None]-last)*rate
    # Staleness is measured in units of the asset's own recent movement frequency.
    stale=np.tanh(age/1.5)
    recent=trailing(r*r)/np.maximum(.03,cov.diagonal()/40)
    shock=np.tanh(np.log(recent.clip(.05,20)))
    clock=t[::STRIDE]
    source_noise=.25*np.exp(1.3*(stale[clock]-.4)+.65*shock[clock])
    base_precision=np.linalg.inv(cov+.25*np.diag(cov.diagonal()))
    fixed=-base_precision/np.diag(base_precision)[:,None]
    np.fill_diagonal(fixed,0)
    diag=np.arange(50)
    dynamic=np.broadcast_to(cov,(len(clock),50,50)).copy()
    dynamic[:,diag,diag]+=source_noise*cov.diagonal()[None]
    precision=np.linalg.inv(dynamic)
    dynamic=-precision/precision[:,diag,diag,None]
    dynamic[:,diag,diag]=0
    delta=dynamic-fixed
    local=[]
    for b in (p[:,:,2]-base.RATIO*p[:,:,1],p[:,:,1]*base.RATIO):
        # Split when the peer reliability matrix changes. Each block sees its opening tick only.
        padded=np.pad(b,((0,(-len(t))%STRIDE),(0,0)))
        adjust=(padded.reshape(-1,STRIDE,50)@delta.transpose(0,2,1)).reshape(-1,50)[:len(t)]
        local.extend((adjust,adjust*stale,adjust*shock))
    x=np.stack(local,axis=2)
    return x-x.mean(1,keepdims=True)


def partial(raw,p,graph,w):
    m,r=peer.changes(raw,p,graph,w)
    q=p.copy();q[:,:,2]+=m+r
    return q


def moments(raw,p,y,cov):
    z=state(raw,p,cov).reshape(-1,6)
    e=base.target_residual(p,y)
    v=(e-e.mean(1,keepdims=True)).ravel()
    return z.T@z/len(z),z.T@v/len(z)


def fit(history):
    g,b=(np.stack(a) for a in zip(*history))
    reg=1.4*np.diag(g.mean(0).diagonal().clip(1e-6,None))
    daily=np.linalg.solve(g+reg,b[:,:,None])[:,:,0]
    pooled=np.linalg.solve(g.mean(0)+reg,b.mean(0))
    return (.6*.5*(pooled+np.median(daily,axis=0))*abs(np.sign(daily).mean(0))).clip(-1.,1.)


def main():
    OUT.mkdir(exist_ok=True)
    cov=covariance();graph=peer.peer_weights();ph=[];mh=[];gains=[]
    for end in (8,10,12,14):
        if ph:
            pw=peer.fit(ph)
            pw[0]=0
            w=fit(mh)
        for d in range(end,end+2):
            p=base.early(d);y=labels(d).reshape(p.shape)
            raw=np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')
            q=partial(raw,p,graph,pw) if end>8 else p
            if end>8:
                z=state(raw,q,cov)@w
                gains.append(base.scores(q,y,z)-base.scores(q,y,0))
            ph.append(peer.moments(raw,p,y,graph))
            mh.append(moments(raw,q,y,cov))
    folds=np.array(gains).reshape(3,2,2).mean(1)
    gate=bool(np.all(((folds>0).sum(0)>=2)&(folds.mean(0)>0)&(folds[-1]>0)))
    report={'validation_date_ids':list(range(10,16)),'inner_fold_y3_and_tail_gain':folds.tolist(),'retained':gate}
    save=OUT/'results.json';save.write_text(json.dumps(report,indent=2))
    print('fold gains',folds,'gate',gate,flush=True)
    if not gate:
        return
    w=fit(mh)
    np.savez(OUT/'model.npz',covariance=cov,readout=w)
    import polars as pl
    import replay_session_experts_16d as original
    with np.load(peer.OUT/'model.npz') as st:
        graph,pw=st['peers'],st['readout']
    rows=[];details={}
    for d in range(16,20):
        raw=original.features(original.DATA,d);p,head,n=base.replay_parent(d,raw)
        q=partial(raw,p,graph,pw)
        head=partial(raw[:n],head,graph,pw)
        def apply(raw,p):
            new=p.copy();new[:,:,2]+=state(raw,p,cov)@w
            return new
        final=apply(raw,q)
        err=float(np.max(abs(apply(raw[:n],head)-final[:n])))
        if not np.isfinite(final).all() or err>=1e-4:
            raise ValueError('prefix replay failed')
        y=pl.scan_parquet(original.DATA).filter(pl.col('date')==d).sort('time','code').select('y0','y1','y2').collect().to_numpy()
        row=(q.reshape(-1,3),final.reshape(-1,3),y);rows.append(row)
        ds=measure([row]);details[d+1]=dict(score=ds['score'],gain=ds['gain'],prefix_error=err)
    report.update(pooled=measure(rows),daily=details)
    save.write_text(json.dumps(report,indent=2))
    print(report['pooled'],flush=True)


if __name__=='__main__':
    main()
