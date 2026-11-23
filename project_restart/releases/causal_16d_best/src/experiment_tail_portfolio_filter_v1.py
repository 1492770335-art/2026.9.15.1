"""Walk-forward filtering of predicted return bands by time scale and cross-stock covariance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART, CACHE, labels, corr, measure

OUT = ART / 'tail_portfolio_filter_v1'
REFERENCE = ART / 'session_and_passage_calibrated_16d' / 'raw_replay.json'
PARTS = ('replicated_revision_slope_v1', 'conditional_short_transfer_v1', 'session_restart_v1')
RATIO = .161 / .272


def early(day):
    if 8 <= day < 16:
        return np.load(ART / 'cutoff_reference_panel_v1' / f'panel_{day}.npy')
    return legacy_early(day)


def legacy_early(day):
    """Historical reference for reproducing audits of previously selected blends."""
    if day < 12:
        return np.load(ART / 'opening_oof_cache_v1' / f'day_{day:02d}.npy')
    name = 'intraday_relative_calibration_v1' if day < 14 else 'pairwise_book_tree_v1'
    return np.load(ART / name / f'pred_{day}.npy')


def spatial_basis():
    cov = []
    for d in range(8):
        x = np.load(CACHE / f'day_{d:02d}_x.npy', mmap_mode='r')[:, :, 0].astype(float)
        block = x[:len(x)//20*20].reshape(-1,20,50).sum(1)
        block -= block.mean(1,keepdims=True)
        cov.append(np.cov(block.T))
    values, vectors = np.linalg.eigh(np.mean(cov,axis=0))
    shared = np.ones((50,50)) / 50
    sectors = vectors[:,-5:] @ vectors[:,-5:].T
    return np.stack((shared, sectors, np.eye(50)-shared-sectors))


def average(x,k):
    cs = np.concatenate((np.zeros_like(x[:1]),x.cumsum(0,dtype=float)))
    t = np.arange(1,len(cs))
    left = (t-k).clip(0)
    return (cs[t]-cs[left])/(t-left)[:,None,None]


def bands(p):
    x = p.astype(float)
    x[:,:,2] -= RATIO*x[:,:,1]
    x[:,:,1] -= .272/.559*x[:,:,0]
    slow = average(x,96)
    mid = average(x,12)
    return np.concatenate((x-mid,mid-slow,slow),axis=2)


def filtered(x,projection):
    return np.einsum('ij,tjk->tik',projection,x,optimize=True)


def target_residual(p,y):
    result = []
    for j in (0,1):
        b = (p[:,:,2] - j*RATIO*p[:,:,1]).astype(float)
        v = (y[:,:,2] - j*RATIO*y[:,:,1]).astype(float)
        b -= b.mean()
        v -= v.mean()
        scale = (v*b).mean()/(b*b).mean()
        result.append(v/np.clip(scale,.35,3)-b)
    return .5*(result[0]+result[1])


def moments(p,y,basis):
    x = bands(p)
    e = target_residual(p,y).ravel()
    out = []
    for proj in basis:
        z = filtered(x,proj).reshape(len(e),-1)
        z -= z.mean(0)
        out.append((z.T@z/len(z),z.T@e/len(z)))
    return out


def fit(history):
    weights = []
    for days in zip(*history):
        gram,rhs = (np.stack(a) for a in zip(*days))
        g = gram.mean(0)
        penalty = 1.4*np.diag(g.diagonal().clip(.00005,None))
        per_day = np.linalg.solve(gram+penalty,rhs[:,:,None])[:,:,0]
        same = np.abs(np.sign(per_day).mean(0))
        value = .5*np.linalg.solve(g+penalty,rhs.mean(0)) + .5*np.median(per_day,0)
        weights.append((.7*value*same).clip(-.4,.4))
    return np.stack(weights)


def changes(p,basis,w):
    x = bands(p)
    return [filtered((x@c)[:,:,None],proj)[:,:,0] for c,proj in zip(w,basis)]


def scores(p,y,z):
    def ic(a,b):
        return float(corr(a.reshape(-1,1),b.reshape(-1,1))[0])
    return np.array([ic(p[:,:,2]+z-k*RATIO*p[:,:,1],y[:,:,2]-k*RATIO*y[:,:,1]) for k in (0,1)])


def replay_parent(d,raw):
    digest = hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest()
    full=[];head=[];prefix=[]
    for name in PARTS:
        with np.load(ART/name/'raw_trace'/f'day_{d}.npz') as s:
            if str(s['raw_sha256'])!=digest:
                raise ValueError('raw feature digest mismatch')
            full.append(s['full']);head.append(s['head']);prefix.append(int(s['prefix_ticks']))
    if len(set(prefix))!=1:
        raise ValueError('parent prefix lengths differ')
    def combine(a):
        p=a[0].copy()
        p[:,:,1:] += a[1][:,:,1:]-a[2][:,:,1:]
        return p
    return combine(full),combine(head),prefix[0]


def main():
    OUT.mkdir(exist_ok=True)
    basis=spatial_basis()
    history=[];dates=[];report={}
    for end in (8,10,12,14):
        w=fit(history) if history else None
        for d in range(end,end+2):
            p=early(d);y=labels(d).reshape(p.shape)
            if w is not None:
                ref=scores(p,y,0)
                dates.append([scores(p,y,c)-ref for c in changes(p,basis,w)])
            history.append(moments(p,y,basis))
    deltas=np.array(dates)
    by_fold=deltas.reshape(3,2,3,2).mean(1)
    gate=((by_fold>0).sum(0)>=2)&(deltas.mean(0)>0)&(by_fold[-1]>0)
    gate=gate.all(1)
    report.update(training_dates=list(range(1,17)),oof_validation_dates=list(range(11,17)),
                  components=['market','leading_five_relative_modes','remaining_relative_modes'],
                  metrics=['Y3','ticks21_to60'],fold_deltas=by_fold.tolist(),retained=gate.tolist())
    path=OUT/'results.json'
    path.write_text(json.dumps(report,indent=2))
    print('rolling gates',gate,'fold changes',by_fold.tolist(),flush=True)
    if not gate.any():
        return
    w=fit(history)*gate[:,None]
    np.savez(OUT/'model.npz',basis=basis,weights=w)
    import replay_session_experts_16d as original
    import polars as pl
    reference=json.loads(REFERENCE.read_text())
    rows=[];daily={};all_scores=[]
    for d in range(16,20):
        raw=original.features(original.DATA,d)
        p,head,n=replay_parent(d,raw)
        def advance(v):
            q=v.copy()
            q[:,:,2]+=sum(changes(v,basis,w))
            return q
        q=advance(p);partial=advance(head)
        err=float(np.max(abs(q[:n]-partial)))
        if not np.isfinite(q).all() or err>=1.e-4:
            raise ValueError('prefix replay failed')
        y=pl.scan_parquet(original.DATA).filter(pl.col('date')==d).sort('time','code').select('y0','y1','y2').collect().to_numpy().reshape(p.shape)
        row=tuple(a.reshape(-1,3) for a in (p,q,y))
        rows.append(row);ds=measure([row])
        if abs(ds['baseline_score']-reference['daily'][str(d+1)]['score'])>2.e-6:
            raise ValueError('parent differs from independently replayed best')
        daily[d+1]=dict(score=ds['score'],gain=ds['gain'],prefix_error=err)
        all_scores.append((p,q,y))
    p,q,y=(np.concatenate(a) for a in zip(*all_scores))
    report.update(pooled=measure(rows),daily=daily,baseline_band_corr=scores(p,y,0).tolist(),new_band_corr=scores(q,y,0).tolist())
    if report['pooled']['rows']!=936250:
        raise ValueError('invalid validation row count')
    path.write_text(json.dumps(report,indent=2))
    print(report['pooled'],report['new_band_corr'],flush=True)


if __name__=='__main__':
    main()
