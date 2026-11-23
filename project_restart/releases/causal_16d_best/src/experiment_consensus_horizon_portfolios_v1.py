"""Walk-forward combination of horizon increments in market, correlated-stock and individual portfolios."""
import json
import numpy as np
from scipy.optimize import minimize
from experiment_causal_cross_horizon_ridge_v1 import ART, CACHE, labels, measure
import experiment_tail_portfolio_filter_v1 as panel
from experiment_eight_date_expert_contrasts_v1 import SOURCES
import experiment_matured_expert_tail_v1 as parent

OUT=ART/'consensus_horizon_portfolios_v1'
PARENT=ART/'equal_expert_relative_tail_v1'
UNITS=np.array([.559,.272,.161],np.float32)


def bands(p):
    return np.diff(p/UNITS,axis=2,prepend=np.zeros_like(p[:,:,:1]))


def features(p,forecasts,axes):
    own=bands(p)
    parts=[bands(v) for v in forecasts]
    other=np.mean(parts,axis=0)
    shared=np.mean([parts[i].mean(1,keepdims=True) for i in (0,1,3)],axis=0)-own.mean(1,keepdims=True)
    diff=other-own
    diff-=diff.mean(1,keepdims=True)
    factors=axes@diff
    return np.stack((np.broadcast_to(shared,diff.shape),factors,diff-factors),axis=2).astype('float32')


def moments(p,y,z):
    out=[]
    for j in range(3):
        x=(z[:,:,:,:j+1].reshape(len(p)*50,-1)*UNITS[j]).astype(float)
        x-=x.mean(0)
        v=p[:,:,j].ravel().astype(float);v-=v.mean()
        truth=y[:,:,j].ravel().astype(float);truth-=truth.mean()
        n=len(v);scale=max(v@v/n,1e-7)
        out.append((x.T@x/n/scale,x.T@v/n/scale,x.T@truth/n/np.sqrt(scale*truth.var()),
                    (v@truth)/n/np.sqrt(scale*truth.var())))
    return out


def fit(history):
    weights=np.zeros((3,3,3),np.float32)
    for j,days in enumerate(zip(*history)):
        g,a,b,ic=map(np.stack,zip(*days))
        def objective(w):
            var=1+2*(a@w)+np.einsum('i,dij,j->d',w,g,w)
            delta=(ic+b@w)/np.sqrt(var.clip(1e-8,None))-ic
            # One date is one observation. Penalize unstable gains and heavily moved blends.
            return -delta.mean()+.50*delta.std()+.003*(w@w)
        n=len(a[0]);result=minimize(objective,np.zeros(n),method='L-BFGS-B',bounds=[(0,.45)]*n)
        if not result.success and 'ABNORMAL' not in str(result.message):
            raise ValueError(result.message)
        weights[j,:,:j+1]=result.x.reshape(3,j+1)
    return weights


def predict(p,z,weights):
    return (p+np.einsum('tsgb,hgb->tsh',z,weights,optimize=True)*UNITS).astype('float32')


def verify(axes,w,transform=predict):
    import polars as pl
    import replay_session_experts_16d as replay
    import experiment_peer_implied_tail_v1 as peer
    from experiment_replicated_expert_consensus_v1 import sources
    from experiment_peer_quote_reliability_v1 import partial
    experts=sources();fixed=np.load(PARENT/'weights.npy')[0]
    with np.load(peer.OUT/'model.npz') as st:graph,pw=st['peers'],st['readout']
    ref=json.loads((PARENT/'raw_replay.json').read_text())
    rows=[];details={}
    def calculate(raw,p):
        fs=[fn(raw) for fn in experts]
        prior=parent.fixed_forecast(partial(raw,p,graph,pw),fs,fixed)
        return prior,transform(prior,features(prior,fs,axes),w)
    for d in range(16,20):
        raw=replay.features(replay.DATA,d);p,head,n=panel.replay_parent(d,raw)
        old,q=calculate(raw,p);_,short=calculate(raw[:n],head)
        err=float(np.max(abs(q[:n]-short)))
        if not np.isfinite(q).all() or err>1e-4:raise ValueError('invalid prefix')
        y=pl.scan_parquet(replay.DATA).filter(pl.col('date')==d).sort('time','code').select('y0','y1','y2').collect().to_numpy()
        row=old.reshape(-1,3),q.reshape(-1,3),y;rows.append(row);s=measure([row])
        if abs(s['baseline_score']-ref['daily'][str(d+1)]['score'])>2e-6:raise ValueError('parent mismatch')
        details[d+1]=dict(gain=s['gain'],score=s['score'],prefix_error=err)
    result=measure(rows)
    if result['rows']!=936250:raise ValueError('row count')
    return dict(pooled=result,daily=details)


def training_day(d,history):
    p=panel.early(d);y=labels(d).reshape(p.shape)
    fs=[np.load(ART/src/f'side_{d}.npy') for src in SOURCES]
    new=parent.fixed_forecast(p,fs,1.)[:,:,2]-p[:,:,2]
    x=new[::3].ravel().astype(float)
    e=panel.target_residual(p[::3],y[::3]).ravel()
    day=float(x@x/len(x)),float(x@e/len(x))
    return p,y,fs,day


def fixed_weight(stats):
    if not stats:return 0.
    g,b=map(np.array,zip(*stats));reg=.75*max(g.mean(),.00001)
    daily=b/(g+reg)
    return np.clip(.65*(.65*b.mean()/(g.mean()+reg)+.35*np.median(daily))*abs(np.sign(daily).mean()),0,.6)


def main():
    OUT.mkdir(exist_ok=True);axes=panel.spatial_basis()[1].astype('float32')
    hist=[];priors=[];report={'folds':[]};daily=[]
    for end in (8,10,12,14):
        c=fixed_weight(priors);w=fit(hist) if hist else None;rows=[]
        for d in range(end,end+2):
            p,y,fs,day=training_day(d,priors);priors.append(day)
            p=parent.fixed_forecast(p,fs,c)
            z=features(p,fs,axes)
            if w is not None:
                q=predict(p,z,w)
                row=tuple(a.reshape(-1,3) for a in (p,q,y));rows.append(row);s=measure([row])
                daily.append(np.subtract(s['candidate'],s['base']))
            hist.append(moments(p[::3],y[::3],z[::3]))
        if rows:
            s=measure(rows);report['folds'].append(s);print(end+1,s,flush=True)
    gain=np.stack([np.subtract(s['candidate'],s['base']) for s in report['folds']])
    gate=(gain.min(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    report['retained']=gate.tolist()
    path=OUT/'results.json';path.write_text(json.dumps(report,indent=2))
    if gate.any():
        w=fit(hist)*gate[:,None,None];np.savez(OUT/'model.npz',axes=axes,weights=w)
        report['development']=verify(axes,w);path.write_text(json.dumps(report,indent=2))
        print(report['development'],flush=True)


if __name__=='__main__':main()
