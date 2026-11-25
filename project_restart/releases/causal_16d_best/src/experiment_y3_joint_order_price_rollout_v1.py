"""Forecast from the outstanding forty-frame order and price state, including trades that have not yet occurred."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import y3_joint_order_price_state as process

trial,study,history,book=process.trial,process.study,process.history,process.book
OUT=trial.clock.ART/'y3_joint_order_price_rollout_v1'
AGES=((0,0),(1,2),(3,4),(5,9),(10,18),(19,19),(20,29),(30,39))
ARMS=('direct known-state forecast','full recursive forecast','future order mediation','future price mediation',
      'cross-name transmission','own market dynamics','wrong order routing','all forward windows','future order and price paths')
CANDIDATES=(1,2,3,4,7,8)
CONTROLS=(0,5,6)


def weights(ar,kind):
    w=ar.reshape(5,100,100).astype(float).copy()
    if kind=='no new orders':w[:,:,:50]=0
    if kind=='own':w*=np.tile(np.eye(50),(2,2))
    if kind=='shuffled':
        ix=np.random.default_rng(938).permutation(50)
        for v in w:
            for j in (0,50):
                old=v[j:j+50,:50].copy()
                repl=old[np.ix_(ix,ix)]
                np.fill_diagonal(repl,old.diagonal())
                v[j:j+50,:50]=repl
    # A forecast is a linear function of the last forty frames. Backward recursion is exact.
    previous=np.zeros((40,100,50))
    previous[0,50:]=np.eye(50)
    total=np.zeros_like(previous)
    bands=[]
    for h in range(1,61):
        cur=np.zeros_like(previous)
        cur[:-1]=previous[1:]
        use=previous[0].copy()
        if h>1:
            if kind=='direct':use[:]=0
            if kind=='no price feedback':use[50:]=0
        for gate,(lo,hi) in zip(w,process.LAGS):
            cur[lo-1:hi]+=(gate@use)/(hi-lo+1)
        total+=cur
        if h in (5,20,40,60):
            # Pool coefficients within age buckets to make full-grid replay fast.
            bands.append(np.concatenate([total[a:b+1].mean(0) for a,b in AGES]))
            total=np.zeros_like(total)
        previous=cur
    return np.stack(bands,0)


@lru_cache(maxsize=7)
def params(end):
    st=process.fitted(end)
    vals=[weights(st['ar'],s) for s in ('direct','full','no new orders','no price feedback','own','shuffled')]
    return dict(**{k:v for k,v in st.items() if k!='ar'},rollout=np.stack(vals).astype('float32'))


def estimates(raw,t,st):
    state=(process.state(raw,st)-st['mu'])/st['sc']
    cs=np.vstack((np.zeros(100),state.cumsum(0)))
    x=np.concatenate([cs[(t+1-a).clip(0)]-cs[(t-b).clip(0)] for a,b in AGES],1).astype('float32')
    m=st['rollout']
    co=m.transpose(2,0,1,3).reshape(m.shape[2],-1)
    return (x@co).reshape(len(x),6,4,50)*st['sc'][50:]


def design_with(raw,p,t,st):
    direct,full,muted,no_price,own,wrong=estimates(raw,t,st).transpose(1,0,3,2)
    cols=[]
    for v in (direct,full,full-muted,full-no_price,full-own,own,wrong):
        cols.append(np.stack((v[:,:,:2].sum(2),v[:,:,2:].sum(2)),2))
    cols.extend((full,full-muted,full-no_price))
    return trial.clock.center(np.concatenate((p[t],*cols),2)).astype('float32')


def historical_design(d):
    st=params(max(4,d-d%2))
    return lambda raw,p,t:design_with(raw,p,t,st)


def fit(rows):
    lengths=[2]*7+[4,8]
    bounds=3+np.cumsum([0]+lengths)
    return dict(**book.prev.gram(rows,[(np.arange(lo,hi),3,.35) for lo,hi in zip(bounds,bounds[1:])]),**params(len(rows)+4))


changes=book.changes


def correction():
    with np.load(OUT/'model.npz') as f:st=dict(f)
    def update(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),st),st)[int(st['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


def audit():
    stats=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d);st=params(d-d%2)
        t=np.r_[np.arange(80,2260,4),np.arange(2460,len(raw)-60,4)]
        model=estimates(raw,t,st)
        ret=process.state(raw,st)[:,50:]
        cs=np.vstack((np.zeros(50),ret.cumsum(0)))
        windows=[]
        for h,(lo,hi) in enumerate(process.BANDS):
            y=trial.clock.center(cs[t+hi+1]-cs[t+lo]).ravel()
            windows.append([float(np.corrcoef(trial.clock.center(v).ravel(),y)[0,1]) if np.std(v)>1e-9 else None
                            for v in np.moveaxis(model[:,:,h],1,0)])
        stats.append(windows)
    mean=[[float(np.mean(v)) if all(x is not None for x in v) else None for v in zip(*[d[h] for d in stats])]
          for h in range(len(process.BANDS))]
    info=dict(days=[9,16],bands=process.BANDS,models=['direct','full','suppress future orders','suppress price feedback','own','shuffled'],
              target='market-relative sum of arcsinh-transformed one-frame returns',daily=stats,mean=mean)
    (OUT/'rollout_skill.json').write_text(json.dumps(info,indent=2,allow_nan=False))
    print('rollout component correlations',np.round(np.array(mean,dtype=float),4),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
    audit()
