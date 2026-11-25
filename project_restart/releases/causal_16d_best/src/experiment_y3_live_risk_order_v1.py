"""Update the stock-space projector for an order response from already observed intraday returns."""
from functools import lru_cache
import sys

import numpy as np
import torch

import joint_factor_order_references as history
import experiment_y3_continuous_factor_order_v1 as spectrum

parent,trial,study=history.incumbent,history.trial,history.study
OUT=trial.clock.ART/'y3_live_risk_order_v1'
ARMS=('static split-date weights','live covariance','stock drift in factor weights','live replicated weights',
      'rotation with prior ranks','weight transfer without rotation','high-confidence correction',
      'incorrect stock identities','current session only','low-variance stock response','late live transfer')
CANDIDATES=(1,2,3,4,5,6,8,9,10)
CONTROLS=(0,7)


@lru_cache(maxsize=7)
def state(end):
    st=spectrum.state(end)
    return dict(head=st['weights'][5],u=st['basis'][5],mix=st['mix'][5],
                prior=np.mean([spectrum.cov(d) for d in range(end)],0))


def kernel(u,m):
    return np.einsum('sk,ki,rk->isr',u,m,u,optimize=True)


def update(prior,obs,u,m):
    if len(obs)<5:
        return u,m
    n=len(obs)
    g=obs.T@obs/n
    sd=np.sqrt(g.diagonal().clip(.00001))
    rho=n/(n+75.)
    c=(1-rho)*prior+rho*g/sd[:,None]/sd[None,:]
    _,v=np.linalg.eigh(u.T@c@u)
    v=v[:,::-1]
    # Transfer probabilistic memberships to the updated basis via squared overlaps.
    return u@v,v.T**2@m


def features(raw,t,st):
    u,m=st['u'],st['mix']
    x=spectrum.parent.features(raw,t)
    payoff=(x@st['head'].transpose(1,0,2).reshape(x.shape[-1],9)).reshape(len(t),50,3,3)
    r=raw[:,:,0].astype(float)-.00136
    r-=r.mean(1,keepdims=True)
    cs=np.concatenate((np.zeros_like(r[:1]),r.cumsum(0)))
    origin=np.r_[np.arange(0,min(2401,len(r))-19,20),np.arange(2401,len(r)-19,20)]
    sample=cs[origin+20]-cs[origin]
    finish=origin+20
    fixed=kernel(u,m)
    # Re-evaluate soft memberships on disjoint, nonoverlapping return intervals.
    out=np.empty((len(t),50,10,3),dtype='float32')
    for block in np.unique(t//160):
        boundary=160*block
        ids=np.flatnonzero(t//160==block)
        observed=sample[finish<=boundary]
        vu,soft=update(st['prior'],observed,u,m)
        live=kernel(vu,m)
        rotated=kernel(vu,soft)
        halves=[update(st['prior'],observed[k::2],u,m) for k in (0,1)]
        left,right=[kernel(a,m) for a,_ in halves]
        agree=.5*(left@right+right@left)
        prior_sq=fixed@fixed
        stable=fixed+agree-prior_sq
        rank=kernel(u,(u.T@vu)**2@m)
        perm=np.roll(np.roll(live-fixed,17,axis=1),17,axis=2)
        session=sample[(finish<=boundary)&(origin>=(2401 if boundary>=2401 else 0))]
        local,ms=update(st['prior'],session,u,m)
        last=live.copy()
        last[:2]=0
        mats=(fixed,live,live-fixed,stable,rotated-fixed,rank-fixed,agree-prior_sq,perm,kernel(local,ms)-fixed,last)
        for j,mat in enumerate(mats):
            out[ids,:,j]=np.einsum('isr,trib->tsb',mat.astype('float32'),payoff[ids],optimize=True)
    return trial.clock.center(out.reshape(len(t),50,-1))


def historical_design(d):
    st=state(max(4,d-d%2))
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=features(raw,t,st)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    s=[np.arange(3+3*i,6+3*i) for i in range(10)]
    return dict(**spectrum.parent.clocks.linear.gram(rows,[(g,3,.43) for g in s+[s[2][1:]]]),**state(len(rows)+4))


changes=spectrum.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st=dict(f)
    def apply(raw,p):
        x=features(raw,np.arange(len(raw)),st)
        return changes(np.concatenate((trial.clock.center(p),x),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
