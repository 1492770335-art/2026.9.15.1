"""Read shared delayed price surprises from two disjoint panels within each historical return cluster."""
from functools import lru_cache
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiment_dense_market_correction_rollforward_v1 import REFERENCE,read_p,event,onset,rel,strip
from experiment_dense_disjoint_settled_error_transfer_v1 import units

OUT=REFERENCE.parent/'dense_cluster_lag_confirmation_v1'
ARMS=('cluster_persistence','directional_half_panel_transfer','real_minus_mixed_cluster')


@lru_cache(maxsize=16)
def cash_geometry(day):
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    g=np.zeros((50,50))
    for a,b,m,ok in event.sessions(raw,grid):
        path=m.cumsum(0)
        gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        t=np.arange(1,len(m)-20,20)
        valid=gap[t+21]==gap[t]
        cash=(path[t+20]-path[t])*valid
        cash=(cash-cash.sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1))*valid
        g+=cash.T@cash
    sd=g.diagonal().clip(1e-6)**.5
    return g/np.outer(sd,sd)


@lru_cache(maxsize=1)
def groups():
    g=np.mean([cash_geometry(d) for d in range(4)],axis=0)
    val,axis=np.linalg.eigh(g)
    z=axis[:,-6:]*np.sqrt(val[-6:])
    centers=[np.argmax((z*z).sum(1))]
    for _ in range(4):
        dist=((z[:,None]-z[centers])**2).sum(2)
        centers.append(np.argmax(dist.min(1)))
    c=z[centers]
    for _ in range(12):
        cost=((z[:,None]-np.repeat(c,10,axis=0))**2).sum(2)
        rows,slots=linear_sum_assignment(cost)
        group=np.stack([rows[slots//10==j] for j in range(5)])
        c=z[group].mean(1)
    # Independently split each cluster. The mixed control uses the exact same stocks.
    perm=np.random.default_rng(721455).permutation(10)
    return group[:,perm].reshape(5,2,5)


def moments(data,ids):
    v=data[:,ids].mean(3)
    v-=v.mean(0)
    q=v[:,:,::-1]
    den=(v*v).mean(0)
    lag=np.zeros_like(den)
    for k in range(1,7):
        lag+=(1-k/7)*(1-k/12)*(v[k:]*q[:-k]).mean(0)
    return np.stack((den,lag),axis=-1)


def blocks(m,ok,p,unit,group):
    path=m.cumsum(0)
    gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    mixed=group.transpose(2,1,0)
    g=np.zeros((2,5,2,2))
    for lo in range(120,len(m),120):
        t=np.arange(lo-115,lo,5)
        valid=gap[t+1]==gap[t-5]
        e=(path[t]-path[t-5]-unit*p[t-5,:,0])*valid
        e=(e-e.sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1))*valid
        g=.77*g+np.stack((moments(e,group),moments(e,mixed)))
        if lo>=240:
            den=g[...,0]
            scale=np.sqrt(den*den[:,:,::-1])+.4*den.mean((1,2),keepdims=True)+1e-7
            yield lo,min(lo+120,len(p)),g[...,1]/scale


@lru_cache(maxsize=48)
def record(day,unit):
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    p=read_p(day)
    rows=[z for a,b,m,ok in event.sessions(raw,grid) for _,_,z in blocks(m,ok,p[a:b],unit,groups())]
    return np.median(rows,axis=0)


def train(end):
    unit=float(np.median([units(d)[0] for d in range(4,end)]))
    return np.r_[unit,groups().ravel(),np.median([record(d,unit) for d in range(4,end)],axis=0).ravel()]


def signals(raw,grid,p,bank):
    group=bank[1:51].reshape(5,2,5).astype(int)
    mixed=group.transpose(2,1,0)
    base=bank[51:].reshape(2,5,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=strip.center(p[a:b,:,1:].astype(float))
        for lo,hi,obs in blocks(m,ok,p[a:b],bank[0],group):
            z=.25*np.tanh(2*(obs-base))
            w=z[0]
            ctrl=q[lo:hi,mixed].mean((2,3))
            half=q[lo:hi,group].mean(3)
            group_value=half.mean(2)*w.mean(1)[None,:,None]
            view=out[a+lo:a+hi]
            for j in range(5):
                ids=group[j].ravel()
                view[:,ids,0]=group_value[:,j,None]
                view[:,ids,2]=group_value[:,j,None]
                for k in (0,1):
                    view[:,group[j,k],1]=half[:,j,1-k,None]*(w[j,k]-w[j,1-k])
            for j in range(5):
                view[:,mixed[j].ravel(),2]-=ctrl[:,j,None]*z[1,j].mean()
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
