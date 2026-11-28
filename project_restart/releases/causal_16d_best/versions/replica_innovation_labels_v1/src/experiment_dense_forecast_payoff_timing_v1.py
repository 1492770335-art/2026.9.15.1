"""Measure when each forecast's unique component is paid, using already observed cross-basket returns."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_multitaper_common_surprises_v1 as spec
from experiment_dense_spectral_forecast_conditioning_v1 import REFERENCE,read_p,event,onset,rel,strip,units,spectrum


OUT=REFERENCE.parent/'dense_forecast_payoff_timing_v1'
ARMS=('forward_payoff_share','time_reversal_contrast','forecast_vintage_shift')
LAG=np.arange(-12,13)
SHIFT=np.exp(2j*np.pi*spec.FREQ[:,None]*LAG)


def spectrum_matrix(z,shared):
    f=spectrum(z)
    own=np.einsum('kfsc,kfsd->fcd',f.conj(),f,optimize=True)
    if shared:
        total=f.sum(2)
        own=(np.einsum('kfc,kfd->fcd',total.conj(),total,optimize=True)-own)/(f.shape[2]-1)
    return own*(spec.MULT/(f.shape[0]*f.shape[2]))[:,None,None]


def prices(s,j):
    # Remove the shorter forecast exposures from the long price quote before
    # measuring its covariance with disjoint, positive- and negative-lag cash.
    g=s.sum(0).real
    a=g[:j,:j]
    sd=np.sqrt(a.diagonal().clip(.00001))
    c=a/np.outer(sd,sd)
    eig,v=np.linalg.eigh((c+c.T)*.5)
    inv=((v/(eig.clip(.01)+.22))@v.T)/np.outer(sd,sd)
    w=(inv@g[:j,j]).clip(-4,4)
    component=np.eye(4)[j]
    component[:j]=-w
    cov=((s[:,:,3]@component)@SHIFT).real
    scale=np.sqrt(max(component@g@component,1e-5)*max(g[3,3],1e-5))
    cov=cov/scale
    near,full=((1,4),(4,12))[j-1]
    def leg(start,stop):
        return cov[(LAG>=start)&(LAG<=stop)].sum()/np.sqrt(stop-start+1)
    early=leg(1,near)
    late=leg(near+1,full)
    reflected=leg(-full,-near-1)
    return np.r_[late-early,late-reflected,late/(.35+abs(late)+abs(early))],w


def blocks(m,ok,p,unit):
    path=m.cumsum(0)
    gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t=np.arange(5,len(m),5)
    good=gap[t+1]==gap[t-5]
    cash=path[t]-path[t-5]
    obs=np.concatenate((p[t]*unit,cash[...,None]),axis=-1)
    cm=(obs*good[...,None]).sum(1)/good.sum(1).clip(1)[:,None]
    local=(obs-cm[:,None])*good[...,None]
    ids=spec.BASKETS
    baskets=(obs[:,ids]*good[:,ids,None]).sum(2)/good[:,ids].sum(2).clip(1)[...,None]
    for lo in range(5*(spec.N+1),len(p),120):
        stop=t.searchsorted(lo)
        sl=slice(stop-spec.N,stop)
        common=spectrum_matrix(baskets[sl],True)
        specific=spectrum_matrix(local[sl],False)
        state=[]
        weights=[]
        for j in (1,2):
            timing,w=prices(common,j)
            relative,_=prices(specific,j)
            state.append(timing-relative)
            weights.append(w)
        yield lo,min(lo+120,len(p)),np.stack(state),weights


@lru_cache(maxsize=48)
def record(d,unit):
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    p=read_p(d)
    rows=[z for a,b,m,ok in event.sessions(raw,grid) for _,_,z,_ in blocks(m,ok,p[a:b],unit)]
    return np.median(rows,axis=0)


def train(end):
    unit=tuple(np.median([units(d) for d in range(4,end)],axis=0))
    return np.r_[unit,np.median([record(d,unit) for d in range(4,end)],axis=0).ravel()]


def signals(raw,grid,p,bank):
    unit=bank[:3]
    center=bank[3:].reshape(2,3)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b].mean(1,dtype=float)*unit
        for lo,hi,z,weights in blocks(m,ok,p[a:b],unit):
            state=.24*np.tanh((z-center)*1.5)
            for j,w in enumerate(weights,1):
                origin=q[lo:hi]
                old=q[lo-10:hi-10]
                unique=(origin[:,j]-origin[:,:j]@w)/unit[j]
                prev=(old[:,j]-old[:,:j]@w)/unit[j]
                delta=state[j-1]*np.column_stack((unique,unique,prev-unique))
                out[a+lo:a+hi,:,:,j-1]=delta[:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
