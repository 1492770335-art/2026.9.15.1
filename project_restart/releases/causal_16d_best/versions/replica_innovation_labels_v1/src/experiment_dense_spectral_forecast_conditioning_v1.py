"""Estimate forward uncertainty after conditioning the cash spectrum on forecasts known at issue."""
from functools import lru_cache
import sys
import numpy as np
import torch

import experiment_dense_multitaper_common_surprises_v1 as spectral
import experiment_dense_spectral_forward_cash_v1 as forward
import prepare_dense_spectral_forward_cash_reference_v1 as ref
from experiment_dense_market_correction_rollforward_v1 import event,onset,rel,strip,official
from experiment_dense_disjoint_settled_error_transfer_v1 import units


REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_spectral_forecast_conditioning_v1'
ARMS=('tail_risk_after_forecast_controls','risk_removed_by_live_quotes','lagged_innovation_after_quotes')
# Columns of the cross-spectral matrix multiply three cash windows, the
# quotes issued at the origin, and innovations completed strictly before it.
SOURCE=np.array([0,0,0,1,2,1,0])
omega=2j*np.pi*spectral.FREQ
TRANSFORM=np.column_stack((forward.LEGS,np.ones((len(omega),2)),1-np.exp(-omega),
                           np.exp(-omega[:,None]*np.arange(1,5)).sum(1)))


def read_p(day):
    p=ref.read_p(day)
    if 8<=day<16:
        p=p.copy()
        p[...,0]+=official.short_market_changes()[day][:,None]
    return p


def spectrum(x):
    v=x-x.mean(0)
    return np.fft.rfft(spectral.TAPERS[:,:,None,None]*v[None],axis=1)


def covariance(v,shared):
    product=np.einsum('kfsc,kfsd->fcd',v.conj(),v,optimize=True)
    if shared:
        total=v.sum(2)
        product=(np.einsum('kfc,kfd->fcd',total.conj(),total,optimize=True)-product)/(v.shape[2]-1)
    product*=spectral.MULT[:,None,None]/(v.shape[0]*v.shape[2])
    terms=product[:,SOURCE[:,None],SOURCE[None,:]]
    g=np.einsum('fij,fi,fj->ij',terms,TRANSFORM.conj(),TRANSFORM,optimize=True).real
    sd=np.sqrt(g.diagonal().clip(.00001))
    corr=g/np.outer(sd,sd)
    eig,u=np.linalg.eigh((corr+corr.T)*.5)
    return np.outer(sd,sd)*((u*eig.clip(.025))@u.T)


def readings(g):
    vals=[]
    for j in (1,2):
        def conditional(ids):
            ix=np.array(ids)
            mat=g[np.ix_(ix,ix)]
            target=g[ix,j]
            w=np.linalg.solve(mat+.25*np.diag(mat.diagonal().clip(1e-8)),target)
            return max(1e-8,g[j,j]-2*w@target+w@mat@w),w
        legs=list(range(j))
        simple,_=conditional(legs)
        full,_=conditional(legs+[3,4,5])
        _,w=conditional(legs+[3,4,5,6])
        denom=g[:j,:j].sum().clip(1e-7)
        vals.append((np.log(.05+full/denom),np.log(simple/full),w[-1]))
    return np.array(vals)


def blocks(m,ok,p,unit):
    path=m.cumsum(0)
    gaps=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t=np.arange(5,len(p),5)
    good=gaps[t+1]==gaps[t-5]
    short=p[t-5,:,0]*unit[0]
    error=path[t]-path[t-5]-short
    marks=np.stack((error,short,p[t-5,:,1]*unit[1]/4),axis=-1)
    market=(marks*good[...,None]).sum(1)/good.sum(1).clip(1)[:,None]
    local=(marks-market[:,None])*good[...,None]
    ids=spectral.BASKETS
    baskets=(marks[:,ids]*good[:,ids,None]).sum(2)/good[:,ids].sum(2).clip(1)[...,None]
    past=np.vstack((np.zeros((1,3)),market.cumsum(0)))
    for lo in range(5*(spectral.N+1),len(p),120):
        stop=t.searchsorted(lo)
        ix=slice(stop-spectral.N,stop)
        common=readings(covariance(spectrum(baskets[ix]),True))
        specific=readings(covariance(spectrum(local[ix]),False))
        hi=min(lo+120,len(p))
        k=t.searchsorted(np.arange(lo,hi),side='right')
        recent=past[k,0]-past[k-4,0]
        yield lo,hi,common-specific,recent


@lru_cache(maxsize=48)
def record(d,unit):
    p=read_p(d)
    raw,grid=[onset.settled.load(d,key) for key in ('x','time')]
    states=[z for a,b,m,ok in event.sessions(raw,grid) for _,_,z,_ in blocks(m,ok,p[a:b],unit)]
    return np.median(states,axis=0)


def train(end):
    unit=tuple(np.median([units(d) for d in range(4,end)],axis=0))
    return np.r_[unit,np.median([record(d,unit) for d in range(4,end)],axis=0).ravel()]


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    unit=bank[:3]
    center=bank[3:].reshape(2,3)
    for a,b,m,ok in event.sessions(raw,grid):
        market=p[a:b].mean(1,dtype=float)*unit
        for lo,hi,z,recent in blocks(m,ok,p[a:b],unit):
            w=.20*np.tanh(z-center)
            for j in (1,2):
                tail=(market[lo:hi,j]-market[lo:hi,j-1])/unit[j]
                delta=np.column_stack((w[j-1,0]*tail,w[j-1,1]*tail,w[j-1,2]*recent/unit[j]))
                out[a+lo:a+hi,:,:,j-1]=delta[:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
