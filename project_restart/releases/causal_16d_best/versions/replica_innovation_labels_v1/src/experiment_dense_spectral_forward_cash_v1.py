"""Infer the covariance of disjoint forward return intervals from the observed short-error cross-spectrum."""
from functools import lru_cache
import sys
import numpy as np
import torch

import experiment_dense_multitaper_common_surprises_v1 as spectral
import prepare_dense_multitaper_surprise_reference_v1 as ref
from experiment_dense_market_correction_rollforward_v1 import event,onset,rel,strip,official
from experiment_dense_disjoint_settled_error_transfer_v1 import units


REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_spectral_forward_cash_v1'
ARMS=('tail_projected_on_near','separate_early_interval_slopes','unexplained_tail_risk')
PHASE=np.exp(2j*np.pi*spectral.FREQ[:,None]*np.arange(12))
LEGS=np.stack([PHASE[:,a:b].sum(1) for a,b in ((0,1),(1,4),(4,12))],axis=1)
BASIS=(LEGS.conj()[:,:,None]*LEGS[:,None,:]).real


def read_p(day):
    p=ref.read_p(day)
    if 8<=day<16:
        p=p.copy()
        p[...,0]+=official.short_market_changes()[day][:,None]
    return p


def slopes(power):
    g=np.einsum('f,fij->ij',power,BASIS)
    # A signed cross-periodogram can give negative eigenvalues. Shrink its
    # symmetric covariance before solving for maturity-to-maturity slopes.
    val,u=np.linalg.eigh(g)
    g=(u*val.clip(1e-7))@u.T
    result=[]
    for j in (1,2):
        sub=g[:j,:j]
        floor=.20*np.diag(sub.diagonal().clip(1e-6))
        coef=np.linalg.solve(sub+floor,g[:j,j])
        pooled=g[:j,j].sum()/(sub.sum()+floor.sum())
        conditional=g[j,j]-2*coef@g[:j,j]+coef@sub@coef
        unexplained=np.log(.05+conditional/max(.00001,sub.sum()))
        result.append(np.r_[pooled,coef[0],coef[-1],unexplained])
    return np.stack(result)


def blocks(m,ok,p,unit):
    path=m.cumsum(0)
    gaps=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    ticks=np.arange(5,len(m),5)
    good=gaps[ticks+1]==gaps[ticks-5]
    error=path[ticks]-path[ticks-5]-p[ticks-5,:,0]*unit[0]
    for lo in range(5*(spectral.N+1),len(p),120):
        stop=ticks.searchsorted(lo)
        ix=slice(stop-spectral.N,stop)
        cross,stock=spectral.spectra(error[ix],good[ix])
        yield lo,min(lo+120,len(p)),slopes(cross.mean(0))-slopes(stock)


@lru_cache(maxsize=48)
def record(d,unit):
    p=read_p(d)
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    z=[v for a,b,m,ok in event.sessions(raw,grid) for _,_,v in blocks(m,ok,p[a:b],unit)]
    return np.median(z,axis=0)


def train(end):
    unit=tuple(np.median([units(d) for d in range(4,end)],axis=0))
    return np.r_[unit,np.median([record(d,unit) for d in range(4,end)],axis=0).ravel()]


def signals(raw,grid,p,bank):
    unit=bank[:3]
    center=bank[3:].reshape(2,4)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b].mean(1,dtype=float)*unit
        for lo,hi,state in blocks(m,ok,p[a:b],unit):
            slope=.22*np.tanh(state-center)
            for j in (1,2):
                near=q[lo:hi,j-1]
                first=q[lo:hi,0]
                tail=q[lo:hi,j]-near
                total=slope[j-1,0]*near
                split=slope[j-1,1]*first+slope[j-1,2]*(near-first)
                risk=slope[j-1,3]*tail
                out[a+lo:a+hi,:,:,j-1]=np.column_stack((total,split,risk))[:,None]/unit[j]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
