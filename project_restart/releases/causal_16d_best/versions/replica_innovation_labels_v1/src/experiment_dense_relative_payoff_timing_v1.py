"""Use other stocks' within-group forecast settlement timing to calibrate a recipient's relative long return."""
from functools import lru_cache
import sys

import numpy as np
import torch

import prepare_dense_timing_budget_reference_v1 as ref
import experiment_dense_forecast_payoff_timing_v1 as timing
import experiment_dense_multitaper_common_surprises_v1 as spec
from experiment_dense_cluster_lag_confirmation_v1 import groups
from experiment_dense_market_correction_rollforward_v1 import event,onset,rel,strip,official
from experiment_dense_disjoint_settled_error_transfer_v1 import units


REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_relative_payoff_timing_v1'
ARMS=('outside_group_timing','incremental_group_timing','near_neutral_replicated_timing')


def read_p(d):
    q=ref.read_p(d)
    if 8<=d<16:
        q=q.copy()
        q[...,0]+=official.short_market_changes()[d][:,None]
    return q


def blocks(m,ok,p,unit,ids):
    path=m.cumsum(0)
    gaps=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t=np.arange(5,len(m),5)
    good=gaps[t+1]==gaps[t-5]
    x=np.concatenate((p[t]*unit,(path[t]-path[t-5])[...,None]),axis=2)[:,ids]
    mask=good[:,ids]
    mean=(x*mask[...,None]).sum(2)/mask.sum(2).clip(1)[...,None]
    local=(x-mean[:,:,None])*mask[...,None]
    for lo in range(5*(spec.N+1),len(m),120):
        stop=t.searchsorted(lo)
        f=spec.spectral(local[stop-spec.N:stop].reshape(spec.N,-1)).reshape(4,-1,5,10,4)
        s=np.einsum('kfgsi,kfgsj->gfij',f.conj(),f,optimize=True)*spec.MULT[None,:,None,None]/40
        state=np.empty((5,2,2))
        weights=np.zeros((5,2,2))
        for k in range(5):
            donor=(s.sum(0)-s[k])/4
            for j in (1,2):
                foreign,w=timing.prices(donor,j)
                own,_=timing.prices(s[k],j)
                state[k,j-1]=(foreign[1],own[1])
                weights[k,j-1,:j]=w
        yield lo,min(lo+120,len(m)),state,weights


@lru_cache(maxsize=48)
def record(day,unit):
    p=read_p(day)
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    ids=groups().reshape(5,10)
    s=[z for a,b,m,ok in event.sessions(raw,grid) for _,_,z,_ in blocks(m,ok,p[a:b],unit,ids)]
    return np.median(s,axis=0)


def train(end):
    unit=tuple(np.median([units(d) for d in range(4,end)],axis=0))
    s=np.median([record(d,unit) for d in range(4,end)],axis=0)
    return np.r_[unit,groups().ravel(),s.ravel()]


def signals(raw,grid,p,bank):
    unit=bank[:3]
    ids=bank[3:53].reshape(5,10).astype(int)
    prior=bank[53:].reshape(5,2,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        quote=strip.center(p[a:b].astype(float))*unit
        for lo,hi,z,w in blocks(m,ok,p[a:b],unit,ids):
            forward,inside=np.moveaxis(z-prior,2,0)
            agree=(forward*inside>0)*np.sign(forward)*np.minimum(abs(inside),abs(forward))
            rates=.22*np.tanh(2*np.stack((forward,inside-forward,agree),axis=2))
            block=out[a+lo:a+hi]
            for j in (1,2):
                for k,names in enumerate(ids):
                    q=quote[lo:hi,names]
                    unique=(q[:,:,j]-q[:,:,:j]@w[k,j-1,:j])/unit[j]
                    block[:,:,:,j-1][:,names]=unique[...,None]*rates[k,j-1]
                block[:,:,:,j-1]-=block[:,:,:,j-1].mean(1,keepdims=True)
                # Protect the lower maturity at the decision tick even when
                # the timing coefficient differs across groups.
                near=quote[lo:hi,:,:j]
                z3=block[:,:,2:3,j-1]
                gram=near.swapaxes(1,2)@near
                ridge=.08*gram.diagonal(axis1=1,axis2=2).clip(.00001)[:,None,:]*np.eye(j)
                z3-=near@np.linalg.solve(gram+ridge,near.swapaxes(1,2)@z3)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
