"""Bound the market-wide timing update by the past realized volatility of its horizon's forecast error."""
from functools import lru_cache
import sys
import numpy as np
import torch

import experiment_dense_replicated_timing_hedge_v1 as replica
import experiment_dense_near_neutral_timing_update_v1 as hedge
from experiment_dense_spectral_forecast_conditioning_v1 import REFERENCE,read_p,event,onset,rel,strip,units


OUT=REFERENCE.parent/'dense_timing_covariance_budget_v1'
ARMS=('market_error_risk_cap','bounded_covariance_per_issue','market_share_risk_cap')


def risks(m,ok,p,unit):
    path=m.cumsum(0)
    gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    for lo in range(485,len(m),120):
        hi=min(lo+120,len(m))
        values=[]
        for j,h in enumerate((20,60),1):
            t=np.arange(max(0,lo-480-h),lo-h,5)
            good=gap[t+h+1]==gap[t]
            n=good.sum(1,keepdims=True)
            errors=(path[t+h]-path[t])/unit[j]-p[t,:,j]
            common=(errors*good).sum(1,keepdims=True)/n.clip(1)
            local=errors-common
            a=common[n>=38]
            v=np.sqrt(a.var()+.0002) if len(a) else .03
            relrisk=np.sqrt((local**2*good).sum()/n.sum().clip(1)+.0002)
            values.append((v,v/relrisk))
        yield lo,hi,np.array(values)


@lru_cache(maxsize=48)
def record(d,unit):
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    p=read_p(d)
    records=[s for a,b,m,ok in event.sessions(raw,grid) for _,_,s in risks(m,ok,p[a:b],unit)]
    return np.median(records,axis=0)


def train(end):
    base=hedge.train(end)
    unit=tuple(base[:3])
    median=np.median([record(d,unit) for d in range(4,end)],axis=0)
    return np.r_[base,median.ravel()]


def signals(raw,grid,p,bank):
    base=bank[:-4]
    center=bank[-4:].reshape(2,2)
    signed=replica.signals(raw,grid,p,base)[:,0,2,:]
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        acc=np.zeros(2)
        elapsed=0.
        for lo,hi,risk in risks(m,ok,p[a:b],base[:3]):
            alpha=risk/center.clip(.008)
            z=signed[a+lo:a+hi]
            # Regularize the initial budget with a small fraction of the
            # already observed market forecast RMS at this maturity.
            avg=p[a+max(0,lo-480):a+lo,:,1:].mean(1,dtype=float)
            floor=.035*np.sqrt(np.mean(avg**2,axis=0)).clip(.012)
            typical=np.sqrt((acc+floor**2)/(elapsed+1.)).clip(.0002)
            exposure=alpha[:,0].clip(.35,4.)
            cap=1.5*typical/exposure
            parts=np.stack((z/np.maximum(1.,exposure),
                            cap*np.tanh(z/cap),
                            z/np.maximum(1.,alpha[:,1])),axis=1)
            out[a+lo:a+hi]=parts[:,None]
            acc=.82*acc+(z*z).mean(0)
            elapsed=.82*elapsed+1
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
