"""Separate the lag signature of settled five-tick cash from that of its already-issued forecast."""
from functools import lru_cache
import sys
import numpy as np
import torch

import prepare_dense_market_covariance_signature_reference_v1 as ref
import prepare_dense_basket_release_reference_v1 as official
from experiment_dense_market_covariance_signature_v1 import BLOCK,strip,event,onset,rel

REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_market_unpriced_covariance_v1'
ARMS=('unexplained_persistence','forecast_precedes_cash_surprise','cash_precedes_forecast_revision')

def read_p(d):
    p=ref.read_p(d)
    if 8<=d<16:
        p=p.copy()
        p[...,0]+=official.short_market_changes()[d][:,None]
    return p

def component(x):
    mean=x.mean(1,keepdims=True)
    return (mean,x-mean)

def metrics(actual,issued):
    r=actual-actual.mean(0)
    f=issued-issued.mean(0)
    rr,ff=component(r),component(f)
    info=np.zeros((2,5))
    for j,(ret,pred) in enumerate(zip(rr,ff)):
        error=ret-pred
        info[j,0]=(ret*ret).mean()
        info[j,1]=(error*error).mean()
        for k in range(1,7):
            w=(1-k/12)*(1-k/7)
            info[j,2]+=2*w*(error[:-k]*error[k:]).mean()
            info[j,3]+=2*w*(pred[:-k]*error[k:]).mean()
            info[j,4]+=2*w*(error[:-k]*pred[k:]).mean()
    return info

def paths(p,m,scale):
    path=m.cumsum(0)
    mat=np.zeros((2,5))
    for start in range(0,len(m),BLOCK):
        stop=min(start+BLOCK,len(m))
        if start:
            t=np.arange(start-BLOCK+5,start,5)
            mat=.77*mat+metrics(path[t]-path[t-5],p[t-5,:,0].astype(float)*scale)
        if start<2*BLOCK:continue
        # A difference cancels the overall cash volatility shared by market and individual series.
        g=mat[:,2:]/(mat[:,:1]+.2*mat[:,1:2]+1e-6)
        yield start,stop,(g[0]-g[1]).clip(-4,4)

@lru_cache(maxsize=12)
def unit(d):
    p=read_p(d)
    y=onset.settled.load(d,'y')
    _,coef,_=strip.physical()
    return float(strip.daily_scale(p,y)[0].clip(.2,4)*coef[2]/coef[0])

@lru_cache(maxsize=48)
def record(d,scale):
    p=read_p(d)
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    v=[x for a,b,m,_ in event.sessions(raw,grid) for _,_,x in paths(p[a:b],m,scale)]
    return np.median(v,axis=0)

def train(end):
    scale=float(np.median([unit(d) for d in range(4,end)]))
    return np.r_[scale,np.median([record(d,scale) for d in range(4,end)],axis=0)]

def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,_ in event.sessions(raw,grid):
        market=p[a:b].mean(1,dtype=float)
        for lo,hi,x in paths(p[a:b],m,bank[0]):
            coeff=.18*np.tanh(x-bank[1:])
            q=market[lo:hi]
            # Residual cash persistence modifies the long estimate. Priced news
            # followed by a surprise is instead attributed to the shorter live signal.
            short=np.column_stack((q[:,0],q[:,1]))
            direct=q[:,1:]*coeff[0]
            ahead=short*coeff[1]
            reaction=(q[:,1:]-short)*coeff[2]
            out[a+lo:a+hi]=np.stack((direct,ahead,reaction),axis=1)[:,None]
    return out

def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)

if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
