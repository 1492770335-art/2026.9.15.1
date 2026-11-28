"""Predict common-versus-stock risk from lagged cash covariances observed on disjoint stock panels."""
from functools import lru_cache
import sys
import numpy as np
import torch

import experiment_dense_common_phase_correction_v1 as phase
import prepare_dense_basket_release_reference_v1 as official

strip,event,onset,rel=phase.strip,phase.event,phase.onset,phase.rel
REFERENCE,read_p=official.OUT,official.read_p
OUT=REFERENCE.parent/'dense_market_covariance_signature_v1'
ARMS=('short_covariance_share','delayed_common_price_moves','independent_panels_confirm')
GROUP=np.random.default_rng(28613).permutation(50).reshape(2,25)
BLOCK=120

def covariance(z,h):
    z=z-z.mean(0)
    mean=z.mean(1,keepdims=True)
    rel=z-mean
    def sumcov(x):
        v=(x*x).mean()
        for lag in range(1,min(7,h,len(x))):
            # A tapered lag window suppresses high-variance estimates from a short block.
            v+=2*(1-lag/h)*(1-lag/7)*(x[lag:]*x[:-lag]).mean()
        return v
    return np.array([sumcov(mean),sumcov(rel)]).clip(1e-9)

def paths(p,m):
    path=m.cumsum(0)
    sq=np.zeros((2,3,2,2));forecast=np.zeros((2,3,2))
    pieces=[]
    for start in range(0,len(p),BLOCK):
        stop=min(start+BLOCK,len(p))
        if start:
            t=np.arange(start-BLOCK+5,start,5)
            inc=path[t]-path[t-5]
            for j,h in enumerate((4,12)):
                for k,ids in enumerate((GROUP[0],GROUP[1],np.arange(50))):
                    x=inc[:,ids]
                    z=p[t][:,ids,j+1].astype(float)
                    for l,window in enumerate((1,h)):
                        sq[j,k,l]=.77*sq[j,k,l]+covariance(x,window)
                    forecast[j,k]=.77*forecast[j,k]+covariance(z,1)
        if start<2*BLOCK:
            pieces.append((start,stop,None));continue
        def ratio(v):
            return np.log(v[...,0]/v[...,1])
        balance=.5*(ratio(sq)-ratio(forecast)[:,:,None])
        pieces.append((start,stop,balance))
    return pieces

@lru_cache(maxsize=12)
def record(d):
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    p=read_p(d)
    rows=[v for a,b,m,_ in event.sessions(raw,grid) for _,_,v in paths(p[a:b],m) if v is not None]
    return np.median(np.stack(rows),axis=0)

def train(end):
    return np.median(np.stack([record(d) for d in range(4,end)]),axis=0)

def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b,:,1:].mean(1,dtype=float)
        for start,stop,observed in paths(p[a:b],m):
            if observed is None:continue
            u=observed-bank
            short=u[:,2,0]
            extra=u[:,2,1]-u[:,2,0]
            left,right=u[:,0,1],u[:,1,1]
            confirm=np.where(left*right>0,np.sign(left)*np.minimum(abs(left),abs(right)),0.)
            coefficient=.20*np.tanh(np.stack((short,extra,confirm)))
            out[a+start:a+stop]=q[start:stop,None,None,:]*coefficient[None,None]
    return out

def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)

if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
