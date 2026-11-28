"""Hedge the observed near-leg cost of a forward-timing adjustment with already settled short returns."""
from functools import lru_cache
import sys

import numpy as np
import torch
import experiment_dense_forecast_payoff_timing_v1 as timing
from experiment_dense_spectral_forecast_conditioning_v1 import REFERENCE,read_p,event,onset,rel,strip,units


OUT=REFERENCE.parent/'dense_near_neutral_timing_update_v1'
ARMS=('conditional_near_hedge','only_uncompensated_near_cost','separate_live_near_prices')
K=4


@lru_cache(maxsize=12)
def slopes(d):
    p=read_p(d).reshape(-1,3).astype(float)
    y=onset.settled.load(d,'y').reshape(-1,3).astype(float)
    _,c,_=strip.physical()
    p-=p.mean(0)
    y-=y.mean(0)
    # Derivative of the near contribution including the normalization of the full forecast.
    return np.array([(p[:,j]*y[:,j-1]).mean()/p[:,j].var().clip(1e-6)*c[j]/c[j-1]
                     for j in (1,2)])


def raw_signals(m,ok,p,params):
    unit=params[:3]
    center=np.reshape(params[3:9],(2,3))
    market=p.mean(1,dtype=float)*unit
    raw=np.zeros((len(p),2,2))
    for lo,hi,z,ws in timing.blocks(m,ok,p,unit):
        state=.24*np.tanh(1.5*(z-center))
        for j,w in enumerate(ws,1):
            q=market[lo:hi]
            raw[lo:hi,j-1,0]=state[j-1,1]*(q[:,j]-q[:,:j]@w)/unit[j]
            raw[lo:hi,j-1,1]=q[:,j-1]/unit[j]
    return raw


def observations(m,ok,p,raw,params,t,j):
    unit=params[:3]
    h=(5,20)[j]
    stop=t+h
    path=m.cumsum(0)
    gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    valid=gap[stop+1]==gap[t]
    cash=path[stop]-path[t]
    v=valid.sum(1).clip(1)
    near=(cash*valid).sum(1)/v/unit[j+1]
    q=(p[t,:,j+1]*valid).sum(1)/v
    # slope of the actual early label on the full forecast in full-forecast units
    tangent=near-params[9+j]*q
    x=raw[t,j]
    rows=np.column_stack((x,tangent,q))
    return rows[v>=38]


@lru_cache(maxsize=48)
def record(d,params):
    p=read_p(d)
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    g=np.zeros((2,K,K))
    n=np.zeros(2)
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b]
        z=raw_signals(m,ok,q,params)
        for j,h in enumerate((5,20)):
            t=np.arange(485+d%5,len(q)-h,5)
            if not len(t):continue
            x=observations(m,ok,q,z,params,t,j)
            g[j]+=x.T@x
            n[j]+=len(x)
    return g/n.clip(1)[:,None,None]


def train(end):
    bank=timing.train(end)
    _,c,_=strip.physical()
    # slopes() is in label units. Convert to the units of the live cash proxy.
    beta=np.median([slopes(d) for d in range(4,end)],axis=0)*(c[2]/c[1:])/bank[1:3]
    params=tuple(np.r_[bank,beta])
    prior=np.mean([record(d,params) for d in range(4,end)],axis=0)
    return np.r_[params,prior.ravel()]


def compensated(g,mode):
    # Hedge with the part of the near quote distinct from the current full
    # quote. This makes a forecast rescaling contribute nearly zero payoff.
    hedge=np.array((0.,1.,0.,-g[1,3]/(g[3,3]+.001)))
    base=np.array((1.,0.,0.,0.))
    payoff=g[:,2]
    raw=base@payoff
    price=hedge@payoff
    hedge_var=max(hedge@g@hedge,.00002)
    risk=np.sqrt(max(g[0,0],0.)/hedge_var)
    if mode==0:
        ratio=-(base@g@hedge)/(hedge_var+.0003)
    else:
        ratio=max(-raw,0.)*price/(price**2+.000004)
    return hedge*np.clip(ratio,-risk,risk)


def signals(raw,grid,p,bank):
    params=bank[:11]
    prior=bank[11:].reshape(2,K,K)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b]
        z=raw_signals(m,ok,q,params)
        avg=q.mean(1,dtype=float)
        gs=prior.copy()
        for lo in range(485,len(q),120):
            hi=min(lo+120,len(q))
            for j,h in enumerate((5,20)):
                t=np.arange(max(485,lo-120-h),lo-h,5)
                if len(t):
                    x=observations(m,ok,q,z,params,t,j)
                    if len(x):
                        gs[j]=.82*gs[j]+.18*(x.T@x/len(x))
                controls=np.column_stack((z[lo:hi,j],np.zeros(hi-lo),avg[lo:hi,j+1]))
                current=z[lo:hi,j,0]
                for k,(g,mode) in enumerate(((gs[j],0),(prior[j],1),(gs[j],1))):
                    out[a+lo:a+hi,:,k,j]=(current+controls@compensated(g,mode))[:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
