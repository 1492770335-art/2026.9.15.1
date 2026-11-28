"""Retrieve the observed cash payoff of earlier market forecasts in comparable volatility and lag states."""
from functools import lru_cache
import sys
import numpy as np
import torch

from experiment_dense_directional_market_persistence_v1 import BLOCK, REFERENCE, read_p, event, onset, rel, strip
from experiment_dense_disjoint_settled_error_transfer_v1 import units
from experiment_dense_block_information_weight_v1 import stats

OUT=REFERENCE.parent/'dense_analogue_block_payoff_v1'
ARMS=('matched_historical_cash','extra_lag_state_information','intraday_cash_update')
K=5


def descriptors(move,ok,p,unit):
    path=move.cumsum(0)
    gaps=(~ok).cumsum(0)
    acc=np.zeros((2,2))
    for lo in range(BLOCK,len(p),BLOCK):
        t=np.arange(lo-BLOCK+5,lo,5)
        valid=(gaps[t]==gaps[t-5])&ok[t-5]
        err=path[t]-path[t-5]-p[t-5,:,0]*unit[0]
        err=(err-(err*valid).sum(0)/valid.sum(0).clip(1))*valid
        var,lag=stats(err)
        acc=.72*acc+np.array((var,lag))
        if lo>=2*BLOCK:
            key=np.r_[lo/1200,np.log(acc[0]+1e-6),acc[1]/(acc[0]+1e-6)]
            yield lo,min(lo+BLOCK,len(p)),key


def reward(move,ok,p,unit):
    # Called after a whole block has closed. Only origins whose two
    # settlements fall inside that same block contribute a measurement.
    path=move.cumsum(0)
    gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t=np.arange(0,len(move)-60,5)
    moments=[]
    for j,h in enumerate((20,60),start=1):
        valid=gap[t+h+1]==gap[t]
        z=(p[t,:,j]*valid).sum(1)/valid.sum(1).clip(1)
        r=((path[t+h]-path[t])*valid).sum(1)/valid.sum(1).clip(1)/unit[j]
        w=(valid.sum(1)>=38)
        moments.append([np.sum(w*z*(r-z)),np.sum(w*z*z)])
    return np.array(moments)


@lru_cache(maxsize=48)
def record(day,unit):
    p=read_p(day)
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    rows=[]
    for a,b,m,ok in event.sessions(raw,grid):
        for lo,hi,key in descriptors(m,ok,p[a:b],unit):
            if hi-lo==BLOCK:
                r=reward(m[lo:hi],ok[lo:hi],p[a+lo:a+hi],unit)
                rows.append(np.r_[key,r.ravel()])
    return np.stack(rows)


def train(end):
    unit=tuple(np.median([units(d) for d in range(4,end)],axis=0))
    data=np.concatenate([record(d,unit) for d in range(4,end)])
    sd=(np.percentile(data[:,:K],75,axis=0)-np.percentile(data[:,:K],25,axis=0)).clip(.15)
    return np.r_[unit,sd,data.ravel()]


def lookup(key,rows,sd):
    delta=((rows[:,:K]-key)/sd).clip(-5,5)
    d=delta**2
    prior=np.exp(-.3*d[:,0])
    vol=prior*np.exp(-.5*d[:,1:3].sum(1))
    matched=vol*np.exp(-.5*d[:,3:].sum(1))
    pay=rows[:,K:].reshape(-1,2,2)
    def slope(w):
        # Keep the leverage of a single historical price block bounded.
        num,den=np.einsum('n,nhq->hq',w,pay).T
        return num/(den+4.)
    return slope(matched),slope(vol),slope(prior)


def signals(raw,grid,p,bank):
    unit,sd=bank[:3],bank[3:3+K]
    history=bank[3+K:].reshape(-1,K+4)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        market=p[a:b,:,1:].mean(1,dtype=float)
        live=[]
        previous=None
        for lo,hi,key in descriptors(m,ok,p[a:b],unit):
            if previous is not None:
                start,prior_key=previous
                r=reward(m[start:lo],ok[start:lo],p[a+start:a+lo],unit)
                live.append(np.r_[prior_key,r.ravel()])
            expected,vol,clock=lookup(key,history,sd)
            update=np.zeros(2)
            if len(live)>=3:
                nearby,_,avg=lookup(key,np.stack(live),sd)
                update=nearby-avg
            coeff=.18*np.tanh(np.stack((expected-clock,expected-vol,update)))
            out[a+lo:a+hi]=market[lo:hi,None,None,:]*coeff
            previous=lo,key
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
