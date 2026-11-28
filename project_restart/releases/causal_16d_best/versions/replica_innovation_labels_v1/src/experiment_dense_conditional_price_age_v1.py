"""Estimate differential forecast reliability when a stock goes unusually long without repricing."""
from functools import lru_cache
import sys
import numpy as np
import torch
from scipy.signal import lfilter

from experiment_dense_market_and_relative_distal_v1 import REFERENCE, read_p, strip, event, onset, rel

OUT=REFERENCE.parent/'dense_conditional_price_age_v1'
ARMS=('stock_and_clock_age','live_age_deviation','unexpected_wait_and_peer_move')
BINS=8
BASE=8
D=12


def age_series(move,ok):
    t=np.arange(len(move))[:,None]
    updated=(abs(move)>.003)|~ok
    last=np.maximum.accumulate(np.where(updated,t,0),axis=0)
    return np.log1p(t-last),last


@lru_cache(maxsize=16)
def exposure(day):
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    sums=np.zeros((BINS,50,2))
    for a,b,m,ok in event.sessions(raw,grid):
        age,_=age_series(m,ok)
        slots=np.minimum(grid[a:b].astype(int)//600,BINS-1)
        for k in np.unique(slots):
            ix=slots==k
            sums[k]+=np.stack(((age[ix]*ok[ix]).sum(0),ok[ix].sum(0)),axis=1)
    return sums


def prior(end):
    g=sum(exposure(d) for d in range(end))
    shared=(g[:,:,0].sum(0)/g[:,:,1].sum(0).clip(1))
    return (g[:,:,0]+150*shared)/(g[:,:,1]+150)


def features(move,ok,q,clock,ages):
    age,last=age_series(move,ok)
    historical=ages[np.minimum(clock.astype(int)//600,BINS-1)]
    old=age-historical
    decay=np.exp(-1/240.)
    smooth=lfilter([0.,1.-decay],[1.,-decay],old*ok,axis=0)
    surprise=np.tanh((old-smooth)/1.4)
    excess=np.tanh(old/1.4)
    market=q.mean(1,keepdims=True)
    relative=q-market
    price=move.mean(1).cumsum()
    pending=price[:,None]-price[last]
    prediction=np.concatenate((relative[:,:,1:],np.broadcast_to(market[:,:,1:],relative[:,:,1:].shape)),axis=2)
    ctrl=np.concatenate((q,np.broadcast_to(market,q.shape),np.tanh(pending)[...,None],excess[...,None]),axis=2)
    weights=(excess,surprise,surprise*np.tanh(3*pending)*np.sign(q[:,:,2]))
    return [strip.center(np.concatenate((ctrl,prediction*z[...,None]),axis=2)*ok[...,None]) for z in weights]


@lru_cache(maxsize=36)
def moments(day,anchor):
    raw,grid,y=[onset.settled.load(day,k) for k in ('x','time','y')]
    p=read_p(day)
    ages=prior(anchor)
    g=np.zeros((3,D+2,D+2))
    slope=strip.daily_scale(p,y).clip(.2,4)
    for a,b,m,ok in event.sessions(raw,grid):
        t=np.arange(32+day%4,b-a-60,4)
        e=strip.center(y[a+t,:,1:]/slope[1:]-p[a+t,:,1:])
        xs=features(m,ok,p[a:b],grid[a:b],ages)
        for j,x in enumerate(xs):
            v=np.concatenate((x[t],e),axis=2)[ok[t]]
            g[j]+=v.T@v
    return g/max(1.,len(p)*50/4)


def train(end):
    gs=np.stack([moments(d,min(d,6)) for d in range(4,end)])
    weights=[]
    for j in range(3):
        g=gs[:,j].mean(0)
        sd=g.diagonal()[:D].clip(1e-7)**.5
        G=g[:D,:D]/np.outer(sd,sd)
        v=np.eye(D)[:,BASE:]
        v[:BASE]=-np.linalg.solve(G[:BASE,:BASE]+.08*np.eye(BASE),G[:BASE,BASE:])
        v/=sd[:,None]
        gram=v.T@g[:D,:D]@v
        grads=v.T@gs[:,j,:D,D:]
        mu=grads.mean(0)
        shrink=mu**2/(mu**2+2*grads.var(0)/len(gs)+1e-12)
        weights.append(v@np.linalg.solve(gram+.85*np.diag(gram.diagonal().clip(.01)),shrink*mu))
    return np.r_[prior(end).ravel(),np.stack(weights).ravel()]


def signals(raw,grid,p,bank):
    ages=bank[:BINS*50].reshape(BINS,50)
    weights=bank[BINS*50:].reshape(3,D,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        for j,x in enumerate(features(m,ok,p[a:b],grid[a:b],ages)):
            out[a:b,:,j]=x@weights[j]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
