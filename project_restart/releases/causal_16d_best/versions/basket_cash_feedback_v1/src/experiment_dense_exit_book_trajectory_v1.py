"""Measure the delayed return associated with book changes since an open price break and its extremes."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import prepare_dense_live_barrier_reference_v1 as best

barrier = best.model
event,strip,rel,onset = barrier.event,barrier.strip,barrier.rel,barrier.onset
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_exit_book_trajectory_v1'
ARMS = ('book_since_break','book_since_extremes','peer_book_since_extremes')
K = 12
C = 8+2*K
D = C+4*K


def coordinates(raw,t,stocks=None):
    x = raw[t] if stocks is None else raw[t,stocks]
    return np.tanh(.65*np.arcsinh(x[...,:83].astype(np.float32)))


@lru_cache(maxsize=1)
def dictionary():
    g = np.zeros((83,83))
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b in rel.fields.parent.sessions(grid):
            t = np.arange(a+12+d,b,8)
            x = (coordinates(raw,t)-coordinates(raw,t-12)).reshape(-1,83).astype(float)
            g += x.T@x/len(x)
    scale = np.sqrt(g.diagonal()/8).clip(.04)
    ev,v = np.linalg.eigh(g/(8*np.outer(scale,scale)))
    return v[:,-K:]/scale[:,None]/np.sqrt(ev[-K:]+.1)


def extrema(move,start,sign):
    path = move.cumsum(0)
    best_tick = np.zeros_like(start)
    worst_tick = np.zeros_like(start)
    stock = np.arange(50)
    for i in range(len(move)):
        reset = start[i]==i
        past_best = np.where(reset,i,best_tick[i-1])
        past_worst = np.where(reset,i,worst_tick[i-1])
        best_tick[i] = np.where(sign[i]*(path[i]-path[past_best,stock])>0,i,past_best)
        worst_tick[i] = np.where(sign[i]*(path[i]-path[past_worst,stock])<0,i,past_worst)
    return best_tick,worst_tick


def design(raw,move,ok,p,floor,edges,axes,t):
    start,age,sign,frac,width = barrier.active(move,ok,floor)
    high,low = extrema(move,start,sign)
    # Project before gathering snapshots so a tick used by many origins is transformed only once.
    book = coordinates(raw,np.arange(len(move)))@axes
    stock = np.arange(50)
    cur = book[t]
    active = sign[t]!=0
    def change(clock):
        return (cur-book[clock[t],stock])*active[...,None]
    start_diff,high_diff,low_diff = (change(clock) for clock in (start,high,low))
    gap = high_diff-low_diff
    peer = (gap.transpose(0,2,1)@edges.T).transpose(0,2,1)
    s = sign[t]
    meta = np.stack((s,s*age[t]/60,s*frac[t],s*(width[t]/floor.clip(.001)).clip(0,12),
                     move[t]/floor.clip(.001)),axis=2)
    fixed = cur-book[(t-12).clip(0)]
    return strip.center(np.concatenate((p[t],meta,cur,fixed,start_diff,
                                      high_diff,low_diff,peer*active[...,None]),axis=2))


@lru_cache(maxsize=12)
def record(d):
    floor,edges = event.geometry()
    axes = dictionary()
    raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
    p = read_p(d)
    total = np.zeros((D+2,D+2))
    count = 0
    for a,b,move,ok in event.sessions(raw,grid):
        t = np.arange(60+d%5,b-a-60,5)
        x = design(raw[a:b],move,ok,p[a:b],floor,edges,axes,t)
        path = move.cumsum(0)
        cash = np.stack((path[t+20]-path[t],path[t+60]-path[t+20]),axis=2)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+61]==gaps[t]
        row = np.concatenate((x,cash),axis=2)
        row -= ((row*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        r = row[valid]
        total += r.T@r
        count += len(r)
    return total/count


def fit(days,j):
    g = days.mean(0)
    groups = (np.arange(C,C+K),np.arange(C+K,C+3*K),np.arange(C+3*K,D))
    idx = groups[j]
    ctrl = list(range(C if j==0 else (C+K if j==1 else C+3*K)))
    v = np.eye(D)[:,idx]
    v[ctrl] = -barrier.parent.solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.06)
    unit = ((days[:,2,D]+days[:,2,D+1])/days[:,2,2]).clip(.05,10)
    short = (days[:,1,D]/days[:,1,1]).clip(.05,10)
    # Remove the late portion already forecast by the difference of the two incumbent horizons.
    residual = days[:,:D,D+1]/unit[:,None]-days[:,:D,2]+days[:,:D,1]*(short/unit)[:,None]
    cross = residual@v
    mu = cross.mean(0)
    uncertainty = cross.var(0)/len(days)
    return v@barrier.parent.solve(v.T@g[:D,:D]@v,mu**3/(mu**2+uncertainty+1e-9),.8)


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    w = np.stack([fit(days,j) for j in range(3)],axis=1)
    floor,edges = event.geometry()
    return np.r_[floor,edges.ravel(),dictionary().ravel(),w.ravel()]


def signals(raw,grid,p,bank):
    floor,edges = bank[:50],bank[50:2550].reshape(50,50)
    axes,w = bank[2550:-D*3].reshape(83,K),bank[-D*3:].reshape(D,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        x = design(raw[a:b],move,ok,p[a:b],floor,edges,axes,np.arange(b-a))
        out[a:b,:,:,1] = x@w
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
