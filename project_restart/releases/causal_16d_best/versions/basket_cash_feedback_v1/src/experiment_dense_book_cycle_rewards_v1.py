"""Separate price changes explained by a book-state potential from net returns around closed cycles."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import experiment_dense_book_price_phase_v1 as ref

base,strip,event,onset,rel = ref.base,ref.strip,ref.event,ref.onset,ref.rel
REFERENCE,read_p = ref.REFERENCE,ref.read_p
OUT = REFERENCE.parent/'dense_book_cycle_rewards_v1'
ARMS = ('endpoint_price','closed_cycle_reward','same_state_or_symmetric_reward')
S,C,D = 60,8,17


def state(raw,m,ok):
    q = np.tanh(base.depth(raw)@base.axis())
    slow = ref.window(q,20)/np.minimum(np.arange(1,len(q)+1),20)[:,None]
    imbalance = np.digitize(q,[-.65,-.22,.22,.65])
    motion = np.digitize(q-slow,[-.15,.15])
    move = ref.window(m,5)
    last = np.maximum.accumulate(np.where((abs(m)>.001)|~ok,np.arange(len(m))[:,None],0),axis=0)
    young = np.arange(len(m))[:,None]-last<=3
    code = ((imbalance*3+motion)*2+(move>=0))*2+young
    vol = np.sqrt(ref.window(m*m,120)/np.minimum(np.arange(1,len(q)+1),120)[:,None]+.0004)
    return code,vol,q


@lru_cache(maxsize=16)
def edges(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    total = np.zeros((2,S,S))
    for a,b,m,ok in event.sessions(raw,grid):
        ids,vol,_ = state(raw[a:b],m,ok)
        path = m.cumsum(0)
        t = np.arange(20+day%3,b-a-5,3)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+6]==gap[t]
        ret = path[t+5]-path[t]
        ret -= (ret*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)
        ret /= vol[t]*np.sqrt(5)
        code = (ids[t]*S+ids[t+5])[valid]
        total[0] += np.bincount(code,minlength=S*S).reshape(S,S)
        total[1] += np.bincount(code,weights=ret[valid],minlength=S*S).reshape(S,S)
    return total


def rewards(days):
    count,price = sum(edges(i) for i in days)
    # Pooled destinations and source rewards serve as priors for rarely observed edges.
    marginal = (count.sum(0)+1)/(count.sum()+S)
    transition = (count+35*marginal)/(count.sum(1,keepdims=True)+35)
    local = price.sum(1,keepdims=True)/(count.sum(1,keepdims=True)+1)
    r = (price+12*local)/(count+12)
    reverse_support = 2*count*count.T/(count+count.T+1.)
    skew = (r-r.T)/2
    # Minimize sum_ij w_ij (skew_ij - potential[j] + potential[i])^2.
    lap = np.diag(reverse_support.sum(1))-reverse_support
    phi = np.linalg.solve(lap+np.diag(.02*reverse_support.sum(1)+1),
                          -(reverse_support*skew).sum(1))
    gradient = phi[None,:]-phi[:,None]
    cycle = (skew-gradient)*(reverse_support/(reverse_support+20))
    remaining = r-gradient-cycle
    edge_pay = np.stack((gradient,cycle,remaining),axis=2)
    drift = (transition[...,None]*edge_pay).sum(1)
    cumulative = np.zeros_like(drift)
    forecasts = []
    for h in range(1,13):
        cumulative = drift+transition@cumulative
        if h in (1,4,12):
            forecasts.append(cumulative.copy())
    return np.stack(forecasts,axis=2),reverse_support,skew,gradient


def lookup(end):
    tables = [rewards(range(i,end,2))[0] for i in (0,1)]
    a,b = tables
    return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)


def features(raw,m,ok,p,table):
    code,vol,q = state(raw,m,ok)
    levels = table[code]*vol[:,:,None,None]*np.sqrt(5)
    signed = np.stack((q,np.tanh(ref.window(m,5)/(vol*np.sqrt(5))),
                       np.tanh(ref.window(m,20)/(vol*np.sqrt(20)))),axis=2)
    # The summed state model at 20 and 60 ticks enters as a control for the decomposition.
    ctrl = np.concatenate((p,signed,levels.sum(2)[:,:,1:]),axis=2)
    intervals = np.concatenate((levels[:,:,:,:1],np.diff(levels,axis=3)),axis=3)
    return strip.center(np.concatenate((ctrl,intervals.reshape(*m.shape,9)),axis=2))


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    table = lookup(d)
    p = read_p(d)
    g,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%5,b-a-60,5)
        x = features(raw[a:b],m,ok,p[a:b],table)[t]
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        z = np.concatenate((x,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        g += rows.T@rows
        n += len(rows)
    return g/n


def train(end):
    co = ref.fit(np.stack([record(d) for d in range(4,end)]),C,D)
    return np.r_[lookup(end).ravel(),co.ravel()]


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],m,ok,p[a:b],bank[:S*9].reshape(S,3,3))
        out[a:b] = (x@bank[S*9:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


def audit():
    out = {}
    for end in (8,12):
        _,_,s0,p0 = rewards(range(end))
        _,w,s1,p1 = rewards(range(end,end+4))
        a,b = s0-p0,s1-p1
        def corr(x,y):
            return float((w*x*y).sum()/np.sqrt((w*x*x).sum()*(w*y*y).sum()))
        out[end+1] = dict(recurrent_edge_correlation=corr(a,b),endpoint_edge_correlation=corr(p0,p1),
                          remaining_energy=float((w*b*b).sum()/(w*s1*s1).sum()))
    (OUT/'edge_replication.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
