"""Learn receiver-specific changes in first-passage odds from identities and arrival order of other stocks."""
import sys
from functools import lru_cache, partial

import numpy as np
from scipy.ndimage import maximum_filter1d
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_settled_visit_memory_v1 as memory

base,ref,REFERENCE,read_p = memory.base,memory.ref,memory.REFERENCE,memory.read_p
OUT = REFERENCE.parent/'dense_peer_crossing_clock_v1'
ARMS = ('identity_of_crossed_peers','early_versus_late_peers','both_identity_and_order')
LAGS = (5,20,60)
NAMES = np.arange(50)
BINS = 6*3
WEIGHT_SHAPE = (2,50,50,3)


def window(path,h):
    # The origin is the last element for both odd and even window lengths.
    return maximum_filter1d(path,size=h,origin=(h-1)//2,axis=0,mode='nearest')


def observations(m,valid,t,unit,with_targets):
    path = np.cumsum(m,axis=0)/unit
    gaps = np.vstack((np.zeros(50,int),(~valid).cumsum(0)))
    start = (t-30).clip(0)
    half = (t-15).clip(0)
    ok = (t>=30)[:,None] & (gaps[t+1]==gaps[start])
    for sign in (-1,1):
        price = sign*path
        boundary = price[start]+2.
        # Both (inclusive) 15-observation windows together cover start+1 through t.
        high = window(price,15)
        early = high[half]>=boundary
        arrived = early | (high[t]>=boundary)
        risk = ok & ~arrived
        distance = boundary-price[t]
        breadth = (arrived*ok).sum(1)/ok.sum(1).clip(1)
        key = np.digitize(distance,[.5,1.25,2.25,3.5,5.5])*3+np.digitize(breadth,[.22,.48])[:,None]
        identity = base.links.center(arrived[...,None].astype(float),ok)[...,0]
        # Conditional on breadth and arrival status, which peers moved first?
        rate = (early*ok).sum(1)/(arrived*ok).sum(1).clip(1)
        timing = (early-arrived*rate[:,None])*ok
        if with_targets:
            # Targets use future observations only when fitting the auxiliary task.
            outcomes = [window(price,h)[t+h]>=boundary for h in LAGS]
            seen = [risk & (gaps[t+h+1]==gaps[t+1]) for h in LAGS]
            yield sign,key,(identity,timing),risk,np.stack(outcomes,axis=2),np.stack(seen,axis=2)
        else:
            yield sign,key,(identity,timing),risk,None,None


@lru_cache(maxsize=16)
def examples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    rows = []
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(30+day%8,b-a-60,8)
        rows.extend(observations(m,ok,t,base.risk.walk.widths(),True))
    return rows


@lru_cache(maxsize=16)
def counts(day):
    table = np.zeros((BINS,50,3,2))
    for _,key,_,_,y,seen in examples(day):
        for h in range(3):
            np.add.at(table[:,:,h,0],(key,NAMES),seen[...,h])
            np.add.at(table[:,:,h,1],(key,NAMES),seen[...,h]&y[...,h])
    return table


def probs(table):
    total,hit = table[...,0],table[...,1]
    prior = (hit.sum(1,keepdims=True)+1)/(total.sum(1,keepdims=True)+2)
    return (hit+32*prior)/(total+32)


@lru_cache(maxsize=16)
def edges(end):
    days = np.stack([counts(d) for d in range(end)])
    full = days.sum(0)
    co = []
    for d,table in enumerate(days):
        prob = probs(full-table)
        daily = np.zeros(WEIGHT_SHAPE)
        exposure = np.zeros(WEIGHT_SHAPE)
        for _,key,donors,_,y,seen in examples(d):
            err = (y-prob[key,NAMES])*seen
            for k,x in enumerate(donors):
                for h in range(3):
                    daily[k,:,:,h] += x.T@err[...,h]
                    exposure[k,:,:,h] += (x*x).T@seen[...,h].astype(float)
        co.append(daily/(20+exposure))
    co = np.stack(co)
    mu = co.mean(0)
    stable = mu**2/(mu**2+2*co.var(0)/end+1e-7)
    co = mu*stable
    co[:,NAMES,NAMES] = 0.
    # Normalize by the effective number of independent peer contributions at inference.
    return co/np.sqrt(49)


def states(raw,m,ok,p,t,unit,graph):
    path = m.cumsum(0)
    move = np.stack([path[t]-path[(t-h).clip(0)] for h in LAGS],axis=2)
    levels = np.tanh(.7*np.arcsinh(raw[t][:,:,base.risk.walk.ref.model.FIELDS[:4]].astype(float)))
    book = levels[...,::2]-levels[...,1::2]
    ctrl = base.links.center(np.concatenate((p[t],move,book),axis=2),ok[t])
    z = np.zeros((2,len(t),50,3))
    for sign,_,donors,risk,_,_ in observations(m,ok,t,unit,False):
        for k,x in enumerate(donors):
            z[k] += sign*risk[...,None]*(x@graph[k].reshape(50,-1)).reshape(len(t),50,3)
    return ctrl,[base.links.center(x,ok[t]) for x in (z[0],z[1],z.sum(0))]


@lru_cache(maxsize=12)
def record(day):
    designer = partial(states,unit=base.risk.walk.widths(),graph=edges(day))
    return base.record(day,designer,read_p)


def train(end):
    return np.r_[base.risk.walk.widths(),edges(end).ravel(),base.train(end,record).ravel()]


def signals(raw,grid,p,bank):
    designer = partial(states,unit=bank[:50],graph=bank[50:-66].reshape(WEIGHT_SHAPE))
    return base.signals(raw,grid,p,bank[-66:].reshape(2,3,11),designer)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
