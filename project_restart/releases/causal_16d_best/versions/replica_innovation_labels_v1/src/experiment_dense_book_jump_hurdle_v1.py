"""Fit separate book-conditioned expectations for jump counts and conditional excess size."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip,official
import prepare_dense_jump_reward_ladder_reference_v1 as ref


base = ref.model
walk = base.walk
links = walk.links
REFERENCE = ref.OUT
OUT = REFERENCE.parent/'dense_book_jump_hurdle_v1'
ARMS = ('directional_frequency','conditional_severity','frequency_times_severity')
BANDS = ((0,5),(5,20),(20,60))
F = 36
C = 5


def read_p(d):
    p = ref.read_p(d)
    if 8<=d<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[d][:,None]
    return p


def design(raw,m,ok,unit,t):
    path = m.cumsum(0)
    cell = np.rint(path/unit).astype(np.int64)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    previous = (np.arange(len(m))-5).clip(0)
    step = cell-cell[previous]
    actual = (path-path[previous])/unit
    good = (gaps[1:]==gaps[previous]) & (np.arange(len(m))>=5)[:,None]
    book = np.tanh(.7*np.arcsinh(raw[:,:,walk.ref.model.FIELDS].astype(float)))
    pressure = book[...,::2]-book[...,1::2]
    def average(x,n):
        pre = np.concatenate((np.zeros_like(x[:1]),x.cumsum(0)))
        return (pre[t+1]-pre[(t+1-n).clip(0)])/np.minimum(n,t+1).reshape((-1,)+(1,)*(x.ndim-1))
    vol = np.log1p(average(abs(actual)*good,60))
    liquidity = (book[t,...,::2]+book[t,...,1::2]).mean(2)
    short = average(pressure,8)-average(pressure,50)
    mid = (path[t]-path[(t-20).clip(0)])/unit
    fields = np.concatenate((pressure[t],short,np.tanh(np.stack((actual[t],mid/3),axis=2))),axis=2)
    for sign in (-1,1):
        hit = (sign*step>2)*good
        excess = np.where(hit,(sign*actual-2).clip(0),0.)
        freq = average(hit.astype(float),80)
        size = average(excess,80)/(freq+.02)
        signed = sign*fields
        state = np.stack((vol,liquidity,3*freq,np.log1p(size)),axis=2)
        # Fit the same rule to mirrored upward and downward events.
        yield np.concatenate((np.ones_like(vol[...,None]),state,signed,
                              signed*np.tanh(vol[...,None]),signed[...,:5]*liquidity[...,None],
                              (np.tanh(signed[...,0])*freq)[...,None],(np.tanh(signed[...,-1])*size)[...,None]),axis=2)


def outcomes(path,cell,t,sign,unit):
    rows = []
    for lo,hi in BANDS:
        count = np.zeros((len(t),50))
        cash = np.zeros_like(count)
        for lag in range(lo,hi,5):
            hits = sign*(cell[t+lag+5]-cell[t+lag])>2
            count += hits
            cash += hits*(sign*(path[t+lag+5]-path[t+lag])/unit-2)
        rows.append((count,cash))
    return rows


@lru_cache(maxsize=16)
def auxiliary(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    unit = walk.widths()
    stats = np.zeros((3,2,F,F+1))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(50+day%6,b-a-60,6)
        path = m.cumsum(0)
        cell = np.rint(path/unit).astype(np.int64)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+61]==gaps[t-5]
        for sign,features in zip((-1,1),design(raw[a:b],m,ok,unit,t)):
            x = features[valid]
            gram = x.T@x
            for j,(count,cash) in enumerate(outcomes(path,cell,t,sign,unit)):
                hit = count[valid]
                stats[j,0,:,:F] += gram
                stats[j,0,:,F] += x.T@hit
                stats[j,1,:,:F] += x.T@(x*hit[:,None])
                stats[j,1,:,F] += x.T@cash[valid]
        n += 2*valid.sum()
    return stats/n


@lru_cache(maxsize=16)
def hurdle(end):
    rows = np.mean([auxiliary(d) for d in range(end)],axis=0)
    weights = np.zeros((3,2,F+1))
    for j in range(3):
        for k in range(2):
            g = rows[j,k,:,:F]
            rhs = rows[j,k,:,F]
            den = g[0,0]
            mu = g[1:,0]/den
            mean = rhs[0]/den
            cov = g[1:,1:]/den-np.outer(mu,mu)
            root = links.base.whitening(cov,.65)
            co = root@(root.T@(rhs[1:]/den-mu*mean))
            weights[j,k] = np.r_[mean-mu@co,co,mean]
    return weights


def forecasts(raw,m,ok,p,t,unit,law):
    v = np.zeros((len(t),50,3,3))
    for sign,x in zip((-1,1),design(raw,m,ok,unit,t)):
        for j,(freq,size) in enumerate(law):
            number = (x@freq[:-1]).clip(0,(BANDS[j][1]-BANDS[j][0])/5)
            magnitude = (x@size[:-1]).clip(0,5*size[-1])
            a,b = number-freq[-1],magnitude-size[-1]
            v[:,:,j] += sign*unit[:,None]*np.stack((a*size[-1],freq[-1]*b,a*b),axis=2)
    price = m.cumsum(0)
    own = np.stack((price[t]-price[(t-5).clip(0)],price[t]-price[(t-20).clip(0)]),axis=2)
    return links.center(np.concatenate((p[t],own,v.reshape(len(t),50,9)),axis=2),ok[t])


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    scale = strip.daily_scale(p,y).clip(.2,4.)
    ratio = strip.physical()[1]
    g = np.zeros((2,C+11,C+11))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(50+d%6,b-a-60,6)
        x = forecasts(raw[a:b],m,ok,p[a:b],t,walk.widths(),hurdle(d))
        e = y[a+t].astype(float)/scale-p[a+t]
        for h in range(2):
            j = h+1
            far = e[:,:,j]-e[:,:,h]*ratio[j]/ratio[h]*scale[h]/scale[j]
            labels = links.center(np.stack((e[:,:,j],far),axis=2),ok[t])
            v = np.concatenate((x,labels),axis=2)[ok[t]]
            g[h] += v.T@v
        n += ok[t].sum()
    return g/n


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    co = np.zeros((2,C+9,3))
    for h in range(2):
        gs = days[:,h]
        g = gs.mean(0)
        for k in range(3):
            control = np.arange(C) if k<2 else np.r_[0:C,C+np.arange(0,9,3),C+np.arange(1,9,3)]
            w = links.base.whitening(g[np.ix_(control,control)],.05)
            v = np.eye(C+11)[:,C+np.arange(k,9,3)].copy()
            v[control] = -w@(w.T@g[control]@v)
            v = v@links.base.whitening(v.T@g@v,.8)
            grad = v.T@gs[:,:,-2:]
            avg = grad.mean(0)
            whole,far = (avg*avg**2/(avg**2+grad.var(0)/len(gs)+1e-9)).T
            co[h,:,k] = (v@(whole*(whole*far>0)))[:-2]
    return np.r_[walk.widths(),hurdle(end).ravel(),co.ravel()]


def signals(raw,grid,p,bank):
    off = 50+6*(F+1)
    law = bank[50:off].reshape(3,2,F+1)
    co = bank[off:].reshape(2,C+9,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = forecasts(raw[a:b],m,ok,p[a:b],np.arange(b-a),bank[:50],law)
        for h in range(2):
            out[a:b,:,:,h] = x@co[h]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
