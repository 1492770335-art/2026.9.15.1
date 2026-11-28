"""Roll forward an intraday price-level transition model conditional on arrival direction."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip,official
import prepare_dense_revisited_pressure_reference_v1 as ref


links = ref.model.links
REFERENCE = ref.OUT
OUT = REFERENCE.parent/'dense_price_ladder_walk_v1'
ARMS = ('price_level','level_given_arrival','level_direction_interaction')
STEP = np.arange(-2,3)
SIGN = np.sign(STEP)+1
C,D = 13,6


def read_p(day):
    p = ref.read_p(day)
    if 8<=day<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[day][:,None]
    return p


@lru_cache(maxsize=1)
def widths():
    sizes = []
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            t = np.arange(5,b-a,5)
            path = m.cumsum(0)
            gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
            use = gaps[t+1]==gaps[t-5]
            sizes.append((abs(path[t]-path[t-5])*use).sum(0)/use.sum(0).clip(1))
    return np.median(sizes,axis=0).clip(.025)


def probabilities(counts,prior,weight):
    return (counts+weight*prior)/(counts.sum(-1,keepdims=True)+weight)


def lookup(keys,current):
    i = np.searchsorted(keys,current).clip(max=len(keys)-1)
    return i,keys[i]==current


def transitions(cell,gaps,start):
    end = np.arange(max(10,start-720),start,5)
    t = end-5
    valid = gaps[end+1]==gaps[t-5]
    stock = np.broadcast_to(np.arange(50),valid.shape)[valid]
    before = cell[t][valid]
    change = (cell[end][valid]-before).clip(-2,2)
    incoming = np.sign(before-cell[t-5][valid])+1
    origins = 50*before+stock
    keys = np.unique((origins[:,None]+STEP*50).ravel())
    if not len(keys):
        return None
    ix = np.searchsorted(keys,origins)
    counts = np.bincount((ix*3+incoming)*5+change+2,minlength=len(keys)*15).reshape(-1,3,5)
    by_stock = np.zeros((50,3,5))
    np.add.at(by_stock,keys%50,counts)
    # Smooth each name's step probabilities towards the session-wide direction prior.
    prior = probabilities(by_stock.sum(0),np.full((3,5),.2),5)
    momentum = probabilities(by_stock,prior,18)
    plain = probabilities(by_stock.sum(1),np.full((50,5),.2),5)
    levels = probabilities(counts.sum(1),plain[keys%50],12)
    joint = probabilities(counts,momentum[keys%50],12)
    return keys,plain,momentum,levels,joint


def forecast(table,current,incoming):
    keys,plain,momentum,levels,joint = table
    idx,known = lookup(keys,current)
    neighbor,inside = lookup(keys,keys[:,None]+50*STEP)
    stock = keys%50
    name = current%50
    local = np.zeros(len(keys))
    directional = np.zeros((len(keys),3))
    background = np.zeros((50,3))
    drift = plain@STEP
    out = []
    for k in range(1,13):
        value = np.where(inside,local[neighbor],(k-1)*drift[stock,None])
        local = (levels*(STEP+value)).sum(1)
        value = np.where(inside,directional[neighbor,SIGN],background[stock[:,None],SIGN])
        directional = (joint*(STEP+value[:,None,:])).sum(2)
        background = (momentum*(STEP+background[:,SIGN])[:,None,:]).sum(2)
        if k in (1,4,12):
            price_effect = np.where(known,local[idx]-k*drift[name],0.)
            conditional = np.where(known,directional[idx,incoming]-background[name,incoming],0.)
            out.append(np.stack((price_effect,conditional,conditional-price_effect),axis=2))
    horizons = np.stack(out,axis=3)
    return np.diff(horizons,axis=3,prepend=np.zeros_like(horizons[...,:1]))


def states(raw,m,ok,p,t,unit):
    b = np.tanh(.7*np.arcsinh(raw[:,:,ref.model.FIELDS].astype(float)))
    pressure = b[...,::2]-b[...,1::2]
    sums = np.concatenate((np.zeros_like(pressure[:1]),pressure.cumsum(0)))
    sm = (sums[t+1]-sums[(t-31).clip(0)])/np.minimum(t+1,32)[:,None,None]
    ctrl = links.center(np.concatenate((p[t],pressure[t],sm),axis=2),ok[t])
    cell = np.rint(m.cumsum(0)/unit).astype(np.int64)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    folds = np.zeros((len(t),50,3,3))
    for lo in range(80,len(m),80):
        loc = np.flatnonzero((t>=lo)&(t<lo+80))
        if not len(loc):
            continue
        table = transitions(cell,gaps,lo)
        if table is not None:
            tick = t[loc]
            inc = np.sign(cell[tick]-cell[tick-5])+1
            value = forecast(table,50*cell[tick]+np.arange(50),inc)
            folds[loc] = value*(gaps[tick+1]==gaps[tick-5])[...,None,None]*unit[:,None,None]
    tilt = pressure[t].mean(2)
    mixed = np.concatenate((folds,abs(folds)*np.tanh(tilt[:,:,None,None])),axis=3)
    return ctrl,[links.center(mixed[:,:,k],ok[t]) for k in range(3)]


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    ratio = strip.physical()[1]
    grams = np.zeros((2,3,C+D+2,C+D+2))
    count = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(80+d%5,b-a-60,5)
        ctrl,parts = states(raw[a:b],m,ok,p[a:b],t,widths())
        err = y[a+t].astype(float)/beta-p[a+t]
        for h in range(2):
            j = h+1
            far = err[:,:,j]-err[:,:,h]*(ratio[j]/ratio[h])*(beta[h]/beta[j])
            labels = links.center(np.stack((err[:,:,j],far),axis=2),ok[t])
            for k,z in enumerate(parts):
                v = np.concatenate((ctrl,z,labels),axis=2)[ok[t]]
                grams[h,k] += v.T@v
        count += ok[t].sum()
    return grams/count


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    co = np.zeros((2,3,C+D))
    for h in range(2):
        for k in range(3):
            rows = days[:,h,k]
            g = rows.mean(0)
            w = links.base.whitening(g[:C,:C],.03)
            v = np.eye(C+D+2)[:,C:C+D].copy()
            v[:C] = -w@(w.T@g[:C]@v)
            v = v@links.base.whitening(v.T@g@v,.8)
            grad = v.T@rows[:,:,-2:]
            mu = grad.mean(0)
            total,far = (mu*mu**2/(mu**2+grad.var(0)/len(rows)+1e-9)).T
            co[h,k] = (v@(total*(total*far>0)))[:-2]
    return np.r_[widths(),co.ravel()]


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    co = bank[50:].reshape(2,3,C+D)
    for a,b,m,ok in event.sessions(raw,grid):
        ctrl,parts = states(raw[a:b],m,ok,p[a:b],np.arange(b-a),bank[:50])
        for k,z in enumerate(parts):
            out[a:b,:,k] = np.concatenate((ctrl,z),axis=2)@co[:,k].T
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
