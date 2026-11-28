"""Learn the magnitude and destination of observed price jumps separately from ordinary ladder moves."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip,official
import prepare_dense_price_ladder_reference_v1 as ref


walk = ref.model
REFERENCE = ref.OUT
OUT = REFERENCE.parent/'dense_jump_reward_ladder_v1'
ARMS = ('ordinary_move','excess_jump','separate_jump_price')
STEP = np.array([-5,-3,-2,-1,0,1,2,3,5])
SIGN = np.sign(STEP)+1
C = 6


def read_p(d):
    p = ref.read_p(d)
    if 8<=d<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[d][:,None]
    return p


def predict_block(cell,path,unit,gaps,lo,t):
    end = np.arange(max(10,lo-720),lo,5)
    s = end-5
    keep = gaps[end+1]==gaps[s-5]
    name = np.broadcast_to(np.arange(50),keep.shape)[keep]
    if not len(name):
        return np.zeros((len(t),50,3,2))
    old = cell[s][keep]
    new = cell[end][keep]
    step = new-old
    inc = np.sign(old-cell[s-5][keep])+1
    dest_dir = np.sign(step)+1
    group = step.clip(-4,4)+4
    real = (path[end]-path[s])[keep]/unit[name]
    # The sum of both rewards equals the physical, unrounded movement.
    ordinary = np.where(abs(step)<=2,real,step.clip(-2,2))
    rewards = np.column_stack((ordinary,real-ordinary))
    origins = 50*old+name
    destinations = 50*new+name
    keys = np.unique(np.r_[origins,destinations])
    num = len(keys)
    node = 3*np.searchsorted(keys,origins)+inc
    dest = 3*np.searchsorted(keys,destinations)+dest_dir
    stock = keys%50
    local_count = np.bincount(node,minlength=3*num).reshape(num,3)
    observed = np.zeros((num*3,2))
    np.add.at(observed,node,rewards)
    counts = np.zeros((50,3,9))
    np.add.at(counts,(name,inc,group),1.)
    base_count = counts.sum(2)
    mean_reward = np.zeros((50,3,2))
    np.add.at(mean_reward,(name,inc),rewards)
    shared = mean_reward.sum(0)/(base_count.sum(0)[:,None]+1)
    mean_reward = (mean_reward+20*shared)/(base_count[...,None]+20)
    prior = walk.probabilities(counts.sum(0),np.full((3,9),1/9),8)
    prob = walk.probabilities(counts,prior,20)
    idx,known = walk.lookup(keys,50*cell[t]+np.arange(50))
    incoming = np.sign(cell[t]-cell[t-5])+1
    neighbor,inside = walk.lookup(keys,keys[:,None]+50*STEP)
    work = np.zeros((num,3,2))
    background = np.zeros((50,3,2))
    horizons = []
    for k in range(1,13):
        moved = np.where(inside[...,None],work[neighbor,SIGN],background[stock[:,None],SIGN])
        fallback = mean_reward[stock] + np.einsum('nsg,ngr->nsr',prob[stock],moved)
        continuation = work.reshape(-1,2)[dest]
        future = np.column_stack([np.bincount(node,weights=col,minlength=3*num) for col in continuation.T])
        work = ((observed+future).reshape(num,3,2)+20*fallback)/(local_count[...,None]+20)
        background = mean_reward+np.einsum('nsg,ngr->nsr',prob,background[:,SIGN])
        if k in (1,4,12):
            value = work[idx,incoming]-background[np.arange(50),incoming]
            horizons.append(value*known[...,None]*unit[:,None])
    return np.diff(np.stack(horizons,axis=2),axis=2,prepend=np.zeros_like(horizons[0][:,:,None]))


def states(raw,m,ok,p,t,unit):
    path = m.cumsum(0)
    cell = np.rint(path/unit).astype(np.int64)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    x = np.zeros((len(t),50,3,2))
    for lo in range(80,len(m),80):
        loc = np.flatnonzero((t>=lo)&(t<lo+80))
        if len(loc):
            ix = t[loc]
            valid = gaps[ix+1]==gaps[ix-5]
            x[loc] = predict_block(cell,path,unit,gaps,lo,ix)*valid[...,None,None]
    pressure = np.tanh(.65*np.arcsinh(raw[t,:,41].astype(float)))-np.tanh(.65*np.arcsinh(raw[t,:,42].astype(float)))
    controls = np.stack((path[t]-path[(t-20).clip(0)],path[t]-path[(t-60).clip(0)],pressure),axis=2)
    return walk.links.center(np.concatenate((p[t],controls,x.reshape(len(t),50,6)),axis=2),ok[t])


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    units = strip.physical()[1]
    g = np.zeros((2,C+8,C+8))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(80+d%5,b-a-60,5)
        x = states(raw[a:b],m,ok,p[a:b],t,walk.widths())
        e = y[a+t].astype(float)/beta-p[a+t]
        for h in range(2):
            j = h+1
            far = e[:,:,j]-e[:,:,h]*units[j]/units[h]*beta[h]/beta[j]
            labels = walk.links.center(np.stack((e[:,:,j],far),axis=2),ok[t])
            v = np.concatenate((x,labels),axis=2)[ok[t]]
            g[h] += v.T@v
        n += ok[t].sum()
    return g/n


def train(end):
    rows = np.stack([record(d) for d in range(4,end)])
    co = np.zeros((2,C+6,3))
    for h in range(2):
        gs = rows[:,h]
        g = gs.mean(0)
        ctrl = walk.links.base.whitening(g[:C,:C],.03)
        for k,indices in enumerate((range(0,6,2),range(1,6,2),range(6))):
            v = np.eye(C+8)[:,C+np.array(indices)].copy()
            v[:C] = -ctrl@(ctrl.T@g[:C]@v)
            v = v@walk.links.base.whitening(v.T@g@v,.8)
            grad = v.T@gs[:,:,-2:]
            mean = grad.mean(0)
            whole,far = (mean*mean**2/(mean**2+grad.var(0)/len(gs)+1e-10)).T
            co[h,:,k] = (v@(whole*(whole*far>0)))[:-2]
    return np.r_[walk.widths(),co.ravel()]


def signals(raw,grid,p,bank):
    co = bank[50:].reshape(2,C+6,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = states(raw[a:b],m,ok,p[a:b],np.arange(b-a),bank[:50])
        for h in range(2):
            out[a:b,:,:,h] = x@co[h]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
