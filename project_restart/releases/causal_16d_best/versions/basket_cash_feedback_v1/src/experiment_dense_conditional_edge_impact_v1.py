"""Price the same predicted book transition separately by stock and liquidity at forecast issue time."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_future_edge_density_v1 as direct

base,strip,event,onset,rel = direct.base,direct.strip,direct.event,direct.onset,direct.rel
REFERENCE,read_p = direct.REFERENCE,direct.read_p
OUT = REFERENCE.parent/'dense_conditional_edge_impact_v1'
ARMS = ('stock_impact','liquidity_impact','stock_liquidity_interaction')
C,D = 9,18
TABLE = (3,50,4,3,25)


def conditions(raw,m):
    active = direct.edge.cycle.ref.window((abs(m)>.001).astype(float),20)
    return np.stack((raw[:,:,39],active),axis=2)


@lru_cache(maxsize=1)
def thresholds():
    samples = []
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            x = conditions(raw[a:b],m)[20::6]
            samples.append(np.where(ok[20::6,:,None],x,np.nan))
    return np.nanmedian(np.concatenate(samples),axis=0)


@lru_cache(maxsize=16)
def moments(day):
    gram = np.zeros((2,50,4,3,25))
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    for a,b,m,ok in event.sessions(raw,grid):
        _,code,v = direct.fields(raw[a:b],m)
        val = conditions(raw[a:b],m)>thresholds()
        regime = 2*val[:,:,0]+val[:,:,1]
        t = np.arange(40+day%5,b-a-60,5)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        keep = gap[t+61]==gap[t]
        path = m.cumsum(0)
        prefix = (np.arange(50)+np.zeros_like(code[t]))*4+regime[t]
        for j,shifts in enumerate(direct.BANDS):
            for h in shifts:
                s = t+h
                r = path[s+5]-path[s]
                r -= (r*keep).sum(1,keepdims=True)/keep.sum(1,keepdims=True).clip(1)
                idx = (25*prefix+5*code[s]+code[s+5])[keep]
                for k,w in enumerate((None,(r/v[t])[keep])):
                    batch = np.bincount(idx,weights=w,minlength=50*4*25).reshape(50,4,25)
                    gram[k,:,:,j] += batch/len(shifts)
    return gram


def tables(end):
    halves = []
    for split in (0,1):
        z = sum(moments(d) for d in range(split,end,2))
        n,r = z[0]+z[0,...,::-1],z[1]-z[1,...,::-1]
        avg = r.sum((0,1),keepdims=True)/(n.sum((0,1),keepdims=True)+1.)
        def adjust(axis,prior):
            error = r-n*avg
            return error.sum(axis,keepdims=True)/(n.sum(axis,keepdims=True)+prior)
        stock = adjust(1,100.)
        regime = adjust(0,350.)
        combined = (r-n*(avg+stock+regime))/(n+120.)
        halves.append(np.stack(np.broadcast_arrays(stock,regime,combined)))
    a,b = halves
    return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)


def features(raw,m,p,net,table,cuts):
    f,code,vol = direct.fields(raw,m)
    c = conditions(raw,m)>cuts
    regime = 2*c[:,:,0]+c[:,:,1]
    out = np.zeros((*m.shape,D))
    out[:,:,:3] = p
    out[:,:,3:5] = f[:,:,-4:-2]
    out[:,:,5] = p[...,2]*c[:,:,0]
    out[:,:,6] = p[...,2]*c[:,:,1]
    out[:,:,7] = p[...,1]*c[:,:,0]
    out[:,:,8] = p[...,1]*c[:,:,1]
    for start in range(0,len(m),64):
        sl = slice(start,min(start+64,len(m)))
        prob,_ = direct.density(f[sl],code[sl],net)
        for j,band in enumerate(direct.BANDS):
            for k in range(3):
                value = table[k,np.arange(50)[None,:],regime[sl],j]
                out[sl,:,C+3*k+j] = (value*prob[:,:,j]).sum(-1)*vol[sl]*len(band)
    return strip.center(out)


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    tab = tables(day)
    g,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%5,b-a-60,5)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        x = features(raw[a:b],m,p[a:b],direct.encoder(),tab,thresholds())[t]
        z = np.concatenate((x,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        g += rows.T@rows
        n += len(rows)
    return g/n


def train(end):
    co = direct.edge.cycle.ref.fit(np.stack([record(d) for d in range(4,end)]),C,D)
    return np.r_[thresholds().ravel(),tables(end).ravel(),direct.encoder(),co.ravel()]


def signals(raw,grid,p,bank):
    cut = 100+int(np.prod(TABLE))
    table = bank[100:cut].reshape(TABLE)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        f = features(raw[a:b],m,p[a:b],bank[cut:-D*6],table,bank[:100].reshape(50,2))
        out[a:b] = (f@bank[-D*6:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
