"""Predict distant returns through the part of a receiver's future book observable in its peers today."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import REFERENCE, read_p, event, onset, rel, strip, graph


OUT = REFERENCE.parent / 'dense_peer_future_book_bridge_v1'
ARMS = ('direct_peer_price', 'future_book_bridge', 'replicated_response_subspace')
D = 18
C = 2*D+3
N = C+2*D+2
BANDS = ((5,20),(20,60))


def bounded(raw):
    return np.tanh(.65*np.arcsinh(raw.astype(np.float32)))


@lru_cache(maxsize=1)
def coordinates():
    gram = np.zeros((100,100))
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            t = np.arange(20+d,b-a,16)
            x = bounded(raw[a+t])-bounded(raw[a+t-10])
            x = x[ok[t]&ok[t-10]].astype(float)
            gram += x.T@x / len(x)
    sd = np.sqrt(gram.diagonal()).clip(.02)
    lam,vec = np.linalg.eigh(gram/np.outer(sd,sd))
    return vec[:,-D:]/sd[:,None]/np.sqrt(lam[-D:]+.3)


def view(raw,ok,p,t,axes):
    level = bounded(raw).astype(float) @ axes
    level *= ok[...,None]
    sums = np.concatenate((np.zeros_like(level[:1]),level.cumsum(0)))
    def mean(lo,hi):
        return (sums[hi]-sums[lo])/(hi-lo).clip(1)[:,None,None]
    fast = mean((t-4).clip(0),t+1)
    slow = mean((t-34).clip(0),(t-4).clip(0))
    delta = fast-slow
    valid = ok[t] & (t>=34)[:,None]
    def center(x):
        mask = valid[...,None]
        return (x-(x*mask).sum(1,keepdims=True)/mask.sum(1,keepdims=True).clip(1))*mask
    ctrl = center(np.concatenate((fast,delta,p[t]),axis=2))
    return ctrl,center(delta),fast,mean


def message(delta,w,valid):
    return (delta.transpose(0,2,1) @ w).transpose(0,2,1)*valid[...,None]


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    coef = strip.physical()[1]
    gram = np.zeros((2,N,N))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%6,b-a-60,6)
        ctrl,delta,fast,mean = view(raw[a:b],ok,p[a:b],t,coordinates())
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+61]==gaps[t-34]
        for h,(lo,hi) in enumerate(BANDS):
            j = h+1
            peer = message(delta,graph.filters(d)[h,0],ok[t])
            future = mean(t+lo+1,t+hi+1)-fast
            yj = y[a+t,:,j].astype(float)
            err = yj/beta[j]-p[a+t,:,j]
            far = (yj-y[a+t,:,j-1]*coef[j]/coef[j-1])/beta[j]
            obs = np.concatenate((ctrl,peer,future,err[...,None],far[...,None]),axis=2)
            obs = (obs-(obs*valid[...,None]).sum(1,keepdims=True)/valid.sum(1).clip(1)[:,None,None])[valid]
            gram[h] += obs.T@obs
        n += valid.sum()
    return gram/n


def whitening(g, penalty):
    sd = np.sqrt(g.diagonal().clip(1e-7))
    val,u = np.linalg.eigh(g/np.outer(sd,sd))
    return (u/np.sqrt(val.clip(0)+penalty))/sd[:,None]


def readout(gs):
    g = gs.mean(0)
    source = np.eye(N)[:,C:C+D].copy()
    future = np.eye(N)[:,C+D:C+2*D].copy()
    ctrl = whitening(g[:C,:C],.15)
    for v in (source,future):
        v[:C] = -ctrl@(ctrl.T@g[:C]@v)
    source = source@whitening(source.T@g@source,.8)
    future = future@whitening(future.T@g@future,.7)
    avg = gs[:,:,-2].mean(0)@source
    far = gs[:,:,-1].mean(0)@source
    direct = avg*(avg*far>0)
    transfers = []
    for part in (gs[::2],gs[1::2]):
        m = part.mean(0)
        transfers.append((source.T@m@future,future.T@m[:,-2:],source.T@m[:,-2:]))
    bridges = []
    restricted = []
    for fit,test in (transfers,transfers[::-1]):
        link = fit[0]
        bridge = link@test[1]
        bridges.append(bridge[:,0]*(bridge[:,0]*bridge[:,1]>0))
        u,s,vt = np.linalg.svd(link,full_matrices=False)
        repeated = np.diag(u.T@test[0]@vt.T).clip(0)
        retained = repeated/(s+.025)
        reward = u.T@test[2]
        restricted.append(u@(retained.clip(0,1)*reward[:,0]*(reward[:,0]*reward[:,1]>0)))
    co = np.stack((direct,np.mean(bridges,0),np.mean(restricted,0)),axis=1)
    return (source@co)[:C+D]


def train(end):
    gs = np.stack([record(d) for d in range(4,end)])
    co = np.stack([readout(gs[:,h]) for h in range(2)])
    return np.r_[coordinates().ravel(),graph.filters(end)[:,0].ravel(),co.ravel()]


def signals(raw,grid,p,bank):
    axes = bank[:100*D].reshape(100,D)
    w = bank[100*D:100*D+5000].reshape(2,50,50)
    co = bank[100*D+5000:].reshape(2,C+D,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        ctrl,delta,_,_ = view(raw[a:b],ok,p[a:b],np.arange(b-a),axes)
        for h in range(2):
            x = np.concatenate((ctrl,message(delta,w[h],ok)),axis=2)
            out[a:b,:,:,h] = x@co[h]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
