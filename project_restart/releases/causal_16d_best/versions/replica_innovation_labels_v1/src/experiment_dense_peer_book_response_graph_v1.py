"""Estimate a cross-stock graph from reproducible changes in receivers' books before distant returns."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import REFERENCE, read_p, event, onset, rel, strip
import experiment_dense_peer_future_book_bridge_v1 as base


OUT = REFERENCE.parent / 'dense_peer_book_response_graph_v1'
ARMS = ('book_response_graph','above_relabeled_graph','date_stable_graph_price')
D = 8
C = 2*D+3
N = C+2*D+2
PERM = np.random.default_rng(7314).permutation(50)


def raw_view(raw,ok,t,axes):
    level = base.bounded(raw)@axes
    level *= ok[...,None]
    sums = np.concatenate((np.zeros_like(level[:1]),level.cumsum(0)))
    def mean(lo,hi):
        return (sums[hi]-sums[lo])/(hi-lo).clip(1)[:,None,None]
    fast = mean((t-4).clip(0),t+1)
    delta = fast-mean((t-34).clip(0),(t-4).clip(0))
    return fast,delta,mean


def center(x,valid):
    mask = valid[...,None]
    return (x-(x*mask).sum(1,keepdims=True)/mask.sum(1,keepdims=True).clip(1))*mask


@lru_cache(maxsize=16)
def relation(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    total = np.zeros((2,50,D,50,D))
    count = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%6,b-a-20,6)
        fast,delta,mean = raw_view(raw[a:b],ok,t,base.coordinates()[:,-D:])
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+21]==gaps[t-34]
        x = center(delta,valid)
        for h,near in enumerate((5,20)):
            later = center(mean(t+near-4,t+near+1)-fast,valid)
            own = center(np.concatenate((fast,delta,(path[t+near]-path[t])[...,None]),axis=2),valid)
            c = own[valid]
            y = later[valid]
            projection = base.whitening(c.T@c/len(c),.01)
            coef = projection@(projection.T@(c.T@y/len(c)))
            y = later-own@coef
            g = x.reshape(len(t),-1).T@y.reshape(len(t),-1)
            total[h] += g.reshape(50,D,50,D)
        count += len(t)
    tensor = (total/count).transpose(0,1,3,2,4)
    ids = np.arange(50)
    tensor[:,ids,ids] = 0
    mask = (1-np.eye(50))[...,None,None]
    tensor -= mask*tensor.sum(1,keepdims=True)/49
    return tensor.reshape(2,2500,D*D)


@lru_cache(maxsize=16)
def graphs(end):
    days = np.stack([relation(d) for d in range(end)])
    mats = []
    for h in range(2):
        left,right = [days[k::2,h].mean(0) for k in range(2)]
        cross = .5*(left.T@right + right.T@left)
        val,vec = np.linalg.eigh(cross)
        u = vec[:,-6:]
        noise = np.sum(((left-right)@u)**2,axis=0)/4
        strength = val[-6:].clip(0)/(val[-6:].clip(0)+noise+1e-9)
        weights = (((left+right)*.5)@u*strength)@u.T
        mats.append(weights.reshape(50,50,D,D).transpose(0,2,1,3).reshape(50*D,50*D))
    return np.stack(mats)


def features(raw,ok,p,t,axes,w):
    fast,delta,_ = raw_view(raw,ok,t,axes)
    valid = ok[t] & (t>=34)[:,None]
    x = center(delta,valid)
    ctrl = center(np.concatenate((fast,delta,p[t]),axis=2),valid)
    for matrix in w:
        permuted = matrix.reshape(50,D,50,D)[PERM][:,:,PERM].reshape(50*D,50*D)
        parts = [ctrl]
        for m in (matrix,permuted):
            v = (x.reshape(len(t),-1)@m).reshape(len(t),50,D)
            parts.append(center(v,valid))
        yield np.concatenate(parts,axis=2)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    coef = strip.physical()[1]
    grams = np.zeros((2,N,N))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%6,b-a-60,6)
        for h,x in enumerate(features(raw[a:b],ok,p[a:b],t,base.coordinates()[:,-D:],graphs(d))):
            j = h+1
            yy = y[a+t,:,j].astype(float)/beta[j]
            tail = yy-y[a+t,:,j-1]*coef[j]/(beta[j]*coef[j-1])
            target = center(np.stack((yy-p[a+t,:,j],tail),axis=2),ok[t])
            z = np.concatenate((x,target),axis=2)[ok[t]]
            grams[h] += z.T@z
        n += ok[t].sum()
    return grams/n


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    co = np.zeros((2,N-2,3))
    for h in range(2):
        gs = days[:,h]
        g = gs.mean(0)
        for arm in range(3):
            ctrl = np.arange(C) if arm==0 else np.r_[0:C,C+D:C+2*D]
            w = base.whitening(g[np.ix_(ctrl,ctrl)],.10)
            v = np.eye(N)[:,C:C+D].copy()
            v[ctrl] = -w@(w.T@g[ctrl]@v)
            v = v@base.whitening(v.T@g@v,.9)
            gradients = v.T@gs[:,:,-2:]
            avg = gradients.mean(0)
            trust = avg**2/(avg**2+gradients.var(0)/len(gs)+1e-8)
            price,far = (avg*trust).T
            price = price*(price*far>0)
            if arm==2:
                a,b = [gradients[k::2,:,0].mean(0) for k in range(2)]
                price *= (a*b>0)*np.minimum(abs(a),abs(b))/(abs(avg[:,0])+1e-8)
            co[h,:,arm] = (v@price)[:-2]
    return np.r_[base.coordinates()[:,-D:].ravel(),graphs(end).ravel(),co.ravel()]


def signals(raw,grid,p,bank):
    axes = bank[:100*D].reshape(100,D)
    off = 100*D+2*(50*D)**2
    w = bank[100*D:off].reshape(2,50*D,50*D)
    co = bank[off:].reshape(2,N-2,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        for h,x in enumerate(features(raw[a:b],ok,p[a:b],np.arange(b-a),axes,w)):
            out[a:b,:,:,h] = x@co[h]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
