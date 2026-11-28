"""Estimate direction-specific persistence of settled market surprises across disjoint panels."""
from functools import lru_cache
import sys

import numpy as np
import torch

import prepare_dense_disjoint_settled_error_reference_v1 as ref
from experiment_dense_stock_surprise_response_v1 import BLOCK, event, onset, rel, strip, unit, official

REFERENCE = ref.OUT
OUT = REFERENCE.parent / 'dense_directional_market_persistence_v1'
ARMS = ('signed_persistence', 'surprise_confirmed_by_forecast', 'market_breadth_persistence')
GROUP = np.random.default_rng(289137).permutation(50).reshape(2,25)


def read_p(day):
    p = ref.read_p(day)
    if 8<=day<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[day][:,None]
    return p


def observe(move,ok,p,scale):
    path = move.cumsum(0)
    missing = (~ok).cumsum(0)
    acc = np.zeros((3,2,2))
    for lo in range(BLOCK,len(p),BLOCK):
        t = np.arange(lo-BLOCK+5,lo,5)
        valid = (missing[t]==missing[t-5]) & ok[t-5]
        issued = p[t-5,:,0].astype(float)*scale
        surprise = (path[t]-path[t-5]-issued)*valid
        def panel(z):
            return z[:,GROUP].sum(2)/valid[:,GROUP].sum(2).clip(1)
        q = panel(surprise)
        f = panel(issued*valid)
        breadth = panel(np.sign(surprise))
        nxt = q-q.mean(0)
        mom = np.zeros_like(acc)
        for lag in range(1,7):
            prior, later = nxt[:-lag], nxt[lag:,::-1]
            signs = (prior>=0, (prior*f[:-lag])>=0, (prior*breadth[:-lag])>=0)
            w = (1-lag/7)*(1-lag/12)
            for j,mask in enumerate(signs):
                for k,side in enumerate((mask,~mask)):
                    mom[j,k] += w*np.array([(prior*later*side).mean(),(prior**2*side).mean()])
        acc = .77*acc + mom
        if lo>=2*BLOCK:
            slope = acc[:,:,0]/(acc[:,:,1]+.3*acc[:,:,1].sum(1,keepdims=True)+1e-6)
            yield lo,min(lo+BLOCK,len(p)), np.tanh(3*(slope[:,0]-slope[:,1]))


def design(move,ok,p,scale):
    q = p.mean(1,dtype=float)
    x = np.zeros((len(p),3,7))
    x[:,:,0] = 1.
    x[:,:,1:4] = q[:,None,:]
    path = move.cumsum(0)
    gap = (~ok).cumsum(0)
    for lo,hi,z in observe(move,ok,p,scale):
        t = np.arange(lo,hi)
        valid = (gap[t]==gap[t-5]) & ok[t-5]
        err = (path[t]-path[t-5]-p[t-5,:,0]*scale)*valid
        recent = err.sum(1)/valid.sum(1).clip(1)
        # An asymmetric slope changes amplitude according to the side of the
        # current forecast, or according to new short-term evidence and breadth.
        x[lo:hi,0,4:] = z[0]*abs(q[t])
        x[lo:hi,1,4:] = z[1]*q[t]*np.sign(recent*q[t,0])[:,None]
        x[lo:hi,2,4:] = z[2]*q[t]*np.sign(recent*np.sign(err).sum(1))[:,None]
    return x


@lru_cache(maxsize=48)
def record(day,scale):
    p = read_p(day)
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    x = np.concatenate([design(m,ok,p[a:b],scale) for a,b,m,ok in event.sessions(raw,grid)])
    y = y.astype(float)-y.mean((0,1),dtype=float)
    yc = y.mean(1)
    pf = p.astype(float)-p.mean((0,1),dtype=float)
    slope = ((pf*y).sum((0,1))/(pf*pf).sum((0,1))).clip(.2,4)
    err = (yc/slope-pf.mean(1))[:,1:]
    return np.einsum('tji,tjk->jik',x,x)/len(x),np.einsum('tji,th->jih',x,err)/len(x)


def train(end):
    scale = float(np.median([unit(d) for d in range(4,end)]))
    grams,cross = map(np.stack,zip(*(record(d,scale) for d in range(4,end))))
    weights=[]
    for j in range(3):
        g=grams[:,j].mean(0)
        v=np.eye(7)[:,4:]
        v[:4]=-np.linalg.solve(g[:4,:4]+.01*np.diag(g.diagonal()[:4].clip(1e-6)),g[:4,4:])
        gram=v.T@g@v
        gradients=v.T@cross[:,j]
        avg=gradients.mean(0)
        shrink=avg**2/(avg**2+2*gradients.var(0)/len(gradients)+1e-12)
        weights.append(v@np.linalg.solve(gram+1.5*np.diag(gram.diagonal().clip(1e-5)),avg*shrink))
    return np.r_[scale,np.stack(weights).ravel()]


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x=design(m,ok,p[a:b],bank[0])
        out[a:b]=np.einsum('tji,jih->tjh',x,bank[1:].reshape(3,7,2))[:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
