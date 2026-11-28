"""Separate the probability of the recipient's next price crossing from its horizon-dependent payoff."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import experiment_dense_directional_peer_expiry_v1 as event


strip, rel, onset, REFERENCE, read_p = event.strip,event.rel,event.onset,event.REFERENCE,event.read_p
OUT = REFERENCE.parent/'dense_peer_passage_mediation_v1'
ARMS = ('first_crossing_price', 'price_after_first_crossing', 'replicated_total_crossing_price')
N = 18
BINS = 3


def design(move,ok,p,floor,edges):
    x = event.features(move,ok,p,floor,edges)
    # Controls include wrong-edge messages so generic market synchrony alone cannot drive the shift.
    shared = x[...,[8,9]]
    extra = np.concatenate((shared*np.tanh(3*np.abs(x[...,4:5])),
                            shared*np.tanh(2*x[...,4:5]*np.sign(shared)),
                            shared*np.tanh(np.abs(p[...,2:3]))),axis=2)
    return np.concatenate((x,strip.center(extra)),axis=2)


def marked_prices(move,ok,floor,t):
    n = len(move)
    hit = event.signed_events(move,ok,floor)
    sign = hit[...,0].astype(float)-hit[...,1]
    # This lookup is used exclusively for supervised examples during fitting and the training-day audit.
    future = np.minimum.accumulate(np.where(sign!=0,np.arange(n)[:,None],n)[::-1],axis=0)[::-1]
    first = future[t+1]
    stocks = np.arange(50)
    delay = first-t[:,None]
    side = sign[first.clip(max=n-1),stocks]
    masks = [(delay>lo)&(delay<=hi) for lo,hi in ((0,6),(6,20),(20,60))]
    marks = np.stack([side*m for m in masks],axis=2)
    path = move.cumsum(0)
    pre = [path[np.minimum(first,t[:,None]+h),stocks]-path[t] for h in (20,60)]
    full = [path[t+h]-path[t] for h in (20,60)]
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    return marks,np.stack(pre,axis=2),np.stack(full,axis=2),gaps[t+61]==gaps[t]


@lru_cache(maxsize=12)
def gram(d):
    floor, edges = event.geometry()
    raw, grid, y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    rows = []
    for a,b,move,ok in event.sessions(raw,grid):
        t = np.arange(55+d%5,b-a-60,5)
        x = design(move,ok,p[a:b],floor,edges)[t]
        m,pre,full,valid = marked_prices(move,ok,floor,t)
        row = np.concatenate((x,m,pre,full,y[a+t,:,1:]),axis=2)
        # Exclude gaps from every centered target, including the settlement mark.
        row -= ((row*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows.append(row[valid])
    row = np.concatenate(rows)
    return row.T@row/len(row)


def solve(g,h,lam):
    return np.linalg.solve(g+lam*np.diag(g.diagonal().clip(1e-8))+1e-10*np.eye(len(g)),h)


def basis(g):
    ctrl = [0,1,2,3,4,5,6,7,10,11]
    idx = [8,9,12,13,14,15,16,17]
    v = np.eye(N)[:,idx]
    v[ctrl] = -solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.04)
    return v


def propensity(g):
    v = basis(g)
    return v@solve(v.T@g[:N,:N]@v,v.T@g[:N,N:N+BINS],.65)


def price(g):
    # Remove surprise already predicted by own state, including the full incumbent curve.
    idx = np.arange(N,N+BINS)
    ctrl = np.arange(8)
    v = np.eye(N+BINS)[:,idx]
    v[ctrl] = -solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.02)
    marks = v.T@g[:N+BINS,:N+BINS]@v
    pre = g[:N+BINS,N+BINS:N+BINS+2]
    full = g[:N+BINS,N+BINS+2:N+BINS+4]
    beta = g[[1,2],[-2,-1]]/g.diagonal()[1:3].clip(1e-8)
    physical = g[[1,2],[N+BINS+2,N+BINS+3]]/g.diagonal()[1:3].clip(1e-8)
    # The post-crossing price can undo the first jump. Estimate both conditional payoffs.
    leg = solve(marks,v.T@pre,.35)/physical.clip(.05,10)
    tail = solve(marks,v.T@(full-pre),.35)/physical.clip(.05,10)
    total = solve(marks,v.T@(g[:N+BINS,-2:]/beta.clip(.2,4)-g[:N+BINS,1:3]),.35)
    return leg,tail,total


def train(end):
    grams = np.stack([gram(d) for d in range(4,end)])
    halves = [x.mean(0) for x in (grams[::2],grams[1::2])]
    cross = np.stack([np.stack([propensity(halves[j])@z for z in price(halves[1-j])],axis=1)
                      for j in (0,1)])
    co = cross.mean(0)
    # Ridge matrices from different dates need not agree coefficient by coefficient.
    # Require positive covariation of the two independently composed prediction streams.
    g = halves[0][:N,:N]+halves[1][:N,:N]
    for h in (0,1):
        a,b = cross[:, :, 2, h]
        co[:,2,h] *= np.clip((a@g@b)/np.sqrt(max(1e-14,(a@g@a)*(b@g@b))),0,1)
    floor,edges = event.geometry()
    return np.r_[floor,edges.ravel(),co.ravel()]


def signals(raw,grid,p,bank):
    floor,edges = bank[:50],bank[50:2550].reshape(50,50)
    w = bank[2550:].reshape(N,6)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        z = design(move,ok,p[a:b],floor,edges)
        out[a:b] = (z@w).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


def audit():
    result = {}
    for end in (8,12):
        past = np.mean([gram(d) for d in range(4,end)],axis=0)
        new = np.mean([gram(d) for d in range(end,end+4)],axis=0)
        w = propensity(past)
        base = np.zeros((N,BINS))
        ids = [0,1,2,3,4,5,6,7,10,11]
        base[ids] = solve(past[np.ix_(ids,ids)],past[ids,N:N+BINS],.04)
        payoff = price(past)
        part = (w.T@(new[:N,N:N+BINS]-new[:N,:N]@base)).diagonal()
        risk = (w.T@new[:N,:N]@w).diagonal()
        reduced_mse = (2*part-risk)/new.diagonal()[N:N+BINS].clip(1e-7)
        result[f'{end+1}-{end+4}'] = dict(mark_relative_mse_reduction=reduced_mse.tolist(),
            trained_Y3_price_at_crossing=payoff[0][:,1].tolist(),
            trained_Y3_price_after_crossing=payoff[1][:,1].tolist())
    (OUT/'passage_audit.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
