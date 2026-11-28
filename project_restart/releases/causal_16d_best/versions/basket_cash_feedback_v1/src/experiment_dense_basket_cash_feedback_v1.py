"""Calibrate the current donor basket forecast using only earlier recipient basket settlements."""
from functools import lru_cache
import sys

import numpy as np
import torch

import prepare_dense_forward_skill_reference_v1 as reference

base = reference.model
strip, pre, event, onset, rel = base.strip, base.pre, base.event, base.onset, base.rel
REFERENCE, read_p = reference.OUT, reference.read_p
OUT = REFERENCE.parent/'dense_basket_cash_feedback_v1'
ARMS = ('donor_short_settlement', 'recipient_specific_response', 'confirmed_short_and_medium')
BASKETS = np.random.default_rng(83345).permutation(50).reshape(5, 10)
C, D = 6, 15


def basket_mean(v, live):
    return (v * live[..., None]).sum(2) / live.sum(2).clip(1)[..., None]


def observations(p, move, ok):
    """Features for five recipient groups. All responses mature at or before each block start."""
    live = ok[:, BASKETS]
    own = basket_mean(p[:, BASKETS].astype(float), live)
    donor = (own.sum(1, keepdims=True) - own)/4
    path = move.cumsum(0)
    gap = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
    x = np.zeros((len(move), 5, D))
    x[:, :, :C] = np.concatenate((own, donor), axis=2)
    # Normalize the innovation by the donor forecast's settled variance.
    # Accumulate observations by block without counting overlapping cash twice in a leg.
    stats = np.zeros((2, 5, 3, 2))
    cursor = [0, 0]
    for start in range(0, len(move), 96):
        for j, (lo, hi) in enumerate(((0, 5), (5, 20))):
            t = np.arange(cursor[j], start-hi+1, hi-lo)
            if not len(t):
                continue
            cursor[j] = t[-1]+hi-lo
            good = (gap[t+hi+1] == gap[t])[:, BASKETS]
            n = good.sum(2)
            realized = ((path[t+hi]-path[t+lo])[:, BASKETS]*good).sum(2)/n.clip(1)
            w = (n>=7).astype(float)
            den = w.sum(0).clip(1)
            z = donor[t] - ((donor[t]*w[..., None]).sum(0)/den[:, None])
            realized -= (realized*w).sum(0)/den
            cov = (z*(w*realized)[..., None]).sum(0)/den[:, None]
            var = (z*z*w[..., None]).sum(0)/den[:, None]
            stats[j] = .84*stats[j] + np.stack((cov/np.sqrt(hi-lo), var), axis=-1)
        if start < 288:
            continue
        slope = stats[..., 0]/(stats[..., 1]+.02)
        short, medium = slope
        shared = short.mean(0, keepdims=True)
        agree = np.where(short*medium>0, np.sign(short)*np.minimum(abs(short), abs(medium)), 0.)
        sl = slice(start, start+96)
        x[sl,:,6:9] = donor[sl]*shared
        x[sl,:,9:12] = donor[sl]*(short-shared)
        x[sl,:,12:15] = donor[sl]*agree
    return x


@lru_cache(maxsize=12)
def record(d):
    raw, grid, y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    xx, yy, pp = [], [], []
    for a, b, move, ok in event.sessions(raw, grid):
        t = np.arange(48+d%4, b-a-60, 4)
        xx.append(observations(p[a:b],move,ok)[t])
        yy.append(y[a+t,:,1:].astype(float))
        pp.append(p[a+t,:,1:].astype(float))
    x, labels, pred = map(np.concatenate,(xx,yy,pp))
    x -= x.mean((0,1))
    labels -= labels.mean((0,1))
    pred -= pred.mean((0,1))
    slope = ((labels*pred).sum((0,1))/(pred*pred).sum((0,1))).clip(.2,4)
    err = (labels/slope-pred)[:,BASKETS].mean(2).reshape(-1,2)
    x = x.reshape(-1,D)
    return x.T@x/len(x), x.T@err/len(x)


def train(end):
    cov, rhs = map(np.stack, zip(*(record(d) for d in range(4,end))))
    g = cov.mean(0)
    weights = []
    for inds in np.array_split(np.arange(C,D),3):
        v = np.eye(D)[:,inds]
        v[:C] = -pre.peer.base.solve(g[:C,:C],g[:C,inds],.07)
        marginal = v.T@rhs
        avg = marginal.mean(0)
        safe = avg**2/(avg**2+2*marginal.var(0)/len(rhs)+1e-14)
        weights.append(v@pre.peer.base.solve(v.T@g@v,avg*safe,.85))
    return np.stack(weights,axis=1).ravel()


def signals(raw,grid,p,params):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        x = observations(p[a:b],move,ok)
        correction = (x@params.reshape(D,6)).reshape(b-a,5,3,2)
        out[a:b,BASKETS] = correction[:,:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
