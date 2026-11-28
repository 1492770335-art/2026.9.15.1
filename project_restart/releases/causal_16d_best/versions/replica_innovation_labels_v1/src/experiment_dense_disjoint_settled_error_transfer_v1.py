"""Learn low-rank transfer of settled forecast errors between disjoint stock baskets."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_stock_surprise_response_v1 import REFERENCE, read_p, strip, event, onset, rel

OUT = REFERENCE.parent / 'dense_disjoint_settled_error_transfer_v1'
ARMS = ('two_donor_panel_mean', 'independent_panels_agree', 'reproducible_transfer_directions')
GROUPS = np.random.default_rng(92713).permutation(50).reshape(5, 10)
HORIZONS = (5, 20, 60)
NINPUT = 3*20


@lru_cache(maxsize=12)
def units(day):
    _, coef, _ = strip.physical()
    slope = strip.daily_scale(read_p(day), onset.settled.load(day, 'y')).clip(.2, 4.)
    return slope*coef[2]/coef


def settled(move, ok, p, unit):
    path = move.cumsum(0)
    bad = (~ok).cumsum(0)
    out = np.zeros(p.shape, dtype=float)
    for j,h in enumerate(HORIZONS):
        if len(p) > h:
            valid = (bad[h:] == bad[:-h]) & ok[:-h]
            error = (path[h:]-path[:-h])/unit[j]-p[:-h,:,j]
            out[h:,:,j] = np.where(valid, error, 0.)
    return out


def donor_groups(i):
    return [GROUPS[[(i+shift)%5 for shift in pair]].ravel() for pair in ((1,3), (2,4))]


def features(errors, donors):
    e = errors[:,donors]
    return (e-e.mean(1, keepdims=True)).reshape(len(e), NINPUT)


@lru_cache(maxsize=48)
def summary(day, unit):
    raw, grid, y = [onset.settled.load(day, key) for key in ('x','time','y')]
    p = read_p(day)
    rows, target, forecast = [], [], []
    for a,b,move,ok in event.sessions(raw,grid):
        times = np.arange(60+day%5, b-a-60, 5)
        if len(times):
            rows.append(settled(move,ok,p[a:b],unit)[times])
            target.append(y[a+times,:,1:])
            forecast.append(p[a+times,:,1:])
    e, y, p = [np.concatenate(values).astype(float) for values in (rows,target,forecast)]
    y -= y.mean((0,1))
    p -= p.mean((0,1))
    slope = ((p*y).sum((0,1))/(p*p).sum((0,1))).clip(.2,4.)
    residual = y/slope-p
    residual -= residual.mean(1,keepdims=True)
    residual -= residual.mean(0)
    xs, ys = [], []
    for i, recipient in enumerate(GROUPS):
        for donors in donor_groups(i):
            x = features(e,donors)
            x -= x.mean(0)
            xs.append(x.T@x/len(x))
            ys.append(x.T@residual[:,recipient].reshape(len(x),-1)/len(x))
    return np.stack(xs),np.stack(ys)


def train(end):
    unit = tuple(np.median([units(d) for d in range(4,end)],axis=0))
    grams, cross = map(np.stack,zip(*(summary(d,unit) for d in range(4,end))))
    weights = []
    for i in range(10):
        g = grams[:,i].mean(0)
        sd = g.diagonal().clip(1e-5)**.5
        vals, vec = np.linalg.eigh(g/np.outer(sd,sd))
        whiten = ((vec/(vals.clip(0)+1.2)**.5)@vec.T)/sd[:,None]
        # The covariance is not diagonal. Preserve the transpose when whitening
        # the input-output cross moment.
        z = whiten.T @ cross[:,i]
        u,s,v = np.linalg.svd(z.mean(0), full_matrices=False)
        full = whiten @ ((u[:,:4]*s[:4])@v[:4])
        # Directions must predict the same recipient exposures on alternating dates.
        left,right = (z[j::2].mean(0) for j in range(2))
        agree = .5*(left.T@right+right.T@left)
        val, basis = np.linalg.eigh(agree)
        val,basis = val[-4:].clip(0),basis[:,-4:]
        gain = val/(val+.03*np.trace(z.mean(0).T@z.mean(0))+1e-10)
        stable = whiten @ z.mean(0) @ ((basis*gain)@basis.T)
        weights.append(np.stack((full,stable)))
    return np.r_[unit,np.stack(weights).ravel()]


def signals(raw,grid,p,bank):
    w = bank[3:].reshape(5,2,2,NINPUT,20)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        e = settled(move,ok,p[a:b],bank[:3])
        for i,recipient in enumerate(GROUPS):
            messages = [np.einsum('td,kdo->kto',features(e,donors),w[i,j]).reshape(2,b-a,10,2)
                        for j,donors in enumerate(donor_groups(i))]
            left,right = messages
            shared = np.where(left[0]*right[0]>0,
                              np.sign(left[0])*np.minimum(abs(left[0]),abs(right[0])),0.)
            out[a:b,recipient] = np.stack(((left[0]+right[0])*.5,shared,(left[1]+right[1])*.5),axis=2)
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
