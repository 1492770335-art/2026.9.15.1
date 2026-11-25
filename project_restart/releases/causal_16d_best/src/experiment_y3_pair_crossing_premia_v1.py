"""Measure the value of persistent relative forecasts against matched stocks, with additive lag controls."""
import sys

import numpy as np
import torch

import audit_y3_pair_crossings as audit
import y3_risk_rank_validation as history

trial = history.parent
OUT = trial.clock.ART/'y3_pair_crossing_premia_v1'
ARMS = ['current and lagged forecasts', 'pair persistence', 'short confirmation', 'joint confirmation',
        'confirmed only near ties', 'wrong historical stock', 'longer persistence', 'half joint confirmation']


def state(raw):
    return None


def spread(v):
    return v[:, :, None] - v[:, None, :]


def design(p, t):
    q = audit.tail(p)
    short = spread(p[t, :, 1])
    current = spread(q[t])
    orient = np.sign(current)
    scale = .015 + abs(current).mean((1, 2), keepdims=True)
    gap = abs(current)/scale
    strength = np.tanh(gap)
    agreement = orient*np.sign(short)
    # Every stock plays the other 49 at the same tick. Equal weighting avoids identity-dependent sampling.
    def project(v):
        return (orient*v).sum(2)/49
    terms = [trial.clock.center(p[t]), trial.clock.center(np.stack([q[(t-h).clip(0)] for h in (8,32,120)],2))]
    def pair_history(h, shift=False):
        past = q[(t-h).clip(0)]
        if shift:
            past = np.roll(past, 17, axis=1)
        return np.sign(spread(past))*orient
    cols = []
    for lag in (8,32,120):
        pers = pair_history(lag)
        cols.extend((project(pers), project(pers*strength), project(pers*np.exp(-2*gap))))
    cols.extend((project(agreement),project(agreement*strength)))
    for h in (8,32,120):
        both = pair_history(h)*agreement
        cols.extend((project(both), project(both*np.exp(-2*gap))))
    for h in (8,32,120):
        wrong = pair_history(h, True)
        cols.extend((project(wrong),project(wrong*strength),project(wrong*np.exp(-2*gap))))
    return np.concatenate((*terms, np.stack(cols,2)),2).astype('float32')


def record(p,y,d):
    t = np.arange(120+d%7, len(p)-60, 7)
    q = trial.clock.center(p[t, :, 2])
    truth = trial.clock.center(y[t, :, 2])
    late = trial.clock.center(audit.tail(y[t]))
    pred = trial.clock.center(audit.tail(p[t]))
    def residual(v, forecast):
        beta = np.sum(forecast*v, dtype=float)/np.sum(forecast**2, dtype=float).clip(1e-7)
        return v/np.clip(beta,.4,2.)-forecast
    targets = np.stack((residual(truth,q), residual(late,pred)),2).reshape(-1,2)
    x = np.concatenate([design(p,w) for w in np.array_split(t, (len(t)+63)//64)]).reshape(len(targets),-1)
    return x.T@x/len(x), x.T@targets/len(x)


def fit(hist):
    gs, bs = [np.stack(v).astype(float) for v in zip(*hist)]
    g = gs.mean(0)
    def solve(ids, adjust=True):
        idx = np.asarray(ids)
        basis = np.eye(len(g))[:,idx]
        if adjust:
            basis[:6] = -trial.curve.solve(g[:6,:6], g[:6,idx], .025)
        b = basis.T @ bs
        mu = b.mean(0)
        mu *= abs(mu)/(abs(mu)+b.std(0)/np.sqrt(len(bs))+1e-7)
        # Favor corrections that explain both the long label error and its final forty ticks.
        shared = np.where(mu[:,0]*mu[:,1]>0, np.sign(mu[:,0])*np.minimum(abs(mu[:,0]),abs(mu[:,1])), 0)
        return basis @ trial.curve.solve(basis.T@g@basis, .65*mu[:,0]+.35*shared, .8)
    all_history = solve(range(6,23))
    weights = np.stack((solve(range(6),False), solve(range(6,15)),solve([15,16]),all_history,
                        solve([8,11,14,18,20,22]),solve(range(23,32)),solve([12,13,14,21,22]),.5*all_history),1)
    return dict(w=(.44*weights).astype('float32'))


def changes_with_reference(z,m,p):
    out = np.concatenate([design(p,t) @ m['w'] for t in np.array_split(np.arange(len(p)),(len(p)+95)//96)])
    return list(np.moveaxis(out,2,0))


def select(f,d):
    good=[]
    for i in (1,2,3,4,6,7):
        v=f[:,:,0,i]
        if ((v>0).all() and (d[:,0,i]>0).sum()>=4 and f[:,:,1,i].mean(0).min()>0
                and (v-f[:,:,0,5]).mean(0).min()>0):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    return history.correction(sys.modules[__name__])


def layer():
    return history.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
