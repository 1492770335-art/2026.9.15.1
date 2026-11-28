"""Reduce long-horizon training-target noise using five-tick forecast errors after confirmed price events."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_live_donor_settlements_v1 as live

pre = live.donor.pre
strip, event, onset, rel = live.strip, live.event, live.onset, live.rel
REFERENCE, read_p = live.REFERENCE, live.read_p
OUT = REFERENCE.parent/'dense_stopped_shock_teacher_v1'
ARMS = ('direct_cash', 'remove_after_retracement', 'remove_after_retention')
C, D = 3, 3+pre.H+2*pre.K


def confirmed(m, ok):
    t = np.arange(len(m))
    lag = (t-5).clip(0)
    path = m.cumsum(0)
    left = (lag-1).clip(0)
    gap = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
    seen = (t[:, None]>=6)&(gap[t+1]==gap[left])&(abs(m[lag])>.001)
    moved = (path-path[left])*np.sign(m[lag])
    category = np.where(moved>=.5*abs(m[lag]), 2, 1)
    last = np.maximum.accumulate(np.where(seen, t[:, None], -999), axis=0)
    return np.where(t[:, None]-last<=10, category[last.clip(0), np.arange(50)], 0)


def short_samples(m, ok, p):
    n = len(m)-5
    x = np.concatenate((np.ones((*m.shape, 1)), p.astype(float)), axis=2)
    path = m.cumsum(0)
    delta = path[5:]-path[:-5]
    gap = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
    good = gap[6:]==gap[:n]
    state = confirmed(m, ok)[:n]
    return x[:n], delta, state, good


@lru_cache(maxsize=16)
def teacher_moments(day):
    raw, grid = [onset.settled.load(day, k) for k in ('x','time')]
    p = read_p(day)
    grams = np.zeros((3, 5, 5))
    for a, b, m, ok in event.sessions(raw, grid):
        x, delta, state, good = short_samples(m, ok, p[a:b])
        z = np.concatenate((x, delta[..., None]), axis=2)
        for i in range(3):
            v = z[good&(state==i)]
            grams[i] += v.T@v
    return grams


def short_coefs(day):
    # Event-conditional short-value calibration strictly predates each training example.
    if day==4:
        _, units, _ = strip.physical()
        w = np.zeros((3, 4))
        w[:, 1] = units[2]/units[0]
        return w
    g = np.mean([teacher_moments(d) for d in range(4, day)], axis=0)
    cov = g[:, :-1, :-1]
    ridge = np.stack([.03*np.diag(mat.diagonal().clip(.01)) for mat in cov])
    return np.linalg.solve(cov+ridge, g[:, :-1, -1, None])[..., 0]


def calculate_moments(day, transform=None):
    raw, grid, y = [onset.settled.load(day, k) for k in ('x','time','y')]
    p = read_p(day)
    base = pre.encoder()
    frozen = pre.restore(base)
    beta = short_coefs(day)
    g, n = np.zeros((D+6, D+6)), 0
    for a, b, m, ok in event.sessions(raw, grid):
        x, delta, state, keep = short_samples(m, ok, p[a:b])
        short_error = (delta-(x*beta[state]).sum(2))*keep
        t = np.arange(50+day%4, len(m)-60, 4)
        gap = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        history = pre.views(raw[a:b], p[a:b], base, frozen)[t]
        if transform is not None:
            history = transform(history, ok[t])
        labels = [y[a+t, :, 1:]]
        _, denom, _ = strip.physical()
        for s in (1, 2):
            for_leg = short_error*(state==s)
            correction = np.stack([sum(for_leg[t+j] for j in range(5, h, 5))
                                   for h in (20, 60)], axis=2)
            labels.append(labels[0]-correction*(denom[1:]/denom[2]))
        z = np.concatenate((history, *labels), axis=-1)
        z -= ((z*valid[..., None]).sum(1)/valid.sum(1)[:, None].clip(1))[:, None]
        flat = z[valid]
        g += flat.T@flat
        n += len(flat)
    return g/n


@lru_cache(maxsize=12)
def moments(day):
    return calculate_moments(day)


def fit(gs):
    g = gs.mean(0)
    ix = np.arange(C, D)
    v = np.eye(D)[:, ix]
    v[:C] = -pre.peer.base.solve(g[:C, :C], g[:C, ix], .025)
    slope = (gs[:, [1,2], [D,D+1]]/gs[:, [1,2], [1,2]]).clip(.2, 4)
    co = []
    for j in range(3):
        residual = gs[:, :D, D+2*j:D+2*j+2]/slope[:, None]-gs[:, :D, 1:3]
        r = v.T@residual
        avg = r.mean(0)
        safe = avg**2/(avg**2+3*r.var(0)/len(gs)+1e-14)
        co.append(v@pre.peer.base.solve(v.T@g[:D,:D]@v, avg*safe, .85))
    return np.r_[pre.encoder(), np.stack(co, axis=1).ravel()]


def train(end):
    return fit(np.stack([moments(d) for d in range(4, end)]))


def signals(raw, grid, p, params):
    bank = params[:-D*6]
    model = pre.restore(bank)
    out = np.zeros((*p.shape[:2], 3, 2))
    for a, b, _, _ in event.sessions(raw, grid):
        x = pre.views(raw[a:b], p[a:b], bank, model)
        out[a:b] = (x@params[-D*6:].reshape(D, 6)).reshape(b-a, 50, 3, 2)
    return out


def predict(raw, grid, p, params):
    return strip.evaluation.apply(sys.modules[__name__], raw, grid, p, params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
