"""Freeze recipient price state at a confirmed market shock and track later absorption."""
from functools import lru_cache
import sys

import numpy as np
from scipy.signal import lfilter
import torch

import experiment_dense_settled_market_news_v1 as news

base, event, strip = news.base, news.event, news.strip
REFERENCE, read_p, onset, rel = news.REFERENCE, news.read_p, news.onset, news.rel
OUT = REFERENCE.parent / 'dense_confirmed_shock_response_v1'
ARMS = ('initial_absorption', 'subsequent_absorption', 'book_confirmation')
C, D = 10, 16


def design(m, ok, p, raw):
    t = np.arange(len(m))
    prices = m.cumsum(axis=0)
    gap = np.vstack((np.zeros(50, int), (~ok).cumsum(axis=0)))
    donor = news.market.window(news.means(m, ok), 5)
    own = news.market.window(m, 5)
    pressure = base.depth(raw) @ base.axis()
    # Each variance estimate ends before the five-tick move under examination.
    mass = news.ew(np.ones(len(m)), 300)[:, None]
    prev = (t - 5).clip(0)
    own_sd = np.sqrt((news.ew(own**2, 300) / mass + .01)[prev])
    features = np.zeros((*m.shape, D))
    features[..., :3] = p
    features[..., 3] = np.tanh(own / own_sd)
    features[..., 4] = np.tanh(pressure)
    for k, ids in enumerate(news.GROUP):
        sides = news.GROUP.shape[0]
        peers = [(k + i) % sides for i in (1, 2, 3, 4)]
        x = np.column_stack((donor[:, peers[:2]].mean(1), donor[:, peers[2:]].mean(1)))
        sd = np.sqrt((news.ew(x**2, 300) / mass + .0001)[prev])
        z = x / sd
        agreed = (z[:, 0] * z[:, 1] > 0) & (abs(z).min(1) >= 1.35)
        sign = np.sign(x.mean(1))
        started = agreed & ((~np.r_[False, agreed[:-1]]) | (sign != np.r_[0., sign[:-1]])) & (t >= 40)
        last = np.maximum.accumulate(np.where(started, t, -1000))
        s = last.clip(0)
        age = t - last
        valid = (gap[t+1][:, ids] == gap[(s-5).clip(0)][:, ids])
        pulse = (sign[s] * np.minimum(abs(z[s]).min(1), 3.) * np.exp(-age/20) * (age <= 40))[:, None] * valid
        side = sign[s, None]
        at_event = side * own[s][:, ids] / own_sd[s][:, ids]
        after = side * (prices[:, ids] - prices[s][:, ids]) / own_sd[s][:, ids]
        book = side * pressure[s][:, ids]
        features[:, ids, 5] = pulse
        features[:, ids, 6] = pulse * np.exp(-age[:, None]/5)
        features[:, ids, 7] = np.tanh(at_event) * side * valid * (age <= 40)[:, None]
        features[:, ids, 8] = np.tanh(after) * side * valid * (age <= 40)[:, None]
        features[:, ids, 9] = side * np.tanh(book) * valid * (age <= 40)[:, None]
        # Separate stocks with a flat response from those moving against the source basket.
        for j, state in enumerate((at_event, after, book)):
            features[:, ids, C+2*j] = pulse * np.exp(-abs(state))
            features[:, ids, C+2*j+1] = pulse * np.tanh(state)
    return strip.center(features)


@lru_cache(maxsize=12)
def record(d):
    raw, grid, y = [onset.settled.load(d, k) for k in ('x', 'time', 'y')]
    p = read_p(d)
    gram = np.zeros((D+2, D+2))
    n = 0
    for a, b, m, ok in event.sessions(raw, grid):
        t = np.arange(40+d%5, b-a-60, 5)
        f = design(m, ok, p[a:b], raw[a:b])[t]
        gap = np.vstack((np.zeros(50,int), (~ok).cumsum(0)))
        valid = gap[t+61] == gap[t]
        z = np.concatenate((f, y[a+t,:,1:]), axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        gram += rows.T @ rows
        n += len(rows)
    return gram / n


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    g = days.mean(0)
    beta = (days[:,[1,2],[D,D+1]] / days[:,[1,2],[1,2]]).clip(.2,4)
    err = days[:,:D,D:] / beta[:,None] - days[:,:D,1:3]
    co = []
    for j in range(3):
        idx = np.arange(C+2*j,C+2*j+2)
        ctrl = np.arange(C)
        v = np.eye(D)[:,idx]
        v[ctrl] = -base.solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.04)
        day = v.T @ err
        mu = day.mean(0)
        pay = mu**3/(mu**2+2*day.var(0)/len(days)+1e-12)
        co.append(v @ base.solve(v.T@g[:D,:D]@v,pay,.65))
    return np.stack(co,axis=1)


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = design(m,ok,p[a:b],raw[a:b])
        out[a:b] = (x@bank.reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
