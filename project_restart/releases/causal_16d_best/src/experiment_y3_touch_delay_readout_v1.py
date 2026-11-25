"""Reweight the three disjoint future bands by trade/pressure separation and event age."""
from functools import lru_cache
import sys

import numpy as np
import torch

import y3_touch_clock_references as history

touch, study, trial = history.incumbent, history.study, history.trial
OUT = trial.clock.ART/'y3_touch_delay_readout_v1'
ARMS = ('sum of bands','separate payoffs','time since events','gap between events','short versus long separation',
        'interaction between delay and age','ordered gaps with matched age','permuted ages','middle and late by age',
        'middle and late by separation','time order plus recent execution direction')
CANDIDATES = (1,2,3,4,5,6,8,9,10)
CONTROLS = (0,7)


def features(raw,t,st,w):
    vals = touch.inputs(raw,st)[:,:,[0,1,2,3,8,9,10]]
    blocks = []
    for lo,hi in ((0,min(len(vals),2401)), (2401,len(vals))):
        idx = t[(t>=lo)&(t<hi)]-lo
        if not len(idx):
            continue
        v = vals[lo:hi]-touch.clocks.filters.smooth(vals[lo:hi],128)
        p,a = v[:,:,:4],v[:,:,4:]
        banks = []
        wrong = []
        for i,gap in enumerate(touch.SPANS):
            past = touch.clocks.filters.smooth(v,gap)
            lag = np.concatenate((np.zeros_like(past[:1]),past[:-1]))
            z = (lag[:,:,:4,None]*a[:,:,None,:]-p[:,:,:,None]*lag[:,:,None,4:]).reshape(len(v),50,12)
            ages, rev = [], []
            for j,age in enumerate(touch.SPANS):
                head = w[12*(3*i+j):12*(3*i+j+1)]
                local = z@head
                ages.append(touch.clocks.filters.smooth(local,age)[idx])
                rev.append(touch.clocks.filters.smooth(local,touch.SPANS[2-j])[idx])
            banks.append(np.stack(ages,2))
            wrong.append(sum(rev))
        bank = np.stack(banks,2)
        gap,age = bank.sum(3),bank.sum(2)
        total = gap.sum(2)
        interaction = bank-bank.mean(3,keepdims=True)-bank.mean(2,keepdims=True)+bank.mean((2,3),keepdims=True)
        diag = sum(bank[:,:,j,j] for j in range(3))
        direction = np.tanh(3*touch.clocks.filters.smooth(v[:,:,2],12)[idx])
        sels = [total.sum(2,keepdims=True), total, age.reshape(len(idx),50,9), gap.reshape(len(idx),50,9),
                gap[:,:,0]-gap[:,:,2], interaction[:,:,:2,:2].reshape(len(idx),50,12),diag,
                sum(wrong), age[:,:,:,1:].reshape(len(idx),50,6), gap[:,:,:,1:].reshape(len(idx),50,6),
                total*np.abs(direction[:,:,None])]
        blocks.append(trial.clock.center(np.concatenate(sels,2)))
    return np.concatenate(blocks)


@lru_cache(maxsize=7)
def learned(end):
    return touch.params(end)[3]


def historical_design(d):
    cache = {}
    def build(raw,p,t):
        key = len(raw),t.tobytes()
        if key not in cache:
            cache[key] = features(raw,t,touch.queues.calibration(),learned(max(4,d-d%2)))
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    stops = np.cumsum([3,1,3,9,9,3,12,3,3,6,6,3])
    group = [np.arange(a,b) for a,b in zip(stops,stops[1:])]
    return dict(**touch.clocks.linear.gram(rows,[(cols,3,.55) for cols in group]),
                heads=learned(len(rows)+4),**touch.queues.calibration())


changes = touch.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st = dict(f)
    def apply(raw,p):
        x = features(raw,np.arange(len(raw)),st,st['heads'])
        return changes(np.concatenate((trial.clock.center(p),x),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
