"""Differentiate single-side executions inside, at, and beyond the previous quote."""
from functools import lru_cache
import sys

import numpy as np
import torch

import y3_quote_token_references as history

events, study, trial = history.incumbent, history.study, history.trial
book = events.model
OUT = trial.clock.ART/'y3_inferred_print_impacts_v1'
ARMS = ('signed single-side volume', 'fills beyond the old quote', 'fills inside the old quote',
        'fills at the old quote', 'three location classes', 'mirror the inferred price',
        'stale comparison quote', 'beyond fills and quote response', 'price paid relative to old quote')
CANDIDATES = (1, 2, 3, 4, 7, 8)
CONTROLS = (0, 5, 6)


@lru_cache(maxsize=1)
def parameters():
    floors = trial.clock.book.rawday(0)[:, :, [8, 54, 55]].min((0, 1)).astype(float)
    values = []
    for d in range(4):
        total = (trial.clock.book.rawday(d)[:, :, 8].astype(float)-floors[0]).clip(0, None)
        values.append(np.minimum(total, np.quantile(total, .995, axis=0)).mean(0))
    return dict(**book.parameters(), floors=floors, volume_scale=np.mean(values, axis=0).clip(.002))


def prints(raw, state):
    mid, width, offset, valid = book.quote.infer(raw, float(state['k']), state['coefficients'])
    floors = state['floors']
    buy, sell = raw[:, :, 54]>floors[1], raw[:, :, 55]>floors[2]
    sign = buy.astype(int)-sell.astype(int)
    ix = np.r_[0, np.arange(len(raw)-1)]
    old_valid = valid & valid[ix] & (sign!=0)
    old_valid[0] = False
    total = (raw[:, :, 8].astype(float)-floors[0]).clip(0, None)
    volume = np.log1p(total/state['volume_scale']).clip(0, 6)
    sizes = np.stack((sign, sign*volume), axis=2)*old_valid[:, :, None]
    price = mid+offset
    prior = mid[ix]+sign*width[ix]
    return mid, width, price, prior, sign, old_valid, sizes


def design_with(raw, p, t, state):
    mid, width, actual, prior, sign, valid, sizes = prints(raw, state)
    d = sign*(actual-prior)
    move = sign*(mid+sign*width-prior)
    mirror = sign*(2*mid-actual-prior)
    j = (np.arange(len(raw))-8).clip(0)
    elsewhere = sign*(actual-mid[j]-sign*width[j])
    masks = (np.ones_like(d), d>0, d<0, d==0, mirror>0, elsewhere>0,
             (d>0)*np.tanh(move/2), np.arcsinh(d.clip(-30, 30)/2))
    groups = []
    for mask in masks:
        v = sizes*mask[:, :, None]
        cs = np.concatenate((np.zeros_like(v[:1]), v.cumsum(0)))
        groups.append(np.concatenate([(cs[t+1]-cs[(t+1-h).clip(0)])/np.sqrt(h) for h in (5, 20, 60, 180)], axis=2))
    return trial.clock.center(np.concatenate((p[t], *groups), axis=2)).astype('float32')


def design(raw, p, t):
    return design_with(raw, p, t, parameters())


def fit(rows):
    base, beyond, inside, at, reverse, stale, response, paid = [np.arange(3+8*i, 11+8*i) for i in range(8)]
    bank = [(base, 3), (beyond, 11), (inside, 11), (at, 11), (np.r_[beyond, inside, at], 11),
            (reverse, 11), (stale, 11), (np.r_[beyond, response], 11), (paid, 11)]
    return dict(**book.prev.gram(rows, [(g, n, .40) for g, n in bank]), **parameters())


changes = book.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        state = dict(f)
    def update(raw, p):
        return changes(design_with(raw, p, np.arange(len(raw)), state), state)[int(state['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__], events)


if __name__ == '__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__], reference=history)
