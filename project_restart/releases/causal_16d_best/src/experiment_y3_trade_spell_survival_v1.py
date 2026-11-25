"""Predict the future signed flow before the current run of same-side old-quote fills ends."""
from functools import lru_cache
import sys

import numpy as np
import torch

import y3_inferred_print_references as history

model, study, trial = history.incumbent, history.study, history.trial
book = model.book
OUT = trial.clock.ART/'y3_trade_spell_survival_v1'
ARMS = ('present direction and activity', 'future activity before first reversal', 'net subsequent flow',
        'flow after reversal', 'remaining flow and quote response', 'constant-length spell',
        'other stock renewal table cell', 'twenty-to-sixty step remaining flow')
CANDIDATES = (1, 2, 3, 4, 7)
CONTROLS = (0, 5, 6)


def state(raw, params):
    _, _, price, old, side, live, sizes = model.prints(raw, params)
    action = np.where(live & (price==old), side, 0)
    now = np.arange(len(raw))[:, None]
    stock = np.arange(50)[None, :]
    previous = np.maximum.accumulate(np.where(action!=0, now, -1), axis=0)
    prev = np.r_[0, np.arange(len(raw)-1)]
    last_side = action[previous.clip(0), stock]
    flip = (action!=0) & (action!=last_side[prev])
    flip[0] = action[0]!=0
    begin = np.maximum.accumulate(np.where(flip, now, 0), axis=0)
    count = np.vstack((np.zeros(50, dtype=int), (action!=0).cumsum(0)))
    length = count[1:]-count[begin, stock]
    gap = now-previous
    alive = (previous>=0) & (gap<=80)
    pace = (previous-begin).clip(0)/np.maximum(1, length-1)
    recent = count[1:]-count[(now[:, 0]-59).clip(0)]
    b = np.digitize(length, (2, 3, 5, 9))
    a = np.digitize(gap, (1, 3, 8, 20))
    f = np.digitize(recent, (4, 11, 22))
    speed = np.digitize(pace, (2., 7.))
    broad = a*4+f
    granular = (broad*5+b)*3+speed
    return action, last_side, alive, broad, granular, length


HORIZONS = ((0, 5), (5, 20), (20, 40), (40, 60))


@lru_cache(maxsize=16)
def observations(day):
    raw = trial.clock.book.rawday(day)
    action, sign, alive, broad, group, length = state(raw, model.parameters())
    t = np.arange(120+day%4, len(raw)-60, 4)
    now = np.arange(len(raw))[:, None]
    cs = np.vstack((np.zeros(50), action.cumsum(0)))
    future = []
    for direction in (-1, 1):
        # Used only for auxiliary labels on completed training dates.
        reverse = np.minimum.accumulate(np.where(action==-direction, now, len(raw))[::-1], axis=0)[::-1]
        future.append(reverse[t+1])
    end = np.where(sign[t]>0, future[1], future[0])
    stocks = np.arange(50)[None, :]
    remain, total = [], []
    for lo, hi in HORIZONS:
        a, b = t[:, None]+1+lo, t[:, None]+1+hi
        remain.append((cs[np.minimum(end, b), stocks]-cs[np.minimum(end, a), stocks])*sign[t])
        total.append((cs[b, stocks]-cs[a, stocks])*sign[t])
    y = np.stack((*remain, *total), axis=2)[alive[t]]
    gs = []
    for bins, size in ((broad, 20), (group, 300)):
        keys = bins[t][alive[t]]
        counts = np.bincount(keys, minlength=size)
        sums = np.stack([np.bincount(keys, weights=target, minlength=size) for target in y.T], axis=1)
        gs.append(np.column_stack((counts, sums)))
    return gs


@lru_cache(maxsize=6)
def parameters(end):
    levels = [observations(d) for d in range(end)]
    broad, cells = [sum(z) for z in zip(*levels)]
    general = broad[:, 1:]/broad[:, :1].clip(1)
    smooth = (cells[:, 1:]+160*general[np.arange(300)//15])/(160+cells[:, :1])
    return dict(**model.parameters(), cells=smooth, broad=general)


def design_with(raw, p, t, params):
    action, sign, alive, broad, group, length = state(raw, params)
    orient = (sign[t]*alive[t])[:, :, None]
    full = params['cells'][group[t]]*orient
    base = params['broad'][broad[t]]*orient
    wrong = params['cells'][np.roll(group[t], 17, axis=1)]*orient
    cs = np.vstack((np.zeros(50), action.cumsum(0)))
    history_flow = np.stack([(cs[t+1]-cs[(t+1-h).clip(0)])/np.sqrt(h) for h in (5, 20, 60)], 2)
    directional = np.tanh(sign[t]*p[t, :, 0])[:, :, None]
    parts = (p[t], history_flow, (np.log1p(length[t])*orient[:, :, 0])[:, :, None],
             full[:, :, :4], full[:, :, 4:], full[:, :, 4:]-full[:, :, :4],
             full[:, :, :4]*directional, base[:, :, :4], wrong[:, :, :4])
    return trial.clock.center(np.concatenate(parts, 2)).astype('float32')


def historical_design(day):
    params = parameters(max(4, day-day%2))
    return lambda raw, p, t: design_with(raw, p, t, params)


def fit(rows):
    present, stay, net, after, response, broad, wrong = [np.arange(3+4*i, 7+4*i) for i in range(7)]
    specs = [(present, 3), (stay, 7), (net, 7), (after, 7), (np.r_[stay, response], 7),
             (broad, 7), (wrong, 7), (stay[-2:], 7)]
    return dict(**book.prev.gram(rows, [(g, n, .4) for g, n in specs]), **parameters(len(rows)+4))


changes = book.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        params = dict(f)
    def update(raw, p):
        return changes(design_with(raw, p, np.arange(len(raw)), params), params)[int(params['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__], model)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__], reference=history)
