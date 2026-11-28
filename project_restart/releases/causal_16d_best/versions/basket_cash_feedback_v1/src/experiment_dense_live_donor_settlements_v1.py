"""Calibrate the information in two donor baskets against only already-settled recipient returns."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_two_basket_market_gru_v1 as donor

strip, event, onset, rel = donor.strip, donor.event, donor.onset, donor.rel
REFERENCE, read_p = donor.REFERENCE, donor.read_p
OUT = REFERENCE.parent/'dense_live_donor_settlements_v1'
ARMS = ('market_settlements', 'relative_settlements', 'market_and_relative')
B = donor.BASKETS
HORIZONS = (20, 60)
CONTROL = 4


@torch.inference_mode()
def basket_views(raw, ok, base, bank):
    model = donor.MarketGRU().double().eval()
    nn.utils.vector_to_parameters(torch.tensor(bank[2*donor.L+3:], dtype=torch.float64), model.parameters())
    model.gru.flatten_parameters()
    state = donor.panel(raw, ok, base, donor.pre.restore(base))
    state = ((state-bank[:donor.L])/bank[donor.L:2*donor.L]).clip(-5, 5)
    output = []
    for start in range(0, len(raw), 64):
        t = np.arange(start, min(start+64, len(raw)))
        z = torch.tensor(donor.windows(state, t), dtype=torch.float64)
        z = F.pad(z, (0, 0, 0, 0, 0, 0, 0, 0, 0, 64-len(t)))
        frame, hist, agree, _ = model(z)
        legs = torch.stack((frame, hist, agree), dim=-1).cpu().numpy()[:len(t)]
        output.append(np.cumsum(legs*bank[2*donor.L:2*donor.L+3, None], axis=2)[:, :, 1:])
    return np.concatenate(output)


@lru_cache(maxsize=16)
def calibration(day):
    raw, grid = [onset.settled.load(day, k) for k in ('x', 'time')]
    p = read_p(day)
    stats = np.zeros((2, 2))
    for a, b, m, ok in event.sessions(raw, grid):
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
        for j, h in enumerate(HORIZONS):
            t = np.arange(48+day%4, len(m)-h, 4)
            live = gaps[t+h+1]==gaps[t]
            q = p[a+t, :, j+1][live].astype(float)
            cash = (path[t+h]-path[t])[live]
            q -= q.mean()
            cash -= cash.mean()
            stats[j] += (q@cash, q@q)
    return stats


def train(end):
    g = np.stack([calibration(d) for d in range(4, end)])
    unit = np.median(g[:, :, 0]/g[:, :, 1].clip(1e-10), axis=0).clip(.03, 6.)
    base = donor.pre.encoder()
    return np.r_[len(base), unit, base, donor.encoder()]


def residual_update(moments, observations):
    # Units of the response are the incumbent's forecast units. The effective
    # sample size is counted in non-overlapping time blocks, not individual stocks.
    if len(moments)<3:
        return np.zeros(observations.shape[:-1])
    n = min(len(moments), 12)
    grams = np.stack(moments[-n:])
    weights = .90**np.arange(n-1, -1, -1)
    weights /= weights.sum()
    g = weights@grams.reshape(n, -1)
    g = g.reshape(grams.shape[1:])
    scale = np.sqrt(np.diag(g)[:-1].clip(1e-9))
    normal = g[:-1, :-1]/scale[:, None]/scale
    v = np.eye(len(scale))[:, CONTROL:]
    v[:CONTROL] = -np.linalg.solve(normal[:CONTROL, :CONTROL]+.1*np.eye(CONTROL),
                                   normal[:CONTROL, CONTROL:])
    transform = v/scale[:, None]
    sides = grams[:, :-1, -1]@transform
    avg = weights@sides
    uncertainty = weights@((sides-avg)**2)*np.sum(weights**2)
    reward = avg*avg**2/(avg**2+2*uncertainty+1e-12)
    w = np.linalg.solve(v.T@normal@v+.7*np.eye(2), reward)
    return (observations@(transform@w)).clip(-.5, .5)


def settle(p, m, ok, views, j, unit):
    h = HORIZONS[j]
    pred = p[:, B].mean(2, dtype=np.float64)
    obs = np.concatenate((pred, views[:, :, j]), axis=-1)
    path = m.cumsum(0)
    gaps = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
    common, relative = [], []
    out = np.zeros((len(m), 5, 2))
    means = np.zeros((1, 1, obs.shape[-1]))
    last = 0
    for start in range(0, len(m), 64):
        stop = min(start+64, len(m))
        # Only consume past origins whose last price is known at this block boundary.
        t = np.arange(last, start-h+1, 4)
        if len(t):
            last = t[-1]+4
            mask = (gaps[t+h+1]==gaps[t])[:, B]
            qty = mask.sum(2)
            good = (qty>=7).all(1)
            cash = (((path[t+h]-path[t])[:, B]*mask).sum(2)/qty.clip(1))/unit
            data = np.concatenate((obs[t], (cash-pred[t, :, j+1])[..., None]), axis=-1)[good]
            if len(data)>=6:
                avg = data.mean(1)
                # Center each block for fitting and use past block means to center new forecasts.
                means = .8*means+.2*avg[:, None, :-1].mean(0, keepdims=True) if common else avg[:, None, :-1].mean(0, keepdims=True)
                avg -= avg.mean(0)
                diff = (data-data.mean(1, keepdims=True)).reshape(-1, data.shape[-1])
                common.append(avg.T@avg/len(avg))
                relative.append(diff.T@diff/len(diff))
        x = obs[start:stop]
        out[start:stop, :, 0] = residual_update(common, x.mean(1, keepdims=True)-means)
        out[start:stop, :, 1] = residual_update(relative, x-x.mean(1, keepdims=True))
    return out


def signals(raw, grid, p, params):
    cut = 3+int(params[0])
    out = np.zeros((*p.shape[:2], 3, 2))
    for a, b, m, ok in event.sessions(raw, grid):
        views = basket_views(raw[a:b], ok, params[3:cut], params[cut:])
        for j in range(2):
            z = settle(p[a:b], m, ok, views, j, params[1+j])
            cols = np.stack((z[:, :, 0], z[:, :, 1], z.sum(2)), axis=-1)
            out[a:b, :, :, j] = np.repeat(cols, 10, axis=1)[:, np.argsort(B.ravel())]
    return out


def predict(raw, grid, p, params):
    return strip.evaluation.apply(sys.modules[__name__], raw, grid, p, params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
