"""Test whether settled forecast surprises persist differently across individual stocks."""
from functools import lru_cache
import sys

import numpy as np
import torch

import prepare_dense_market_unpriced_reference_v1 as ref
from experiment_dense_market_unpriced_covariance_v1 import BLOCK, event, onset, rel, strip, unit, official

REFERENCE = ref.OUT
OUT = REFERENCE.parent / 'dense_stock_surprise_response_v1'
ARMS = ('relative_cash_persistence', 'response_to_peer_surprises', 'relative_forecast_reversal')


def read_p(day):
    p = ref.read_p(day)
    if 8 <= day < 16:
        p = p.copy()
        p[..., 0] += official.short_market_changes()[day][:, None]
    return p


def paths(move, ok, p, scale):
    path = move.cumsum(0)
    missing = (~ok).cumsum(0)
    moments = np.zeros((50, 5))
    for start in range(BLOCK, len(p), BLOCK):
        end = min(start + BLOCK, len(p))
        t = np.arange(start-BLOCK+5, start, 5)
        valid = (missing[t] == missing[t-5]) & ok[t-5]
        cash = path[t] - path[t-5]
        issued = p[t-5, :, 0] * scale
        surprise = cash-issued
        surprise -= (surprise*valid).sum(0) / valid.sum(0).clip(1)
        surprise *= valid
        peer = (surprise.sum(1, keepdims=True)-surprise)/(valid.sum(1, keepdims=True)-valid).clip(1)
        residual = (surprise-peer)*valid
        forecast = (issued-issued.mean(1, keepdims=True))*valid
        forecast -= (forecast.sum(0)/valid.sum(0).clip(1))*valid
        block = np.zeros_like(moments)
        block[:, 0] = (residual**2).mean(0)
        block[:, 1] = (peer**2).mean(0)
        for k in range(1, 7):
            taper = (1-k/12)*(1-k/7)
            block[:, 2] += taper*(residual[:-k]*residual[k:]).mean(0)
            block[:, 3] += taper*(peer[:-k]*residual[k:]).mean(0)
            block[:, 4] -= taper*(forecast[:-k]*residual[k:]).mean(0)
        moments = .77*moments + block
        if start >= 2*BLOCK:
            denom = moments[:, 0] + .5*np.median(moments[:, 0]) + 1e-6
            peer_var = moments[:, 1] + .5*np.median(moments[:, 1]) + 1e-6
            yield start, end, moments[:, 2:] / np.stack((denom, peer_var, denom), axis=1)


@lru_cache(maxsize=48)
def record(day, scale):
    raw, grid = [onset.settled.load(day, key) for key in ('x', 'time')]
    p = read_p(day)
    sample = [z for a,b,move,ok in event.sessions(raw,grid)
              for _,_,z in paths(move,ok,p[a:b],scale)]
    return np.median(sample, axis=0)


def train(end):
    scale = float(np.median([unit(day) for day in range(4, end)]))
    typical = np.median([record(day, scale) for day in range(4, end)], axis=0)
    return np.r_[scale, typical.ravel()]


def signals(raw, grid, p, bank):
    out = np.zeros((*p.shape[:2], 3, 2))
    typical = bank[1:].reshape(50, 3)
    for a,b,move,ok in event.sessions(raw,grid):
        for lo,hi,observed in paths(move,ok,p[a:b],bank[0]):
            f = p[a+lo:a+hi, :, 1:].astype(float)
            common = f.mean(1, keepdims=True)
            local = f-common
            coeff = .35*np.tanh(observed-typical)
            coeff -= coeff.mean(0)
            delta = np.stack((local*coeff[None,:,0,None], common*coeff[None,:,1,None],
                              local*coeff[None,:,2,None]), axis=2)
            out[a+lo:a+hi] = delta-delta.mean(1, keepdims=True)
    return out


def predict(raw, grid, p, params):
    return strip.evaluation.apply(sys.modules[__name__], raw, grid, p, params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
