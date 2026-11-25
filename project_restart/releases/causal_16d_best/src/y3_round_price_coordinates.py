"""Infer the current integer price coordinate from earlier quantized relative returns."""
from functools import lru_cache
import warnings

import numpy as np

import y3_observed_price_grid as grid

trial = grid.trial


def observed_steps(raw, k):
    flat = float(np.float16(.00136))
    r = raw[:, :, 0].astype(float)-flat
    before = np.exp(np.vstack((np.zeros(50), np.log1p(k*r).cumsum(0)[:-1])))
    move = r*before
    step = np.array([grid.estimate(v, .15) for v in move.T])
    ratio = move/step
    counts = np.rint(ratio).astype(int)
    prefix = counts.cumsum(0)-counts
    good = (counts!=0)&(abs(ratio-counts)<.06)
    # The f000 rounding interval grows with the magnitude of the encoded value.
    width = abs(np.spacing(raw[:, :, 0]).astype(float))
    lo = (r-.5*width)
    hi = (r+.5*width)
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = counts/lo, counts/hi
    return counts, prefix, good, np.minimum(a,b), np.maximum(a,b)


def estimate_k():
    initial = grid.trial.curve.settled.historical_geometry()[0]
    moments = []
    for d in range(4):
        n, prefix, good, lo, hi = observed_steps(grid.trial.clock.book.rawday(d), initial)
        for s in range(50):
            ok = good[:, s]
            x = prefix[ok, s].astype(float)
            y = .5*(hi[ok,s]+lo[ok,s])
            x -= x.mean()
            y -= np.median(y-initial*x)
            mask = abs(y-initial*x)<.007
            x, y = x[mask], y[mask]
            x -= x.mean()
            moments.append((x@x, x@(y-y.mean())))
    return sum(r for _,r in moments)/sum(g for g,_ in moments)


def initial_price(prefix, good, lo, hi, k):
    # Aggregate the float16 rounding constraints with symmetric trimming for anomalous events.
    low = np.where(good, lo/k-prefix, np.nan)
    high = np.where(good, hi/k-prefix, np.nan)
    return .5*(np.nanquantile(low,.92,axis=0)+np.nanquantile(high,.08,axis=0))


@lru_cache(maxsize=1)
def calibration():
    rough = estimate_k()
    values = []
    for day in range(4):
        raw = trial.clock.book.rawday(day)
        _, x, good, lo, hi = observed_steps(raw, rough)
        values.extend(initial_price(x, good, lo, hi, rough)*rough)
    candidates = rough+np.linspace(-1.25e-6, 1.25e-6, 2001)
    loss = np.cos(2*np.pi*np.array(values)[:, None]/candidates).mean(0)
    return float(candidates[np.argmax(loss)])


def coordinates(raw, k=None):
    if k is None:
        k = calibration()
    ret = raw[:, :, 0].astype(float)-float(np.float16(.00136))
    log = np.log1p(k*ret).cumsum(0)
    level = np.zeros_like(log, dtype=int)
    valid = np.zeros_like(log, dtype=bool)
    for start in range(120, len(raw), 120):
        source = raw[max(0,start-960):start]
        counts, x, good, low, high = observed_steps(source, k)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            opening = initial_price(x, good, low, high, k)
        projected = np.rint(np.nan_to_num(opening))+counts.sum(0)
        ok = (good.sum(0)>=18)&(abs(opening-np.rint(opening))<.32)&(projected>10)
        period = slice(start, start+120)
        level[period] = np.rint(projected*np.exp(log[period]-log[start-1]))
        valid[period] = ok
    return level, valid
