"""Estimate price increments in opening-price units using only earlier observations."""
from functools import lru_cache

import numpy as np

import y3_same_deadline_references as history

trial = history.trial


def price_path(raw):
    k, _, _ = trial.curve.settled.historical_geometry()
    r = raw[:, :, 0].astype(float)-float(np.float16(.00136))
    log = np.log1p(k*r).cumsum(0)
    price = np.expm1(log)/k
    return price, np.diff(price, axis=0, prepend=np.zeros_like(price[:1]))


def estimate(events, fallback):
    v = np.abs(events[events!=0])
    if len(v)<12:
        return fallback
    unit = np.quantile(v, .08)
    if unit<=0:
        return fallback
    # Check whether smaller observed jumps lie near integer multiples of the proposed unit.
    # The loss uses relative distance to integer multiples, without rewarding arbitrarily tiny units.
    v = v[v<=8.5*unit]
    if len(v)<10:
        return fallback
    for _ in range(3):
        step = np.rint(v/unit)
        ok = (step>=1)&(abs(v/unit-step)<.22)
        if ok.sum()<5:
            break
        unit = np.median(v[ok]/step[ok])
    return max(unit, .001)


@lru_cache(maxsize=1)
def prior():
    rows = []
    for d in range(4):
        _, move = price_path(trial.clock.book.rawday(d))
        rows.append([estimate(v, .15) for v in move.T])
    return np.median(rows, axis=0)


def snapshots(raw, block=120):
    price, moves = price_path(raw)
    starts = np.arange(0, len(raw), block)
    units = []
    last = prior()
    for start in starts:
        if start:
            last = np.array([estimate(v, last[s]) for s,v in enumerate(moves[max(0,start-720):start].T)])
        units.append(last)
    return price, moves, starts, np.stack(units)
