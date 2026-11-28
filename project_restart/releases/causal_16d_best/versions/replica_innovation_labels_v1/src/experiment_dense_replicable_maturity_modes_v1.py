"""Estimate distal price responses along maturity profiles reproducible across training dates."""
from functools import lru_cache
import sys
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, official, rel, strip
import prepare_dense_settled_pearson_tail_reference_v1 as ref

REFERENCE = ref.OUT
OUT = REFERENCE.parent / 'dense_replicable_maturity_modes_v1'
TARGETS = (2,)
ARMS = ('unrestricted_distal', 'repeatable_maturity_profile', 'repeatable_distal_only')
N_BLOCKS = 16
HISTORY = 20


def read_p(day):
    p = ref.read_p(day)
    if 8 <= day < 16:
        p = p.copy()
        p[..., 0] += official.short_market_changes()[day][:, None]
    return p


def matrix(raw, p, live, t):
    now = np.tanh(.65 * np.arcsinh(raw[t].astype(float)))
    before = np.maximum(0, t - HISTORY)
    lag = np.tanh(.65 * np.arcsinh(raw[before].astype(float)))
    innovation = (now - lag) * (live[before] & (t >= HISTORY)[:, None])[..., None]
    x = np.concatenate((p[t], now, innovation), axis=-1)
    ok = live[t]
    mean = (x * ok[..., None]).sum(1) / ok.sum(1).clip(1)[:, None]
    return (x - mean[:, None]) * ok[..., None]


@lru_cache(maxsize=16)
def record(day):
    raw, grid = [onset.settled.load(day, k) for k in ('x', 'time')]
    p = read_p(day)
    gram = cross = 0.
    count = 0
    for a, b, m, ok in event.sessions(raw, grid):
        path = m.cumsum(0)
        gap = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
        ticks = np.arange(HISTORY + day % 11, b - a - 5 * N_BLOCKS, 11)
        for t in np.array_split(ticks, max(1, (len(ticks) + 95) // 96)):
            x = matrix(raw[a:b], p[a:b], ok, t)
            times = t[:, None] + 5 * np.arange(N_BLOCKS + 1)
            moves = np.diff(path[times], axis=1).transpose(0, 2, 1)
            valid = gap[times[:, -1] + 1] == gap[t]
            mbar = (moves * valid[..., None]).sum(1) / valid.sum(1).clip(1)[:, None]
            x, y = x[valid], (moves - mbar[:, None])[valid]
            gram += x.T @ x
            cross += x.T @ y
            count += len(x)
    return gram / count, cross / count


def reproducing(curves, distal):
    # Estimate signal energy in maturity space from independent calendar groups.
    # Same-date products contain the sampling variance even when the true response is zero.
    cut = 4 if distal else 0
    c = curves[:, :, cut:]
    a, b = c[::2].mean(0), c[1::2].mean(0)
    cross = (a.T @ b + b.T @ a) / 2
    noise = (a - b).T @ (a - b) / 4
    eig, v = np.linalg.eigh(cross)
    err = (v * (noise @ v)).sum(0)
    rel = eig.clip(0) / (eig.clip(0) + 2 * err.clip(0) + 1e-12)
    weight = ((np.arange(cut, N_BLOCKS) >= 4) & (np.arange(cut, N_BLOCKS) < 12)).astype(float)
    return v @ (rel * (v.T @ weight))


def train(end):
    grams, crosses = map(np.stack, zip(*(record(d) for d in range(4, end))))
    g = grams.mean(0)
    # Fit in the complement of the incumbent's three current forecasts.
    adj = np.linalg.solve(g[:3, :3] + np.eye(3) * 1e-8, g[:3, 3:])
    projection = np.vstack((-adj, np.eye(len(g) - 3)))
    cov = projection.T @ g @ projection
    sd = cov.diagonal().clip(1e-6) ** .5
    eig, v = np.linalg.eigh(cov / np.outer(sd, sd))
    basis = projection @ (v / (sd[:, None] * np.sqrt(eig.clip(0) + .7)))
    curves = np.einsum('if,dib->dfb', basis, crosses)
    directions = [curves[:, :, 4:12].sum(2), curves @ reproducing(curves, False),
                  curves[:, :, 4:] @ reproducing(curves, True)]
    weights = []
    for daily in directions:
        mean = daily.mean(0)
        variance = daily.var(0) / max(1, len(daily) - 1)
        weights.append(basis @ (mean * mean**2 / (mean**2 + 2 * variance + 1e-10)))
    return np.stack(weights, axis=1)


def signals(raw, grid, p, bank):
    out = np.zeros((*p.shape[:2], len(ARMS), 1))
    for a, b, _, ok in event.sessions(raw, grid):
        for start in range(0, b-a, 64):
            t = np.arange(start, min(start + 64, b-a))
            out[a+t, :, :, 0] = matrix(raw[a:b], p[a:b], ok, t) @ bank
    return out


def predict(raw, grid, p, params):
    return strip.evaluation.apply(sys.modules[__name__], raw, grid, p, params)


if __name__ == '__main__':
    strip.evaluation.rolling(sys.modules[__name__])
