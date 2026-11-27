"""Compare stock payoff horizon pooling with horizon weights learned from market response on independent dates."""

import numpy as np
import clock
from clock import market, tape, base, nonlinear, precision


def fit(hist, j, h, n_controls=6):
    c = np.stack(hist)
    g = c.mean(0)
    n = len(g) - 3
    v = np.zeros(n)
    v[n_controls + j] = 1
    ids = np.arange(n_controls)
    hh = g[np.ix_(ids, ids)]
    # Retain earlier forecasts and the existing clock-specific change as controls.
    v[ids] = -np.linalg.solve(
        hh + 0.045 * np.diag(hh.diagonal().clip(1e-8)), g[ids, n_controls + j]
    )
    slope = (c[:, h, n + h] / c[:, h, h]).clip(0.25, 4.0)
    gradient = (c[:, :n, n + h] / slope[:, None] - c[:, :n, h]) @ v
    rhs = gradient.mean()
    rhs *= abs(rhs) / (abs(rhs) + gradient.std() / np.sqrt(len(c)) + 1e-8)
    w = v * rhs / (1.9 * (v @ g[:n, :n] @ v) + 1e-8)
    risk = np.sqrt(w @ g[:n, :n] @ w / g[h, h])
    return w * min(1, 0.045 / max(risk, 1e-8))
