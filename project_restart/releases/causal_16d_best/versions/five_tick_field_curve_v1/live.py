"""Update the market-state transition using only the book states that have already arrived."""

import numpy as np
import moments as data
import precision
from moments import market, tape, base


def model(st, mk):
    g = np.mean(mk, 0)
    s = np.mean(st, 0)
    n = (len(g) - 1) // 3
    sd = np.sqrt(g.diagonal()[:n]).clip(0.006)
    ss = np.sqrt(s.diagonal()[:n]).clip(0.03)
    ratio = sd / ss
    h = 0.5 * (g[:n, :n] + s[:n, :n] * np.outer(ratio, ratio)) / np.outer(sd, sd)
    val, u = np.linalg.eigh(h + 0.25 * np.eye(n))
    root = (u / np.sqrt(val)) / sd[:, None]
    lag = 0.5 * (
        g[:n, n : 2 * n]
        + g[:n, 2 * n : -1]
        + (s[:n, n : 2 * n] + s[:n, 2 * n : -1]) * np.outer(ratio, ratio)
    )
    op = root.T @ lag @ root
    eig, axis = np.linalg.eigh(0.5 * (op + op.T))
    reward = (
        0.5 * root.T @ (g[:n, -1] + s[:n, -1] * ratio * np.sqrt(g[-1, -1] / s[-1, -1]))
    )
    pick = np.argsort(abs(eig * (axis.T @ reward)))[-8:]
    z = root @ np.column_stack((reward, axis[:, pick]))
    scale = np.sqrt(np.diag(z.T @ g[:n, :n] @ z)).clip(0.001)
    z = z / scale
    gram = z.T @ g[:n, :n] @ z
    tc = z.T @ (g[:n, n : 2 * n] + g[:n, 2 * n : -1]) @ z / 2
    return z, gram, tc


def stat(q, y, x):
    mat = np.column_stack((q.mean(1, dtype=float), x, y.mean(1, dtype=float)))
    mat -= mat.mean(0)
    return mat.T @ mat / len(mat)


def fit(hist, j, h):
    c = np.stack(hist)
    g = c.mean(0)
    n = len(g) - 3
    idx = 5 + j
    ids = np.arange(5)
    v = np.zeros(n)
    v[idx] = 1.0
    hh = g[np.ix_(ids, ids)]
    v[ids] = -np.linalg.solve(
        hh + 0.05 * np.diag(hh.diagonal().clip(1e-8)), g[ids, idx]
    )
    slope = (c[:, h, n + h] / c[:, h, h]).clip(0.25, 4.0)
    grad = (c[:, :n, n + h] / slope[:, None] - c[:, :n, h]) @ v
    mu = grad.mean()
    mu *= abs(mu) / (abs(mu) + grad.std() / np.sqrt(len(c)) + 1e-7)
    w = v * mu / (2.0 * (v @ g[:n, :n] @ v) + 1e-8)
    risk = np.sqrt(w @ g[:n, :n] @ w / g[h, h])
    return w * min(1, 0.035 / max(risk, 1e-8))
