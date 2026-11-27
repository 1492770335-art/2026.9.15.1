"""Measure how quickly each book feature decorrelates in market and individual-stock units."""

import numpy as np
from moments import base, market

parent = market.parent
OUT = base.ROOT / "artifacts/dense_cross_scale_clocks_v1"
LAGS = (0, 5, 10, 20, 40, 80)
HORIZONS = (5, 10, 20, 40, 60)


def daily(raw, grid, y, code):
    x, av = parent.state(raw, grid, code, with_market=True)
    m = market.market_state(av, grid)
    n = x.shape[2]
    shape = (len(LAGS), n, n)
    st = np.zeros(shape)
    mk = np.zeros(shape)
    sy = []
    my = []
    sv = []
    mv = []
    ns = []
    for a, b in parent.sessions(grid):
        t = np.arange(a, b - 80, 8)
        if not len(t):
            continue
        xc = x[t].reshape(-1, n).astype(float)
        mc = m[t]
        for i, h in enumerate(LAGS):
            st[i] += xc.T @ x[t + h].reshape(-1, n).astype(float) / 50
            mk[i] += mc.T @ m[t + h]
        five = base.UNITS[0]
        twenty = base.UNITS[1]
        v = np.stack(
            (
                five * y[t, :, 0],
                five * (y[t, :, 0] + y[t + 5, :, 0]),
                twenty * y[t, :, 1],
                twenty * (y[t, :, 1] + y[t + 20, :, 1]),
                y[t, :, 2],
            ),
            2,
        ).astype(float)
        avg = v.mean(1)
        rel = v - avg[:, None]
        sy.append(xc.T @ rel.reshape(-1, len(HORIZONS)) / 50)
        my.append(mc.T @ avg)
        sv.append(np.square(rel).sum((0, 1)) / 50)
        mv.append(np.square(avg).sum(0))
        ns.append(len(t))
    n = sum(ns)
    return tuple(z / n for z in (st, mk, sum(sy), sum(my), sum(sv), sum(mv)))


def main():
    OUT.mkdir(exist_ok=True)
    code = parent.codec()
    for d in range(16):
        load = lambda k: np.load(base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")
        result = daily(load("x"), load("time").astype(int), load("y"), code)
        np.savez(
            OUT / f"day_{d:02d}.npz",
            **dict(
                zip(
                    (
                        "stock",
                        "market",
                        "stock_pay",
                        "market_pay",
                        "stock_var",
                        "market_var",
                    ),
                    result,
                )
            ),
        )
    print("wrote disjoint-date lag and horizon moments", flush=True)
