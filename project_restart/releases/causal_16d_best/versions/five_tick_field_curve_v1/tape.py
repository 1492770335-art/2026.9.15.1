"""Chronological reference for the stock-regularized common-market Y3 readout."""

import numpy as np
import market

parent = market.parent
base = market.base
OUT = base.ROOT / "artifacts/dense_market_reliability_tape_v1"


def main():
    OUT.mkdir(exist_ok=True)
    code = parent.codec()
    stock = []
    common = []
    sfit = []
    mfit = []
    for d in range(16):
        load = lambda k: np.load(base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")
        raw, grid, y = load("x"), load("time").astype(int), load("y")
        x, av = parent.state(raw, grid, code, with_market=True)
        m = market.market_state(av, grid)
        if d >= 4:
            if d % 4 == 0:
                sb = parent.components(stock)
                mb = market.operators(stock, common)
            p = np.load(market.TAPE / f"forecast_{d:02d}.npy")
            means = p.mean(1, dtype=float)
            z = x @ sb
            cm = m @ mb
            if d >= 6:
                if d == 6 or d % 4 == 0:
                    pw = parent.fit(sfit, 0)
                    mw = market.fit(mfit, 2)
                change = np.column_stack((means, cm)) @ mw
                old = p.copy()
                old[:, :, 2] += (
                    np.concatenate((p - p.mean(1, keepdims=True), z), 2) @ pw
                )
                old[:, :, 2] += change[:, None]
                np.savez(
                    OUT / f"day_{d:02d}.npz",
                    reference=old,
                    change=change,
                    directions=cm,
                )
            sfit.append(parent.stats(p, y, z))
            mfit.append(market.stats(p, y, cm))
        stock.append(parent.moment(x, grid, y))
        common.append(market.record(m, grid, y))
    with np.load(market.OUT / "weights.npz") as f:
        np.testing.assert_array_equal(market.operators(stock, common), f["bank"])
        np.testing.assert_array_equal(market.fit(mfit, 2), f["w"])
    print("development tape rebuilt and final market coefficients verified", flush=True)
