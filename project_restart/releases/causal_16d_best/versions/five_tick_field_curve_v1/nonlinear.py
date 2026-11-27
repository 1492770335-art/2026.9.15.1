"""Transfer nonlinear payoff responses of slow book modes from individual names to the market."""

import json
import numpy as np
import live
from live import market, tape, base, precision

parent = market.parent
OUT = base.ROOT / "artifacts/dense_nonlinear_payoff_prior_v1"
ARMS = ("magnitude", "interactions", "market_residual")
ROOT = live.data.OUT


def expansion(u):
    u = u.clip(-4.0, 4.0)
    # Linear terms permit an exact within-training projection of the nonlinear increment.
    amplitude = np.tanh(np.abs(u) - 1)
    main = np.concatenate((u * amplitude, np.tanh(u * u - 1)), axis=-1)
    product = []
    for i in range(1, u.shape[-1]):
        product.extend(
            (
                u[..., 0] * amplitude[..., i],
                u[..., i] * amplitude[..., 0],
                u[..., 0] * u[..., i],
            )
        )
    return np.concatenate((u, main, np.stack(product, axis=-1)), axis=-1)


def sample(day, mp, sp, ss):
    load = lambda k: np.load(base.CACHE / f"day_{day:02d}_{k}.npy", mmap_mode="r")
    raw, grid, y = load("x"), load("time").astype(int), load("y")
    x, av = parent.state(raw, grid, parent.codec(), with_market=True)
    m = market.market_state(av, grid)
    gm = gs = 0.0
    nt = 0
    for a, b in parent.sessions(grid):
        t = np.arange(a + day % 5, b - 60, 5)
        # Each row of this offline target represents a disjoint 20-tick part of Y3.
        ys = base.UNITS[1] * np.stack([y[t + h, :, 1] for h in (0, 20, 40)], 2).astype(
            float
        )
        ym = ys.mean(1)
        z = x[a:b] @ sp
        z = z - parent.smooth(z, 320)
        u = z[t - a]
        stock = expansion(u)
        labels = (ys - ym[:, None, :]) * ss
        smat = np.concatenate((stock, labels), 2).reshape(-1, stock.shape[2] + 3)
        mm = expansion(m[t] @ mp)
        mmat = np.column_stack((mm, ym))
        gs += smat.T @ smat / 50
        gm += mmat.T @ mmat
        nt += len(t)
    return gs / nt, gm / nt


def train(end, st, mk):
    mp = live.model(st, mk)[0][:, :6]
    s = np.mean(st, 0)
    g = np.mean(mk, 0)
    n = (len(g) - 1) // 3
    ratio = np.sqrt(g.diagonal()[:n]).clip(0.006) / np.sqrt(s.diagonal()[:n]).clip(0.03)
    sp = mp * ratio[:, None]
    scale = np.sqrt(g[-1, -1] / s[-1, -1])
    rows = [sample(d, mp, sp, scale) for d in range(end)]
    stock, mkt = [np.stack(s) for s in zip(*rows)]
    n = len(mkt[0]) - 3
    linear = mp.shape[1]
    first = 3 * linear

    def weights(hist, col):
        g = hist.mean(0)
        start, stop = col
        ids = np.arange(start)
        cols = np.arange(start, stop)
        proj = np.eye(n)[:, cols]
        h = g[np.ix_(ids, ids)]
        proj[ids] = -np.linalg.solve(
            h + 0.09 * np.diag(h.diagonal().clip(1e-5)), g[np.ix_(ids, cols)]
        )
        hg = proj.T @ g[:n, :n] @ proj
        # Training gradients use only the second and third segments of the 60-tick horizon.
        daily = hist[:, :n, -2:].sum(2) @ proj
        rhs = daily.mean(0)
        rhs *= abs(rhs) / (abs(rhs) + daily.std(0) / np.sqrt(len(daily)) + 1e-7)
        return proj @ np.linalg.solve(hg + 0.8 * np.diag(hg.diagonal().clip(1e-5)), rhs)

    pooled = 0.55 * stock + 0.45 * mkt
    a = weights(pooled, (linear, first))
    b = weights(pooled, (first, n))
    c = weights(mkt, (linear, n)) - weights(stock, (linear, n))
    return mp, np.column_stack((a, b, c))


def run():
    OUT.mkdir(exist_ok=True)
    signals = {}
    st = []
    mk = []
    hist = []
    rows = {(k, h): [] for k in ARMS for h in (1, 2)}
    for d in range(16):
        with np.load(ROOT / f"day_{d:02d}.npz") as f:
            s, g, m = f["stock"], f["market"], f["state"]
        if d >= 6:
            if d in (6, 8, 12):
                mp, bank = train(d, st, mk)
                print("fitted nonlinear prior through day", d, flush=True)
            with np.load(tape.OUT / f"day_{d:02d}.npz") as f:
                q = f["reference"]
                dm = f["change"]
                fixed = f["directions"][:, 2]
            if d >= 8:
                q += np.load(precision.OUT / f"oof_{d:02d}.npy")[:, None]
            y = np.load(base.CACHE / f"day_{d:02d}_y.npy", mmap_mode="r")
            z = expansion(m @ mp) @ bank
            xx = np.column_stack((dm, fixed, z))
            if d in (8, 12):
                w = {key: live.fit(hist, ARMS.index(key[0]), key[1]) for key in rows}
            if d >= 8:
                mat = np.column_stack((q.mean(1, dtype=float), xx))
                for (k, h), v in rows.items():
                    dx = mat @ w[k, h]
                    signals[d, k, h] = dx.astype("float32")
                    delta = np.broadcast_to(dx[:, None], q.shape[:2])
                    v.append(market.metric(q[..., h], y[..., h], delta))
            hist.append(live.stat(q, y, xx))
        st.append(s)
        mk.append(g)
    report = {
        f"{k}_{h}": dict(
            folds=[market.gain(sum(v[:4])), market.gain(sum(v[4:]))],
            days=[market.gain(g) for g in v],
        )
        for (k, h), v in rows.items()
    }
    picks = {}
    for h in (1, 2):
        ok = [
            k
            for k in ARMS
            if min(report[f"{k}_{h}"]["folds"]) > 0
            and sum(z > 0 for z in report[f"{k}_{h}"]["days"]) >= 5
        ]
        if ok:
            picks[str(h)] = max(ok, key=lambda k: min(report[f"{k}_{h}"]["folds"]))
    (OUT / "rolling.json").write_text(
        json.dumps(dict(selected=picks, results=report), indent=2)
    )
    print(picks, {k: v["folds"] for k, v in report.items()}, flush=True)
    if not picks:
        return
    for d in range(8, 16):
        n = next(len(a) for (day, _, _), a in signals.items() if day == d)
        adj = np.zeros((n, 3), dtype="float32")
        for h, k in picks.items():
            adj[:, int(h)] = signals[d, k, int(h)]
        np.save(OUT / f"oof_{d:02d}.npy", adj)
    w = np.zeros((8, 3))
    for h, k in picks.items():
        w[:, int(h)] = live.fit(hist, ARMS.index(k), int(h))
    mp, bank = train(16, st, mk)
    np.savez(OUT / "weights.npz", mp=mp, bank=bank, w=w)
