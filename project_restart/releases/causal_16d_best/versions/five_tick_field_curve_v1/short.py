"""Estimate shared market moves from the original forecasts using contemporaneous sender activity."""

import json
import numpy as np
import settings as base
from settings import TAPE
from book import codec, decode, history, sessions, metric, gain

OUT = base.ROOT / "artifacts/dense_count_weighted_quotes_v1"
ARMS = ("market", "receiver")


def signals(raw, grid, p, code, include_receiver=True):
    for a, b in sessions(grid):
        s = raw[a:b]
        c = np.rint(decode(s[:, :, 77:83], code)).clip(0, 512)
        left, right, u, v, e, f = np.moveaxis(c, 2, 0)
        total = left + right
        inc = (left - u).clip(0) + (right - v).clip(0)
        # Transform each measure of activity into a sender reliability weight.
        measures = np.stack(
            (
                np.log1p(total),
                np.log1p(inc),
                np.log1p(e),
                (1 + u + v) / (1 + total),
                (1 + f) / (1 + e),
            ),
            2,
        ).clip(0, 8)
        slow = history(measures, 80)

        def normalize(x):
            delta = x - x.mean(1, keepdims=True)
            return np.tanh(
                delta / (0.2 + np.sqrt(np.mean(delta**2, axis=1, keepdims=True)))
            )

        weights = np.concatenate((normalize(measures), normalize(measures - slow)), 2)
        weights -= weights.mean(1, keepdims=True)
        pred = p[a:b].astype(float)
        pred -= pred.mean(1, keepdims=True)
        # Covariance between direction and activity is exactly the difference from a uniform basket.
        votes = np.einsum("tsh,tsm->thm", pred, weights) / 50
        votes = votes.reshape(len(s), -1)
        prev = history(votes, 16)
        shared = np.concatenate((votes - prev, prev), 1)
        if not include_receiver:
            yield a, b, shared, None
            continue
        # Except for the harmless normalization constant, receiver readouts multiply a
        # leave-one-stock-out vote by that receiver's opposite participation.
        # For the smoothed vote, likewise subtract the receiver's own smoothed contribution.
        local = ((pred[:, :, :, None] * weights[:, :, None, :]) / 50).reshape(
            len(s), 50, -1
        )
        old = history(local, 16)
        rest = (shared[:, None] - np.concatenate((local - old, old), 2)) * (50 / 49)
        clock = weights[:, :, [0, 8]]
        receiver = np.concatenate((rest * clock[:, :, :1], rest * clock[:, :, 1:2]), 2)
        receiver -= receiver.mean(1, keepdims=True)
        yield a, b, shared, receiver.astype("float32")


def record(raw, grid, p, y, code):
    per = []
    for a, b, shared, receiver in signals(raw, grid, p, code):
        pred = p[a:b].astype(float)
        truth = y[a:b].astype(float)
        mean, actual = pred.mean(1), truth.mean(1)
        market = np.column_stack((mean, shared, actual))
        market -= market.mean(0)
        t = np.arange(0, b - a, 3)
        names = np.concatenate(
            (pred[t] - mean[t, None], receiver[t], truth[t] - actual[t, None]), 2
        )
        names = names.reshape(-1, names.shape[2])
        per.append((b - a, market.T @ market / (b - a), names.T @ names / len(names)))
    return [sum(r[0] * r[i] for r in per) / len(p) for i in (1, 2)]


def train(hist, side):
    c = np.stack([g[side] for g in hist])
    g = c.mean(0)
    n = len(g) - 3
    proj = np.eye(n)[:, 3:]

    def reg(m, lam):
        return m + lam * np.diag(m.diagonal().clip(1e-8))

    proj[:3] = -np.linalg.solve(reg(g[:3, :3], 0.035), g[:3, 3:n])
    h = proj.T @ g[:n, :n] @ proj
    slopes = np.stack(
        [(c[:, j, n + j] / c[:, j, j]).clip(0.25, 4) for j in range(3)], 1
    )
    v = proj.T @ (c[:, :n, n:] / slopes[:, None] - c[:, :n, :3])
    rhs = v.mean(0)
    rhs *= np.abs(rhs) / (np.abs(rhs) + v.std(0) / np.sqrt(len(c)) + 1e-6)
    w = np.linalg.solve(reg(h, 1.2), rhs)
    risk = np.sqrt((w * (h @ w)).sum(0).clip(0) / g.diagonal()[:3])
    return proj @ (w * np.minimum(1.0, 0.055 / risk.clip(1e-8)))


def apply(raw, grid, p, code, weights):
    out = {key: np.zeros_like(p) for key in weights}
    for a, b, shared, receiver in signals(
        raw, grid, p, code, include_receiver="receiver" in weights
    ):
        m = p[a:b].mean(1, dtype=float)
        for key, w in weights.items():
            if key == "market":
                out[key][a:b] = (np.column_stack((m, shared)) @ w)[:, None]
            else:
                out[key][a:b] = np.concatenate((p[a:b] - m[:, None], receiver), 2) @ w
    return out


def main():
    OUT.mkdir(exist_ok=True)
    code = codec()
    hist, results = [], {(key, j): [] for key in ARMS for j in range(3)}
    for d in range(4, 16):
        raw = np.load(base.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        grid = np.load(base.CACHE / f"day_{d:02d}_time.npy").astype(int)
        p = np.load(TAPE / f"forecast_{d:02d}.npy")
        y = np.load(base.CACHE / f"day_{d:02d}_y.npy")
        if d in (8, 12):
            w = {key: train(hist, i) for i, key in enumerate(ARMS)}
        if d >= 8:
            changes = apply(raw, grid, p, code, w)
            for (key, j), rows in results.items():
                rows.append(metric(p[..., j], y[..., j], changes[key][..., j]))
            if d == 8:
                short = apply(raw[:193], grid[:193], p[:193], code, w)
                for key in short:
                    np.testing.assert_allclose(
                        short[key], changes[key][:193], atol=1e-7, rtol=0
                    )
        hist.append(record(raw, grid, p, y, code))
    report, weights, which = {}, {}, {}
    for (key, j), rows in results.items():
        folds = [gain(sum(s)) for s in (rows[:4], rows[4:])]
        days = [gain(s) for s in rows]
        report[f"{key}_{j}"] = dict(folds=folds, days=days)
        if min(folds) > 0 and sum(v > 0 for v in days) >= 5:
            which.setdefault(key, []).append(j)
    for key, heads in which.items():
        w = train(hist, ARMS.index(key))
        w[:, [i for i in range(3) if i not in heads]] = 0
        weights[key] = w
    (OUT / "rolling.json").write_text(
        json.dumps(dict(results=report, selected=which), indent=2)
    )
    print({k: v["folds"] for k, v in report.items()}, which, flush=True)
    if not weights:
        return
    np.savez(OUT / "weights.npz", code=code, **weights)
