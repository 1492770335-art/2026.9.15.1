"""Estimate the sign of forecast revisions from already matured returns in each session."""

import json


import numpy as np
from scipy.signal import lfilter
import experiment_y3_dense_forecast_vintages_v1 as v

OUT = v.OUT.parent / "y3_intraday_vintage_orientation_v1"


def design(raw, q, t, geom, w):
    pieces = []
    k, co, row = geom
    for a, b in v.price.segments(len(raw)):
        ii = t[(t >= a) & (t < b)] - a
        if not len(ii):
            continue
        ticks = np.arange(0, b - a, 4)
        # Calculate revisions on the original session grid, including exact current query ticks.
        u = np.union1d(ticks, ii)
        x = v.features(raw[a:b], q[a:b], u, geom)
        x = x.reshape(len(u), 50, len(v.LAGS), 4)
        old_model = x.reshape(len(u), 50, -1) @ w
        basis = np.stack(
            (
                old_model,
                x[:, :, :3, 0].mean(2),
                x[:, :, 3:, 0].mean(2),
                x[:, :, :3, 1].mean(2),
                x[:, :, 3:, 1].mean(2),
                (x[:, :, :, 0] - x[:, :, :, 2]).mean(2),
            ),
            axis=2,
        )
        z = basis[np.searchsorted(u, ticks)]
        endpoints = v.feedback.cash.compound(raw[a:b], k)
        avail = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(raw[a:b], row)).cumsum(0)))
        idx = np.arange(len(ticks))
        issue = (idx - 15).clip(0)
        begin = ticks[issue]
        valid = (ticks >= 60)[:, None] & (avail[ticks + 1] == avail[begin])
        actual = co[2, 0] + co[2, 1] * np.expm1(endpoints[ticks + 1] - endpoints[begin + 1]) / k
        forecast = q[a + begin, :, 2]
        count = valid.sum(1).clip(1)

        def avg(x):
            return (x * valid[:, :, None]).sum(1) / count[:, None]

        def center(x):
            return x - avg(x)[:, None, :]

        zold = center(z[issue])
        pp = center(forecast[:, :, None])
        yy = center(actual[:, :, None])
        g = np.column_stack((avg(pp * pp), avg(pp * yy), avg(zold * pp), avg(zold * yy), avg(zold * zold)))

        def slopes(half, mask):
            s = 1 / half
            mass = lfilter([s], [1, s - 1], mask.astype(float))
            gbar = lfilter([s], [1, s - 1], g * mask[:, None], axis=0) / (mass[:, None] + 0.00001)
            vp, py = gbar[:, 0], gbar[:, 1]
            zp, zy, zz = np.split(gbar[:, 2:], 3, axis=1)
            proj = zy - zp * (py / (vp + 0.0001)).clip(0, 3)[:, None]
            floor = np.r_[0.12 * np.mean(w * w) + 0.000002, np.full(5, 0.014)]
            return (0.35 * proj / (zz + floor) * mass[:, None] / (0.35 + mass[:, None])).clip(-2, 2)

        def make(half):
            observed = count >= 32
            pooled = slopes(half, observed)
            even = slopes(half, observed & ((idx // 20) % 2 == 0))
            odd = slopes(half, observed & ((idx // 20) % 2 == 1))
            stable = np.where(even * odd > 0, np.sign(even) * np.minimum(abs(even), abs(odd)), 0)
            return pooled, stable

        now = basis[np.searchsorted(u, ii)]
        where = ii // 4
        out = []
        for half in (60, 180):
            pooled, stable = make(half)
            for weight in (pooled, stable):
                signal = now * weight[where, None, :]
                out.extend((signal[:, :, 0], signal[:, :, 1:].mean(2)))
        out.append(np.roll(out[1], 17, axis=1))
        pieces.append(np.stack(out, axis=2).astype(np.float32))
    return np.concatenate(pieces)


def train():
    history, cal, blocks = [], [], []
    geom = v.feedback.geometry()
    from experiment_causal_cross_horizon_ridge_v1 import labels

    for d in range(4, 16):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        y = labels(d).reshape(-1, 50, 3)
        with np.load(v.TAPE / f"day_{d:02d}.npz") as f:
            q = np.concatenate((f["short"], f["dense"][:, :, None]), axis=2)
            t = f["t"]
        if d >= 6:
            if d % 2 == 0:
                w = v.fit(history)
            z = design(raw, q, t, geom, w)
            if d >= 8:
                if d % 2 == 0:
                    size = v.price.sizing(cal)
                blocks.append(v.price.rscore(q[t], y[t], z * size))
                if d == 8:
                    cut = 2503
                    use = t < cut
                    np.testing.assert_allclose(
                        z[use], design(raw[:cut], q[:cut], t[use], geom, w), atol=3e-7, rtol=0
                    )
            cal.append(v.price.rscore(q[t], y[t], z))
        valid = ((t < 2401 - 60) | (t >= 2401)) & (t < len(q) - 60)
        x = v.features(raw, q, t[valid], geom)
        history.append(v.moments(x, q[t[valid], :, 2], y[t[valid], :, 2]))
    half = np.stack([v.price.diff(sum(g))[1:] for g in (blocks[:4], blocks[4:])])
    daily = np.stack([v.price.diff(g)[1:] for g in blocks])
    first = int(np.argmax(half[0, :8]))
    pick = (
        first
        if (half[:, first] > np.maximum(0, half[:, 8])).all() and (daily[:, first] > 0).sum() >= 5
        else None
    )
    OUT.mkdir(exist_ok=True)
    result = dict(first=first, pick=pick, halves=half.tolist(), daily=daily.tolist())
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    if pick is not None:
        np.savez(OUT / "model.npz", w=v.fit(history), pick=pick, size=v.price.sizing(cal)[pick])
    print("intraday revision adjustment", half, "selected", pick, flush=True)


def load_change():
    with np.load(OUT / "model.npz") as f:
        w, col, size = f["w"], int(f["pick"]), float(f["size"])
    geom = v.feedback.geometry()
    return lambda raw, q: size * design(raw, q, np.arange(len(q)), geom, w)[:, :, col]


def correction():
    parent, change = v.gate.correction(), load_change()

    def apply(raw, p):
        delta = parent(raw, p)
        q = p.copy()
        q[:, :, 2] += delta
        return delta + change(raw, q)

    return apply


if __name__ == "__main__":
    import argparse
    import torch
    import y3_portfolio_walkforward as runner
    import experiment_short_medium_variance_ratio_v1 as mixed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        runner.replay(
            OUT,
            correction,
            load_final_y2=lambda: mixed.final_head(1),
            load_final_y1=lambda: mixed.final_head(0),
        )
