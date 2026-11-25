"""Calibrate the five-tick response on the same donor basket that issues each signal."""

import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_independent_peer_sensors_v1 as prior
import experiment_short_medium_variance_ratio_v1 as mixed
import build_short_reference_for_peer_sensors_v1 as reference

v = prior.v
OUT = prior.OUT.parent / "y1_source_specific_peer_response_v1"


def source(raw, q, geom, j):
    k, co, row = geom
    level = v.feedback.cash.compound(raw, k)
    miss = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(raw, row)).cumsum(0)))
    t = np.arange(len(raw))
    lag = (2, 8)[j]
    old = (t - lag).clip(0)
    z = (level[t + 1] - level[old + 1]) * co[j, 1] / k + q[t, :, j] - q[old, :, j]
    valid = (t >= lag)[:, None] & (miss[t + 1] == miss[old])
    return np.tanh(z * 2) * valid, level, miss


def geometry(raw, q, geom, j):
    rows = []
    for a, b in v.price.segments(len(raw)):
        rows.append(source(raw[a:b], q[a:b], geom, j)[0][::5])
    x = np.concatenate(rows)
    return x.T @ x / len(x)


def design(raw, q, t, geom, op):
    k, co, _ = geom
    parts = []
    for a, b in v.price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        p = q[a:b]
        x, level, miss = source(raw[a:b], p, geom, 0)
        ticks = np.arange(0, b - a, 2)
        idx = np.arange(len(ticks))
        issue = (ticks - 5).clip(0)
        valid = (ticks >= 5)[:, None] & (miss[ticks + 1] == miss[issue])
        actual = co[0, 1] * np.expm1(level[ticks + 1] - level[issue + 1]) / k
        out = np.empty((len(ix), 50, 6))
        for j, r in enumerate(prior.GROUPS):
            ma, mb, _ = op[j]
            A = 2 * np.tanh(x[:, prior.GROUPS[(j + 1) % 3]] @ ma / 2)
            B = 2 * np.tanh(x[:, prior.GROUPS[(j + 2) % 3]] @ mb / 2)
            ok = valid[:, r]
            n = ok.sum(1).clip(1)

            def center(z):
                return (z - (z * ok).sum(1, keepdims=True) / n[:, None]) * ok

            y = center(actual[:, r])
            base = center(p[issue][:, r, 0])

            def ew(z):
                return lfilter([1 / 140], [1, -139 / 140], z, axis=0)

            beta = (ew((base * y).sum(1) / n) / (0.004 + ew((base**2).sum(1) / n))).clip(0, 2)
            err = y - base * beta[:, None]

            def fit(sensor):
                sample = sensor[issue]
                xy = sample[:, :, None] * err[:, None, :]

                def coef(side):
                    use = (n >= int(0.65 * len(r))) & ((idx // 6) % 2 == side)
                    return ew(xy * use[:, None, None]) / (0.12 + ew(sample**2 * use[:, None]))[:, :, None]

                l, rr = coef(0), coef(1)
                return np.where(l * rr > 0, np.sign(l) * np.minimum(abs(l), abs(rr)), 0)

            wa, wb = fit(A), fit(B)

            def score(sensor, w, age=0):
                slots = (ix // 2 - age).clip(0)
                return np.einsum("ti,tij->tj", sensor[ix], w[slots], optimize=True)

            left, right = score(A, wa), score(B, wb)
            own = 0.5 * (left + right)
            cross = 0.5 * (score(B, wa) + score(A, wb))
            consensus = np.where(left * right > 0, np.sign(left) * np.minimum(abs(left), abs(right)), 0)
            stale = 0.5 * (score(A, wa, 128) + score(B, wb, 128))
            for col, z in enumerate((own, consensus, own - cross, cross, stale, np.roll(own, 5, axis=1))):
                out[:, r, col] = z
        parts.append((out - out.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(parts)


def train():
    history, cal, days = [], [], []
    geom = v.feedback.geometry()
    from experiment_causal_cross_horizon_ridge_v1 import labels

    for d in range(4, 16):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        with np.load(prior.tape.OUT / f"day_{d:02d}.npz") as f:
            q, t = f["prediction"], f["t"]
        if d >= 6:
            with np.load(reference.OUT / f"day_{d:02d}.npz") as f:
                np.testing.assert_array_equal(f["t"], t)
                p = q[t].copy()
                p[:, :, :2] = f["baseline"]
            if d % 2 == 0:
                op = prior.axes(history)
            y = labels(d).reshape(q.shape)
            z = design(raw, q, t, geom, op)
            if d >= 8:
                if d % 2 == 0:
                    size = v.price.sizing(cal)
                days.append(mixed.stat(p, y[t], z * size, 0))
                if d == 8:
                    cut = 2487
                    use = t < cut
                    np.testing.assert_allclose(
                        z[use], design(raw[:cut], q[:cut], t[use], geom, op), atol=3e-7, rtol=0
                    )
            cal.append(mixed.stat(p, y[t], z, 0))
        history.append(geometry(raw, q, geom, 0))
    halves = np.stack([v.price.diff(sum(g))[1:] for g in (days[:4], days[4:])])
    daily = np.stack([v.price.diff(g)[1:] for g in days])
    j = int(np.argmax(halves[0, :3]))
    pick = (
        j
        if (halves[:, j] > np.maximum(0, halves[:, 3:].max(1))).all() and (daily[:, j] > 0).sum() >= 5
        else None
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(halves=halves.tolist(), daily=daily.tolist(), first=j, pick=pick), indent=2)
    )
    if pick is not None:
        np.savez(OUT / "model.npz", cov=np.mean(history, 0), col=pick, size=v.price.sizing(cal)[pick])
    print(
        "five-tick source-specific response",
        halves,
        "positive dates",
        (daily[:, j] > 0).sum(),
        "selected",
        pick,
        flush=True,
    )


def change():
    with np.load(OUT / "model.npz") as f:
        op = prior.axes([f["cov"]])
        c, sz = int(f["col"]), float(f["size"])
    geom = v.feedback.geometry()
    return lambda raw, q: sz * design(raw, q, np.arange(len(q)), geom, op)[:, :, c]


def final_head():
    old, extra = mixed.final_head(0), change()
    return lambda raw, p: old(raw, p) + extra(raw, p)


if __name__ == "__main__":
    import argparse
    import torch
    import y3_portfolio_walkforward as runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        runner.replay(
            OUT, prior.correction, load_final_y2=lambda: mixed.final_head(1), load_final_y1=final_head
        )
