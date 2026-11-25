"""Scale an existing signed forecast using the part of future return size explained by live volatility."""

import argparse
import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_forecast_trajectory_coupling_v1 as trajectory

price = trajectory.price
OUT = price.OUT.parent / "y3_incremental_magnitude_ratio_v1"


def risk(raw, t):
    pieces = []
    for lo, hi in price.segments(len(raw)):
        u = t[(lo <= t) & (t < hi)] - lo
        if not len(u):
            continue
        move = abs(raw[lo:hi, :, 0].astype(float) - 0.00136).clip(0, 8)
        sm = lfilter([1 / 75], [1, 1 / 75 - 1], move - move[:1], axis=0) + move[:1]
        pieces.append(sm[u])
    return np.concatenate(pieces).astype(np.float32)


def bands(q, ratio):
    full = q[:, :, 2]
    near = q[:, :, 1] * ratio
    v = np.stack((full, near, full - near), 2)
    return v - v.mean(1, keepdims=True)


def indices(x, v, ref):
    risk_level, scale = ref
    high = (v > risk_level).astype(int)
    strength = np.digitize(abs(x) / scale, [0.45, 1.15, 2.0])
    return ((np.arange(50)[None, :, None] * 2 + high[:, :, None]) * 4 + strength) * 3 + np.arange(3)


def summarize(x, v, y, ref):
    idx = indices(x, v, ref).ravel()
    shape = (50, 2, 4, 3)
    n = np.bincount(idx, minlength=np.prod(shape)).reshape(shape)
    a = np.bincount(idx, weights=abs(y).clip(0, 7).ravel(), minlength=np.prod(shape)).reshape(shape)
    return np.stack((a, n))


def magnitude(record):
    a, n = record

    def pooling(axes):
        return a.sum(axes, keepdims=True) / n.sum(axes, keepdims=True).clip(1)

    # Borrow from the same asset and same market state before estimating fine strength bins.
    parent = 0.6 * pooling((1, 2)) + 0.4 * pooling((0,))
    return ((a + 180 * parent) / (n + 180)).clip(0.04, 5).astype(np.float32)


def training_panels():
    hist, cal = [], []
    for d, raw, t, q, y, ok, geo in trajectory.training_panels():
        x = trajectory.design(raw, q, t, geo)[0]
        p = q[t].copy()
        if d >= 6:
            if d % 2 == 0:
                w = trajectory.fit(hist)[:, 6]
            v = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)[0]
                p[:, :, 2] += size * v
            cal.append(price.rscore(q[t], y[t], v[:, :, None]))
        hist.append(trajectory.cohort.motion.parent.stat(x[ok], q[t[ok]], y[t[ok]], geo[-1]))
        q[t] = p
        yield d, raw, t, q, y, ok, geo


ARMS = (
    "incumbent",
    "live magnitude ratio",
    "near and distant ratios",
    "distal only",
    "live ratio above market peers",
    "agreement between earlier and later training dates",
    "volatility states of other stocks",
)


def fit(hist):
    v, p, y = [np.concatenate(z) for z in zip(*hist)]
    ref = np.median(v, 0), np.sqrt((p * p).mean(0)).clip(0.02)
    records = np.stack([summarize(x, u, t, ref) for u, x, t in hist])

    def make(r):
        m = magnitude(r)
        # An identical table estimator with no knowledge of which volatility state applies at prediction.
        summed = r.sum(2, keepdims=True)
        refm = magnitude(np.repeat(0.5 * summed, 2, axis=2))
        return np.stack((m, refm))

    return (*ref, np.stack([make(s.sum(0)) for s in (records, records[::2], records[1::2])]))


def signals(v, p, ratio, state):
    x = bands(p, ratio)
    ids = indices(x, v, state[:2])
    table = state[2]

    def curve(pair):
        m, ref = pair.reshape(2, -1)[:, ids]
        return np.log(m / ref).clip(-0.65, 0.65)

    all_, first, last = map(curve, table)
    stable = np.where(first * last > 0, np.sign(first) * np.minimum(abs(first), abs(last)), 0.0)
    fake_ids = indices(x, np.roll(v, 15, axis=1), state[:2])
    mw, refw = table[0].reshape(2, -1)[:, fake_ids]
    fake = np.log(mw / refw).clip(-0.65, 0.65)
    effect = x * all_
    z = np.stack(
        (
            effect[:, :, 0],
            effect[:, :, 1] + effect[:, :, 2],
            effect[:, :, 2],
            x[:, :, 0] * (all_[:, :, 0] - all_[:, :, 0].mean(1, keepdims=True)),
            x[:, :, 0] * stable[:, :, 0],
            x[:, :, 0] * fake[:, :, 0],
        ),
        axis=2,
    )
    estimates = table[0].reshape(2, -1)[:, ids]
    return (z - z.mean(1, keepdims=True)).astype(np.float32), estimates


def train():
    hist, cal, days, parts, diagnostics = [], [], [], [], []
    for d, raw, t, q, y, ok, geo in training_panels():
        p = bands(q[t], geo[-1])
        v = risk(raw, t)
        yy = bands(y[t], geo[-1])
        if d >= 6:
            if d % 2 == 0:
                state = fit(hist)
            z, (m, ref) = signals(v, q[t], geo[-1], state)
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                dz = size * z
                days.append(price.rscore(q[t], y[t], dz))
                parts.append(
                    [
                        trajectory.cohort.motion.parent.gram(q[t], y[t], a, geo[-1])
                        for a in dz.transpose(2, 0, 1)
                    ]
                )

                def corr(a, b):
                    return np.corrcoef(a.ravel(), b.ravel())[0, 1]

                diagnostics.append(
                    [
                        dict(
                            mse_live=np.mean((u[ok] - abs(w[ok])) ** 2),
                            mse_fixed=np.mean((f[ok] - abs(w[ok])) ** 2),
                            partial_size_corr=corr(u[ok] - f[ok], abs(w[ok]) - f[ok]),
                        )
                        for u, f, w in zip(m.T.swapaxes(1, 2), ref.T.swapaxes(1, 2), yy.T.swapaxes(1, 2))
                    ]
                )
                if d == 8:
                    cut = t < 2450
                    short = signals(risk(raw[:2450], t[cut]), q[t[cut]], geo[-1], state)[0]
                    np.testing.assert_allclose(short, z[cut], atol=3e-6, rtol=0)
            cal.append(price.rscore(q[t], y[t], z))
        hist.append((v[ok], p[ok], yy[ok]))
    h = np.stack([price.diff(sum(v)) for v in (days[:4], days[4:])])
    ds = np.stack([price.diff(v) for v in days])
    attr = [
        [trajectory.cohort.motion.parent.decompose(g) for g in np.sum(v, axis=0)]
        for v in (parts[:4], parts[4:])
    ]
    eligible = [
        j
        for j in range(1, 6)
        if np.all(h[:, j] > np.maximum(0, h[:, 6]))
        and (ds[:, j] > 0).sum() >= 5
        and all(a[j - 1]["late"] > 0 for a in attr)
    ]
    pick = max(eligible, key=lambda j: h[:, j].min()) if eligible else 0
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                arms=ARMS,
                selected=pick,
                halves=h.tolist(),
                daily=ds.tolist(),
                legs=attr,
                calibration=diagnostics,
            ),
            indent=2,
            default=float,
        )
    )
    print(h, "selected", pick, flush=True)
    if pick:
        r, scale, table = fit(hist)
        np.savez(
            OUT / "model.npz",
            r=r,
            scale=scale,
            table=table,
            ratio=geo[-1],
            pick=pick - 1,
            size=price.sizing(cal)[pick - 1],
        )
    return pick


def load():
    old = trajectory.load()
    with np.load(OUT / "model.npz") as f:
        state = tuple(f[k] for k in ("r", "scale", "table"))
        ratio, pick, size = float(f["ratio"]), int(f["pick"]), float(f["size"])

    def apply(raw, p):
        before = old(raw, p)
        q = p.copy()
        q[:, :, 2] += before
        z = signals(risk(raw, np.arange(len(q))), q, ratio, state)[0]
        return before + size * z[:, :, pick]

    return apply


def correction():
    prior = trajectory.cohort.motion.current.base.memory.correction()
    last = load()

    def apply(raw, p):
        shift = prior(raw, p)
        q = p.copy()
        q[:, :, 2] += shift
        return shift + last(raw, q)

    return apply


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        trajectory.cohort.motion.parent.replay(OUT, correction)
