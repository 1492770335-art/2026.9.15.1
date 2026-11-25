"""Estimate intraday transfer between stock portfolios from past revisions and matured portfolio errors."""

import json
import numpy as np
from scipy.signal import lfilter

import experiment_y3_partial_maturity_orientation_v1 as partial

v = partial.v
OUT = partial.OUT.parent / "y3_matured_revision_factor_transport_v1"


def signals(raw, q, t, geom, oldsize):
    # The deployed vintage arm depends only on the five univariate revisions.
    # Its selected projection is independent of the alternative supervised weight vector.
    w = np.zeros(24)
    full = partial.prior.design(raw, q, np.arange(len(q)), geom, w)[:, :, 3]
    p = q.copy()
    p[:, :, 2] += oldsize * full
    z = partial.design(raw, p, t, geom)[:, :, 1]
    return np.stack((full[t], z), axis=2), p


def geometry(q):
    x = q[::6, :, 1].astype(float)
    x = x - x.mean(1, keepdims=True)
    return x.T @ x / len(x)


def axes(history):
    g = np.mean(history, 0)
    sd = np.sqrt(g.diagonal().clip(0.0001))
    _, vec = np.linalg.eigh(g / sd[:, None] / sd[None, :])
    return vec[:, -5:]


def design(raw, q, t, geom, axis):
    k, co, row = geom
    rows = []
    for a, b in v.price.segments(len(raw)):
        ii = t[(t >= a) & (t < b)] - a
        if not len(ii):
            continue
        ticks = np.arange(0, b - a, 4)
        u = np.union1d(ticks, ii)
        p = q[a:b]
        level = v.feedback.cash.compound(raw[a:b], k)
        miss = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(raw[a:b], row)).cumsum(0)))
        old = (u - 8).clip(0)
        move = co[2, 1] * (level[u + 1] - level[old + 1]) / k
        # Separate price-adjusted changes to the near and distal pieces of the forecast.
        near = (p[u, :, 1] - p[old, :, 1]) * co[2, 1] / co[1, 1] + move
        far = (p[u, :, 2] - p[old, :, 2]) + move - near
        good = (u >= 8)[:, None] & (miss[u + 1] == miss[old])
        revisions = np.stack((near, far), axis=2) * good[:, :, None]
        revisions -= revisions.mean(1, keepdims=True)
        pos = np.arange(len(ticks))

        def bank(basis):
            src = np.einsum("tsc,si->tci", revisions, basis)
            src = 2 * np.tanh(src / 2)
            past = (pos - 15).clip(0)
            ok = (ticks >= 60)[:, None] & (miss[ticks + 1] == miss[ticks[past]])
            n = ok.sum(1).clip(1)
            sufficient = n >= 32

            def center(x):
                return (x - (x * ok).sum(1, keepdims=True) / n[:, None]) * ok

            y = center(co[2, 1] * np.expm1(level[ticks + 1] - level[ticks[past] + 1]) / k)
            pred = center(p[ticks[past], :, 2])

            def ew(x):
                return lfilter([1 / 100], [1, -0.99], x, axis=0)

            slope = (ew((pred * y).sum(1) / n) / (0.003 + ew((pred**2).sum(1) / n))).clip(0, 2)
            error = (y - slope[:, None] * pred) @ axis
            x = src[np.searchsorted(u, ticks[past])]
            xy = x[:, :, :, None] * error[:, None, None, :]
            xx = x * x

            def fit(mask):
                observed = sufficient * mask
                num = ew(xy * observed[:, None, None, None])
                denom = 0.07 + ew(xx * observed[:, None, None])
                return (num / denom[:, :, :, None]).clip(-2, 2)

            pooled = fit(pos >= 0)
            left, right = [fit((pos // 20) % 2 == j) for j in (0, 1)]
            stable = np.where(left * right > 0, np.sign(left) * np.minimum(abs(left), abs(right)), 0)
            here = src[np.searchsorted(u, ii)]

            def encode(coeff, diagonal=False):
                if diagonal:
                    coeff = coeff * np.eye(axis.shape[1])[None, None, :, :]
                transported = np.einsum("tci,tcij->tcj", here, coeff[ii // 4], optimize=True)
                return np.einsum("tcj,sj->tsc", transported, axis, optimize=True)

            return [encode(pooled), encode(stable), encode(pooled, True)]

        full, stable, diagonal = bank(axis)
        wrong = bank(np.roll(axis, 17, axis=0))[0]
        rows.append(np.concatenate((full, stable, full - diagonal, wrong), axis=2).astype(np.float32))
    return np.concatenate(rows)


def train():
    history, old1, old2, cal, days = [], [], [], [], []
    geom = v.feedback.geometry()
    from experiment_causal_cross_horizon_ridge_v1 import labels

    for d in range(4, 16):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        y = labels(d).reshape(-1, 50, 3)
        with np.load(v.TAPE / f"day_{d:02d}.npz") as f:
            t = f["t"]
            q = np.concatenate((f["short"], f["dense"][:, :, None]), axis=2)
        if d >= 6:
            if d % 2 == 0:
                axis = axes(history)
                s1 = v.price.sizing(old1)[0] if old1 else 0
                s2 = v.price.sizing(old2)[0] if old2 else 0
            z, p = signals(raw, q, np.arange(len(q)), geom, s1)
            old1.append(v.price.rscore(q[t], y[t], z[t, :, :1]))
            old2.append(v.price.rscore(p[t], y[t], z[t, :, 1:]))
            p[:, :, 2] += s2 * z[:, :, 1]
            shift = design(raw, p, t, geom, axis)
            if d >= 8:
                if d % 2 == 0:
                    size = v.price.sizing(cal)
                days.append(v.price.rscore(p[t], y[t], shift * size))
                if d == 8:
                    n = 2487
                    np.testing.assert_allclose(
                        shift[t < n], design(raw[:n], p[:n], t[t < n], geom, axis), atol=3e-7, rtol=0
                    )
            cal.append(v.price.rscore(p[t], y[t], shift))
        history.append(geometry(q))
    halves = np.stack([v.price.diff(sum(s))[1:] for s in (days[:4], days[4:])])
    daily = np.stack([v.price.diff(s)[1:] for s in days])
    first = int(np.argmax(halves[0, :6]))
    pick = (
        first
        if (halves[:, first] > np.maximum(0, halves[:, 6 + first % 2])).all()
        and (daily[:, first] > 0).sum() >= 5
        else None
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(halves=halves.tolist(), daily=daily.tolist(), first=first, pick=pick), indent=2)
    )
    if pick is not None:
        np.savez(OUT / "model.npz", axis=axes(history), size=v.price.sizing(cal)[pick], col=pick)
    print("matured revision transport", halves, "selected", pick, flush=True)


def load_change():
    with np.load(OUT / "model.npz") as f:
        axis, col, size = f["axis"], int(f["col"]), float(f["size"])
    geom = v.feedback.geometry()
    return lambda raw, q: size * design(raw, q, np.arange(len(q)), geom, axis)[:, :, col]


def correction():
    previous, change = partial.correction(), load_change()

    def apply(raw, p):
        shift = previous(raw, p)
        q = p.copy()
        q[:, :, 2] += shift
        return shift + change(raw, q)

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
