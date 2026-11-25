"""Refine a return magnitude conditional on tick activity with observed multi-step relative-price variance ratios."""

import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_pooled_revision_feedback_v1 as feedback
import experiment_y3_incremental_magnitude_ratio_v1 as prior

price = feedback.price
payoff = prior.trajectory.cohort.motion.parent
OUT = feedback.OUT.parent / "y3_variance_ratio_magnitude_v1"
NAMES = ("5/1 variance", "20/5 variance", "40/5 variance", "peer-adjusted 20/5 variance")
LEGS = ("distal", "near + distal", "full")
MODES = ("conditional variance ratio", "other stock ratio", "trailing old ratio", "activity alone")
SHAPE = (50, 2, 3, 3, 3)


def risk_states(raw, t, geom):
    k, coeff, row = geom
    parts = []
    for a, b in price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        r = raw[a:b]
        level = feedback.cash.compound(r, k)[1:] / k * coeff[2, 1]
        miss = np.vstack((np.zeros((1, 50), int), (~feedback.cash.present(r, row)).cumsum(0)))
        i = np.arange(len(r))
        variance = []
        for h in (1, 5, 20, 40):
            past = (i - h).clip(0)
            valid = (i >= h)[:, None] & (miss[i + 1] == miss[past])
            move = feedback.centered(level - level[past], valid) / np.sqrt(h)
            obs = np.minimum(move * move, 2.5)

            def smooth(x):
                return lfilter([1 / 160], [1, 1 / 160 - 1], x, axis=0)

            variance.append((0.0005 + smooth(obs)) / (0.1 + smooth(valid)))
        v1, v5, v20, v40 = variance
        ratio = np.stack([np.log(v5 / v1), np.log(v20 / v5), np.log(v40 / v5)], 2)
        states = np.concatenate((ratio, ratio[:, :, 1:2] - ratio[:, :, 1:2].mean(1, keepdims=True)), 2)
        parts.append(np.stack((states[ix], states[(ix - 160).clip(0)]), 2).astype(np.float32))
    return prior.risk(raw, t), np.concatenate(parts)


def geometry(history):
    r, states, p, _ = map(np.concatenate, zip(*history))
    return (
        np.median(r, axis=0),
        np.quantile(abs(p), [0.45, 0.8], axis=0),
        np.quantile(states[:, :, 0], [1 / 3, 2 / 3], axis=0),
    )


def keys(r, states, p, thresholds):
    rv, pv, sv = thresholds
    live = (r > rv).astype(int)
    strength = (abs(p) > pv[0]).astype(int) + (abs(p) > pv[1])
    basekey = ((np.arange(50)[None, :, None] * 2 + live[:, :, None]) * 3 + strength) * 3
    basekey = basekey * 3 + np.arange(3)
    key = (states > sv[0]).astype(int) + (states > sv[1])
    return basekey, key


def fit(history):
    th = geometry(history)
    count = np.zeros((4, *SHAPE))
    total = np.zeros_like(count)
    for r, st, p, y in history:
        row, tag = keys(r, st[:, :, 0], p, th)
        for j in range(4):
            code = (row + 3 * tag[:, :, j, None]).ravel()
            count[j] += np.bincount(code, minlength=np.prod(SHAPE)).reshape(SHAPE)
            total[j] += np.bincount(
                code, weights=np.minimum(abs(y), 5).ravel(), minlength=np.prod(SHAPE)
            ).reshape(SHAPE)
    n = count.sum(4, keepdims=True)
    s = total.sum(4, keepdims=True)
    stock = (s.sum((2, 3), keepdims=True) + 0.05) / (n.sum((2, 3), keepdims=True) + 1)
    coarse = (s + 120 * stock) / (n + 120)
    # A pooled multiplicative ratio borrows evidence from stocks at the same activity and strength.
    broad = (total.sum(1, keepdims=True) + 100 * coarse.mean(1, keepdims=True)) / (
        count.sum(1, keepdims=True) + 100
    )
    base_broad = (s.sum(1, keepdims=True) + 100 * coarse.mean(1, keepdims=True)) / (
        n.sum(1, keepdims=True) + 100
    )
    local = (total + 180 * coarse * (broad / base_broad).clip(0.65, 1.55)) / (count + 180)
    unconditional = (s.sum(2, keepdims=True) + 120 * stock) / (n.sum(2, keepdims=True) + 120)
    return th, coarse, local, unconditional


def design(r, st, p, model):
    th, coarse, local, unconditional = model
    idx, tag = keys(r, st[:, :, 0], p, th)
    _, stale = keys(r, st[:, :, 1], p, th)
    z = []
    proxy = []
    for mode in range(4):
        for j in range(4):
            if mode == 1:
                _, wrong = keys(r, np.roll(st[:, :, 0], 17, 1), p, th)
                key = wrong[:, :, j]
            elif mode == 2:
                key = stale[:, :, j]
            else:
                key = tag[:, :, j]
            # Decode coarse tables via the same dense index as the fine table.
            ref = np.broadcast_to(coarse[j], SHAPE).ravel()[idx]
            val = local[j].ravel()[idx + 3 * key[:, :, None]]
            if mode == 3:
                val = ref
                ref = np.broadcast_to(unconditional[j], SHAPE).ravel()[idx]
            corr = p * np.log((val / ref).clip(0.55, 1.8))
            z.extend((corr[:, :, 2], corr[:, :, 1] + corr[:, :, 2], corr[:, :, 0]))
            if mode == 0:
                proxy.append((ref, val))
    mat = np.stack(z, 2)
    return (mat - mat.mean(1, keepdims=True)).astype(np.float32), proxy


def training_panels():
    oldcal = []
    geom = feedback.geometry()
    for d, raw, t, q, before, y, ok, geo, *_ in feedback.training_panels():
        prev = feedback.design(raw, q, t, geom, selected=3)
        oldstat = price.rscore(before, y[t], prev[:, :, None])
        if d >= 8:
            if d % 2 == 0:
                size = price.sizing(oldcal)[0]
            before = before.copy()
            before[:, :, 2] += prev * size
        yield d, raw, t, q, before, y, ok, geo
        oldcal.append(oldstat)


def train():
    geom = feedback.geometry()
    hist, cal, days, legs, quality = [], [], [], [], []

    def inputs(raw, t, p, y, ratio):
        r, st = risk_states(raw, t, geom)
        return r, st, prior.bands(p, ratio), prior.bands(y[t], ratio)

    for d, raw, t, q, y, ok, geo in prior.training_panels():
        if d >= 6:
            break
        data = inputs(raw, t, q[t], y, geo[-1])
        hist.append(tuple(v[ok] for v in data))
    for d, raw, t, q, before, y, ok, geo in training_panels():
        if d % 2 == 0:
            model = fit(hist)
        r, st, p, truth = inputs(raw, t, before, y, geo[-1])
        z, prox = design(r, st, p, model)
        if d >= 8:
            if d % 2 == 0:
                size = price.sizing(cal)
            dz = z * size
            days.append(price.rscore(before, y[t], dz))
            legs.append([payoff.gram(before, y[t], v, geo[-1]) for v in dz[:, :, :12].transpose(2, 0, 1)])
            quality.append(
                [[((m[ok] - abs(truth[ok])) ** 2).mean((0, 1)).tolist() for m in pair] for pair in prox]
            )
            if d == 8:
                end = 2527
                keep = t < end
                rr, ss = risk_states(raw[:end], t[keep], geom)
                np.testing.assert_allclose(z[keep], design(rr, ss, p[keep], model)[0], rtol=0, atol=1e-7)
        hist.append(tuple(v[ok] for v in (r, st, p, truth)))
        cal.append(price.rscore(before, y[t], z))
    h = np.stack([price.diff(sum(ss))[1:].reshape(4, 4, 3) for ss in (days[:4], days[4:])])
    ds = np.stack([price.diff(g)[1:].reshape(4, 4, 3) for g in days])
    at = [[payoff.decompose(g) for g in np.sum(ss, 0)] for ss in (legs[:4], legs[4:])]
    qs = np.stack([np.mean(ss, 0) for ss in (quality[:4], quality[4:])])
    quality = 1 - qs[:, :, 1] / qs[:, :, 0]
    passed = [
        (j, a)
        for j in range(4)
        for a in range(3)
        if np.all(h[:, 0, j, a] > np.maximum(0, h[:, 1:3, j, a].max(1)))
        and (ds[:, 0, j, a] > 0).sum() >= 5
        and np.all(quality[:, j, 2 if a < 2 else 0] > 0)
        and all(c[3 * j + a]["late"] > 0 for c in at)
    ]
    pick = max(passed, key=lambda v: h[:, 0, v[0], v[1]].min()) if passed else None
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                states=NAMES,
                legs=LEGS,
                modes=MODES,
                pick=pick,
                halves=h.tolist(),
                daily=ds.tolist(),
                components=at,
                magnitude_mse_gain_fraction=quality.tolist(),
            ),
            indent=2,
        )
    )
    if pick is not None:
        (rv, pv, sv), co, lo, un = fit(hist)
        j, a = pick
        np.savez(
            OUT / "model.npz",
            rv=rv,
            pv=pv,
            sv=sv,
            co=co,
            lo=lo,
            un=un,
            col=3 * j + a,
            size=price.sizing(cal)[3 * j + a],
            ratio=geo[-1],
        )
    print(
        "variance ratios, magnitude gain", quality[:, :, 2], "Y3 gain", h[:, 0], "selected", pick, flush=True
    )


def load_change():
    geom = feedback.geometry()
    with np.load(OUT / "model.npz") as f:
        model = (tuple(f[k] for k in ("rv", "pv", "sv")), f["co"], f["lo"], f["un"])
        col, size, ratio = int(f["col"]), float(f["size"]), float(f["ratio"])

    def change(raw, before):
        ticks = np.arange(len(raw))
        r, st = risk_states(raw, ticks, geom)
        bands = prior.bands(before, ratio)
        return size * selected_design(r, st, bands, model, col)

    return change


def selected_design(r, st, p, model, col):
    th, coarse, local, _ = model
    row, tag = keys(r, st[:, :, 0], p, th)
    j, leg = divmod(col, 3)
    ref = np.broadcast_to(coarse[j], SHAPE).ravel()[row]
    val = local[j].ravel()[row + 3 * tag[:, :, j, None]]
    correction = p * np.log((val / ref).clip(0.55, 1.8))
    values = (correction[:, :, 2], correction[:, :, 1] + correction[:, :, 2], correction[:, :, 0])[leg]
    return (values - values.mean(1, keepdims=True)).astype(np.float32)


def correction():
    upstream = prior.trajectory.cohort.motion.current.base.memory.correction()
    feedback_update = feedback.load()
    change = load_change()

    def apply(raw, p):
        shift = upstream(raw, p)
        q = p.copy()
        q[:, :, 2] += shift
        first = feedback_update(raw, q)
        before = q.copy()
        before[:, :, 2] += first
        return shift + (first + change(raw, before))

    return apply


if __name__ == "__main__":
    import argparse
    import torch
    import experiment_y1_price_feedback_v1 as short
    import y3_portfolio_walkforward as runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        runner.replay(OUT, correction, load_final_y2=short.parent.load, load_final_y1=short.load)
