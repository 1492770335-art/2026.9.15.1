"""Decompose forecasts into market, cohort and stock terms using two consecutive pressure transitions."""

import argparse
import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_absolute_block_motion_v1 as motion

price = motion.price
OUT = price.OUT.parent / "y3_pressure_cohort_projection_v1"
ARMS = (
    "retained",
    "recency cohorts",
    "common market from earlier transition",
    "stock cohort from earlier transition",
    "current stock interaction",
    "common and stock projections",
    "misassigned earlier transition",
    "absolute pressure",
)


def group_signal(tags, pred):
    dev = tags - tags.mean(1, keepdims=True)
    dev /= np.sqrt(np.square(dev).mean(1, keepdims=True) + 0.08)
    premium = (dev[:, :, None] * pred).mean(1, keepdims=True)
    market = np.broadcast_to(premium, pred.shape)
    cohort = dev[:, :, None] * premium
    own = dev[:, :, None] * (pred - pred.mean(1, keepdims=True)) - market
    return np.concatenate((market, cohort, own), 2)


def design(raw, q, t, k, sd, axis, ratio):
    rows = []
    for lo, hi in price.segments(len(q)):
        u = t[(t >= lo) & (t < hi)] - lo
        if not len(u):
            continue
        _, marks = motion.parent.score_inputs(raw[lo:hi], k, sd, axis)
        z = marks[:, :, 0]
        smooth = lfilter([0.18], [1.0, -0.82], z - z[:1], axis=0) + z[:1]
        a, b, c = [np.tanh(2.5 * smooth[(u - lag).clip(0)]) for lag in (0, 8, 40)]
        last = a * b
        past = b * c
        # Stocks with identical recent agreement can have different earlier paths.
        ix = u + lo
        pred = q[ix].copy()
        pred[:, :, 1] *= ratio
        pred[:, :, 2] -= pred[:, :, 1]
        level = pred.mean(1, keepdims=True)
        basic = np.concatenate((np.broadcast_to(level, pred.shape), pred - level), 2)

        def contrasts(historical):
            agree = group_signal(historical, pred)
            order = group_signal(last * historical, pred)
            return np.concatenate((agree, order), 2) * (u >= 40)[:, None, None]

        rows.append(
            np.concatenate(
                (
                    basic,
                    group_signal(last, pred),
                    contrasts(past),
                    contrasts(np.roll(past, 17, axis=1)),
                    group_signal(abs(a), pred),
                ),
                2,
            ).astype(np.float32)
        )
    return np.concatenate(rows)


def fit(hist):
    arr = np.stack(hist)
    cov = arr.mean(0)[:-2, :-2]
    grads = arr[:, :-2, -2:]
    recent = np.arange(6, 15)

    def cols(start, offset):
        return np.r_[
            np.arange(start + offset, start + offset + 3), np.arange(start + 9 + offset, start + 12 + offset)
        ]

    common = cols(15, 0)
    cohort = cols(15, 3)
    own = cols(15, 6)
    models = []
    for j, use in enumerate(
        (
            recent,
            common,
            cohort,
            own,
            np.r_[common, cohort],
            np.r_[cols(33, 0), cols(33, 3)],
            np.arange(51, 60),
        )
    ):
        ctrl = np.arange(6 if j == 0 else 15)
        b = np.eye(len(cov))[:, use]
        a = cov[np.ix_(ctrl, ctrl)]
        b[ctrl] = -np.linalg.solve(a + 0.05 * np.diag(a.diagonal().clip(2e-5)), cov[np.ix_(ctrl, use)])
        g = b.T @ cov @ b
        daily = np.einsum("dij,ik->dkj", grads, b)
        mu = daily.mean(0)
        mu *= abs(mu) / (abs(mu) + daily.std(0) / np.sqrt(len(daily)) + 2e-5)
        full, late = mu.T
        shared = np.where(full * late > 0, np.sign(full) * np.minimum(abs(full), abs(late)), 0.0)
        models.append(b @ np.linalg.solve(g + 0.9 * np.diag(g.diagonal().clip(2e-5)), shared))
    return np.stack(models, axis=1).astype(np.float32)


def moment(q, y, ratio):
    q, y = q.astype(float), y.astype(float)
    x = np.stack((q[:, :, 2], ratio * q[:, :, 1], y[:, :, 2], ratio * y[:, :, 1]), axis=2)
    m = x.mean(1, keepdims=True)
    common = np.column_stack((np.ones(len(x)), m[:, 0]))
    within = (x - m).reshape(-1, x.shape[-1])
    w = np.zeros((5, 5))
    w[1:, 1:] = within.T @ within
    return np.stack((50 * common.T @ common, w))


def report(g, prior):
    total = g.sum(0)
    counts = total[0, 0]
    scales, offsets = [], []
    for a, b in ((1, 3), (2, 4)):
        beta = (prior[a, b] - prior[0, a] * prior[0, b] / prior[0, 0]) / (
            prior[a, a] - prior[0, a] ** 2 / prior[0, 0]
        )
        scales.append(beta)
        offsets.append((prior[0, b] - beta * prior[0, a]) / prior[0, 0])
    eye = np.eye(5)
    p = eye[1] * scales[0] + eye[0] * offsets[0]
    pn = eye[2] * scales[1] + eye[0] * offsets[1]
    stats = {}
    for name, pred, real in (
        ("full", p, eye[3]),
        ("first_20", pn, eye[4]),
        ("last_40", p - pn, eye[3] - eye[4]),
    ):
        e = real - pred
        full = e @ total @ e
        row = {}
        for tag, part in zip(("market", "relative"), g):
            centered = part if tag == "relative" else part - np.outer(part[:, 0], part[0]) / counts
            row[tag] = dict(
                mse=(e @ part @ e) / counts,
                error_share=(e @ part @ e) / full,
                corr=(pred @ centered @ real) / np.sqrt((pred @ centered @ pred) * (real @ centered @ real)),
            )
        stats[name] = row
    return stats


def training_panels():
    k, co, _ = price.previous_stage.early_slopes.geometry()
    ratio = co[2, 1] / co[1, 1]
    axis = motion.parent.direction(k)
    sd = price.innovation_model(
        [
            price.observe(price.tape(np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r"), k))
            for d in range(4)
        ]
    )[2]
    history, calibration = [], []
    for d, raw, q, y in motion.records():
        t = np.arange(d % 5, len(q), 5)
        v = motion.design(raw, q, t, k, sd, axis)
        p = q[t].copy()
        if d >= 6:
            if d % 2 == 0:
                w = motion.weights(history)[:, 0]
            shift = v @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(calibration)[0]
                p[:, :, 2] += shift * size
            calibration.append(price.rscore(q[t], y[t], shift[:, :, None]))
        ok = ((t < 2401 - 60) | (t >= 2401)) & (t < len(q) - 60)
        history.append(motion.parent.stat(v[ok], q[t[ok]], y[t[ok]], ratio))
        issued = q.copy()
        issued[t] = p
        yield d, raw, t, issued, y, ok, (k, sd, axis, ratio)


def train():
    hist, cal, days, diagnostic = [], [], [], []
    for d, raw, t, q, y, ok, (k, sd, axis, ratio) in training_panels():
        x = design(raw, q, t, k, sd, axis, ratio)
        diagnostic.append(moment(q[t], y[t], ratio))
        if d >= 6:
            if d % 2 == 0:
                w = fit(hist)
            z = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                days.append(price.rscore(q[t], y[t], z * size))
                if d == 8:
                    n = 2517
                    np.testing.assert_allclose(
                        x[t < n], design(raw[:n], q[:n], t[t < n], k, sd, axis, ratio), rtol=0, atol=1e-6
                    )
            cal.append(price.rscore(q[t], y[t], z))
        hist.append(motion.parent.stat(x[ok], q[t[ok]], y[t[ok]], ratio))
    h = np.stack([price.diff(sum(f)) for f in (days[:4], days[4:])])
    ds = np.stack([price.diff(g) for g in days])
    eligible = [
        j
        for j in (2, 3, 4, 5)
        if h[:, j].min() > max(0, h[:, 6].min(), h[:, 7].min()) and (ds[:, j] > 0).sum() >= 5
    ]
    pick = max(eligible, key=lambda j: h[:, j].min()) if eligible else 0
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(selected=pick, arms=ARMS, halves=h.tolist(), daily=ds.tolist()), indent=2)
    )
    prior = np.sum(diagnostic[:4], axis=(0, 1))
    summary = {
        key: report(sum(folds), prior)
        for key, folds in (("9-12", diagnostic[4:8]), ("13-16", diagnostic[8:]))
    }
    (OUT / "forecast_decomposition.json").write_text(json.dumps(summary, indent=2))
    print(h, "pick", pick, flush=True)
    if pick:
        np.savez(
            OUT / "model.npz",
            w=fit(hist)[:, pick - 1] * price.sizing(cal)[pick - 1],
            k=k,
            sd=sd,
            axis=axis,
            ratio=ratio,
        )
    return pick


def load():
    old = motion.load()
    with np.load(OUT / "model.npz") as f:
        w, geo = f["w"], (float(f["k"]), f["sd"], f["axis"], float(f["ratio"]))

    def apply(raw, p):
        before = old(raw, p)
        q = p.copy()
        q[:, :, 2] += before
        return before + design(raw, q, np.arange(len(q)), *geo) @ w

    return apply


def correction():
    prior = motion.current.base.memory.correction()
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
        motion.parent.replay(OUT, correction)
