"""Align five staggered estimates of past price variation before applying the frozen Y3 readout."""

import json
import numpy as np
from scipy.signal import lfilter

import experiment_y3_local_jump_variation_v1 as parent

price, base = parent.price, parent.base
OUT = price.OUT.parent / "y3_bipower_phase_pool_v1"
ARMS = (
    "retained single phase",
    "single latest phase copy",
    "uniform trailing phases",
    "recency-weighted phases",
    "median trailing phases",
    "one block stale",
    "other stock change in phase average",
)


def features(raw, p, k, sd, axis, w):
    panels = []
    for a, b in price.segments(len(raw)):
        level, x = parent.score_inputs(raw[a:b], k, sd, axis)
        t = np.arange(len(level))
        slow = lfilter([0.05], [1.0, -0.95], x[:, :, 0], axis=0)
        q = p[a:b]
        marks = np.concatenate(
            (x[:, :, :1], slow[:, :, None], x[:, :, 1:], q - q.mean(1, keepdims=True)), axis=2
        )
        taps = parent.variation(level)[:, :, :, [0, 2]]
        # The retained readout has nonzero coefficients on marks, total risk and adjacent bipower.
        assert np.count_nonzero(w[18:30]) == 0 and np.count_nonzero(w[42:]) == 0
        z = []
        offsets = (t[:, None] - np.arange(5)).clip(0)
        for i in range(0, len(t), 128):
            rows = t[i : i + 128]
            latest = taps[rows]
            lags = taps[offsets[rows]]
            mean = lags.mean(1)
            recent = np.einsum("i,tisvh->tsvh", np.array([5.0, 4.0, 3.0, 2.0, 1.0]) / 15, lags)
            stale = taps[(rows - 5).clip(0)]
            local_marks = marks[rows]

            def apply(g):
                u = local_marks @ w[:6]
                for j, pos in enumerate((6, 30)):
                    weights = w[pos : pos + 12].reshape(2, 6)
                    u += (local_marks @ weights.T * g[:, :, :, j]).sum(2)
                return u - u.mean(1, keepdims=True)

            single = apply(latest)
            vals = [
                single,
                *(
                    apply(v) - single
                    for v in (
                        latest,
                        mean,
                        recent,
                        np.median(lags, axis=1),
                        stale,
                        latest + np.roll(mean - latest, 17, axis=1),
                    )
                ),
            ]
            z.append(np.stack(vals, axis=2))
        panels.append(np.concatenate(z))
    return np.concatenate(panels).astype(np.float32)


def train():
    k, co, _ = price.previous_stage.early_slopes.geometry()
    ratio = co[2, 1] / co[1, 1]
    axis = parent.direction(k)
    hist, cal, days, parts = [], [], [], []
    for d, _, p, y, model in base.records():
        raw = np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        t = np.arange(d % 5, len(p), 5)
        x = parent.design(raw, p, t, k, model[2], axis)
        if d >= 6:
            if d % 2 == 0:
                w = parent.weights(hist)[:, 3]
            z = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)[0]
                panel = features(raw, p, k, model[2], axis, w * size)
                np.testing.assert_allclose(panel[t, :, 0], z * size, rtol=0, atol=1e-6)
                np.testing.assert_allclose(panel[:, :, 1], 0.0, rtol=0, atol=0)
                q = p.copy()
                q[:, :, 2] += panel[:, :, 0]
                days.append(price.rscore(q, y, panel[:, :, 1:]))
                parts.append([parent.gram(q, y, panel[:, :, j], ratio) for j in (2, 3, 4)])
                if d == 8:
                    n = 2519
                    np.testing.assert_allclose(
                        panel[:n], features(raw[:n], p[:n], k, model[2], axis, w * size), rtol=0, atol=4e-6
                    )
            cal.append(price.rscore(p[t], y[t], z[:, :, None]))
        ok = ((t < 2401 - 60) | (t >= 2401)) & (t < len(p) - 60)
        hist.append(parent.stat(x[ok], p[t[ok]], y[t[ok]], ratio))
    halves = np.stack([price.diff(sum(v)) for v in (days[:4], days[4:])])
    daily = np.stack([price.diff(v) for v in days])
    attr = [[parent.decompose(g) for g in np.sum(v, axis=0)] for v in (parts[:4], parts[4:])]
    eligible = [
        j
        for j in (2, 3, 4)
        if (halves[:, j] > 0).all()
        and (daily[:, j] > 0).sum() >= 5
        and halves[:, j].min() > max(halves[:, 5].min(), halves[:, 6].min())
        and all(row[j - 2]["late"] > 0 for row in attr)
    ]
    pick = max(eligible, key=lambda j: halves[:, j].min()) if eligible else 0
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(arms=ARMS, selected=pick, halves=halves.tolist(), daily=daily.tolist(), attribution=attr),
            indent=2,
        )
    )
    print("selected", pick, "training Y3 gains", halves, flush=True)


def load():
    saved = json.loads((OUT / "results.json").read_text())
    pick = int(saved["selected"])
    if not pick:
        raise ValueError("no training-period selection")
    with np.load(parent.OUT / "model.npz") as f:
        k, sd, axis, w = float(f["k"]), f["sd"], f["axis"], f["w"]
    original = parent.load()

    def apply(raw, p):
        signals = features(raw, p, k, sd, axis, w)
        return original(raw, p) + signals[:, :, pick]

    return apply


def correction():
    prior = base.memory.correction()
    last = load()

    def apply(raw, p):
        shift = prior(raw, p)
        q = p.copy()
        q[:, :, 2] += shift
        return shift + last(raw, q)

    return apply


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        parent.replay(OUT, correction)
