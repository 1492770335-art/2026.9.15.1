"""Fit price variance ratio magnitude tables for the final, separately calibrated Y1 and Y2 forecasts."""

import json

import numpy as np

import experiment_y1_price_feedback_v1 as short
import experiment_y3_variance_ratio_magnitude_v1 as ratio

price = ratio.price
feedback = short.feedback
OUT = ratio.OUT.parent / "short_medium_variance_ratio_v1"
STATES = ratio.NAMES
MODES = ("own ratio", "reassigned ratio", "old ratio", "activity only")


def centered(p):
    return p - p.mean(1, keepdims=True)


def design(r, st, p, model):
    th, coarse, local, unconditional = model
    idx, live = ratio.keys(r, st[:, :, 0], p, th)
    _, old = ratio.keys(r, st[:, :, 1], p, th)
    _, wrong = ratio.keys(r, np.roll(st[:, :, 0], 17, axis=1), p, th)
    rows = []
    quality = []
    for mode, tag in enumerate((live, wrong, old, live)):
        state = []
        for j in range(4):
            ref = np.broadcast_to(coarse[j], ratio.SHAPE).ravel()[idx]
            value = local[j].ravel()[idx + 3 * tag[:, :, j, None]]
            if mode == 0:
                quality.append((ref, value))
            if mode == 3:
                value = ref
                ref = np.broadcast_to(unconditional[j], ratio.SHAPE).ravel()[idx]
            state.append(centered(p * np.log((value / ref).clip(0.55, 1.8))))
        rows.append(np.stack(state, 2))
    return np.concatenate(rows, 2).astype(np.float32), quality


def stat(p, y, z, j):
    order = [a for a in range(3) if a != j] + [j]
    return price.rscore(p[:, :, order], y[:, :, order], z)


def source(raw, q, geom):
    t = np.arange(len(q))
    return np.stack([short.live(raw, q, t, geom, 0), feedback.design(raw, q, t, geom, selected=0)], 2)


def source_stats(before, y, z):
    return [
        price.rscore(before[:, :, [2, 1 - j, j]], y[:, :, [2, 1 - j, j]], z[:, :, j : j + 1]) for j in (0, 1)
    ]


def weights(hist):
    return np.array([price.sizing(h)[0] for h in zip(*hist)])


def train():
    geom = feedback.geometry()
    warm, calib, hist, stats, days, quality = [], [], [], [[], []], [[], []], []

    def inputs(r, st, p, y, ok):
        return tuple(v[ok] for v in (r, st, centered(p), centered(y)))

    for d, raw, t, q, y, ok, geo in ratio.prior.training_panels():
        if d >= 6:
            break
        r, st = ratio.risk_states(raw, t, geom)
        hist.append(inputs(r, st, q[t], y[t], ok))
        warm.append(source_stats(q[t], y[t], source(raw, q, geom)[t]))
    for d, raw, t, q, before, y, ok, geo in ratio.training_panels():
        if d % 2 == 0:
            w = weights(calib or warm)
            model = ratio.fit(hist)
        src = source(raw, q, geom)[t]
        calib.append(source_stats(before, y[t], src))
        p = before.copy()
        p[:, :, :2] += src * w
        r, st = ratio.risk_states(raw, t, geom)
        z, proxy = design(r, st, centered(p), model)
        if d >= 8:
            if d % 2 == 0:
                sizes = [price.sizing(records) for records in stats]
            for j in range(2):
                days[j].append(stat(p, y[t], z[:, :, :, j] * sizes[j], j))
            actual = abs(centered(y[t])[ok])
            quality.append(
                [[np.mean((v[ok] - actual) ** 2, (0, 1)).tolist() for v in pair] for pair in proxy]
            )
            if d == 8:
                end = 2529
                use = t < end
                rr, ss = ratio.risk_states(raw[:end], t[use], geom)
                np.testing.assert_allclose(
                    z[use], design(rr, ss, centered(p[use]), model)[0], atol=1e-7, rtol=0
                )
        for j in range(2):
            stats[j].append(stat(p, y[t], z[:, :, :, j], j))
        hist.append(inputs(r, st, p, y[t], ok))
    h = np.stack([[price.diff(sum(ss))[1:].reshape(4, 4) for ss in (dd[:4], dd[4:])] for dd in days])
    ds = np.stack([[price.diff(v)[1:].reshape(4, 4) for v in dd] for dd in days])
    q = np.array([np.mean(ss, 0) for ss in (quality[:4], quality[4:])])
    proxy = 1 - q[:, :, 1] / q[:, :, 0]
    choices = []
    for j in range(2):
        passed = [
            i
            for i in range(4)
            if np.all(h[j, :, 0, i] > np.maximum(0, h[j, :, 1:3, i].max(1)))
            and min(h[j, :, 0, i]) > min(h[j, :, 3, i])
            and (ds[j, :, 0, i] > 0).sum() >= 5
            and (proxy[:, i, j] > 0).all()
        ]
        choices.append(max(passed, key=lambda i: h[j, :, 0, i].min()) if passed else None)
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                states=STATES,
                modes=MODES,
                targets=["Y1", "Y2"],
                pick=choices,
                halves=h.tolist(),
                daily=ds.tolist(),
                magnitude_gain_fraction=proxy.tolist(),
            ),
            indent=2,
        )
    )
    if any(i is not None for i in choices):
        (rv, pv, sv), co, lo, un = ratio.fit(hist)
        np.savez(
            OUT / "model.npz",
            rv=rv,
            pv=pv,
            sv=sv,
            co=co,
            lo=lo,
            un=un,
            picks=[-1 if x is None else x for x in choices],
            sizes=[0.0 if x is None else price.sizing(stats[j])[x] for j, x in enumerate(choices)],
        )
    print("Y1 and Y2 gains", h[:, :, 0], "selected", choices, flush=True)


def load_change():
    geom = feedback.geometry()
    with np.load(OUT / "model.npz") as f:
        th = tuple(f[k] for k in ("rv", "pv", "sv"))
        co, lo = f["co"], f["lo"]
        picks, sizes = f["picks"], f["sizes"]

    def change(raw, p):
        r, st = ratio.risk_states(raw, np.arange(len(raw)), geom)
        features = centered(p)
        row, tag = ratio.keys(r, st[:, :, 0], features, th)
        result = np.zeros((*p.shape[:2], 2), np.float32)
        for j, i in enumerate(picks):
            if i < 0:
                continue
            idx = row[:, :, j]
            ref = np.broadcast_to(co[i], ratio.SHAPE).ravel()[idx]
            val = lo[i].ravel()[idx + 3 * tag[:, :, i]]
            result[:, :, j] = sizes[j] * centered(
                features[:, :, j] * np.log((val / ref).clip(0.55, 1.8))
            ).astype(np.float32)
        return result

    return change


def final_head(j):
    import experiment_y1_price_feedback_v1 as short

    previous = short.load() if j == 0 else short.parent.load()
    change = load_change()

    def adjust(raw, p):
        initial = previous(raw, p)
        q = p.copy()
        q[:, :, j] += initial
        return initial + change(raw, q)[:, :, j]

    return adjust


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
            OUT, ratio.correction, load_final_y2=lambda: final_head(1), load_final_y1=lambda: final_head(0)
        )
