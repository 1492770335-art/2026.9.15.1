"""Rebuild all-time rolling current-release forecasts while preserving the original chronological fitting schedules."""

import json

import numpy as np

from settings import TAPE as OUT
import experiment_short_medium_variance_ratio_v1 as mixed

import experiment_y2_whole_forecast_matchups_v1 as medium

OUT.mkdir(exist_ok=True, parents=True)


def dense_short():
    warm, cal, hist = [], [], []
    risk_stats = [[], []]
    geom = mixed.feedback.geometry()
    for d, raw, t, q, y, ok, fg in mixed.ratio.prior.training_panels():
        if d >= 6:
            break
        v, st = mixed.ratio.risk_states(raw, t, geom)
        hist.append(
            tuple(a[ok] for a in (v, st, mixed.centered(q[t]), mixed.centered(y[t])))
        )
        warm.append(mixed.source_stats(q[t], y[t], mixed.source(raw, q, geom)[t]))
    for d, raw, t, q, before, y, ok, fg in mixed.ratio.training_panels():
        if d % 2 == 0:
            s = mixed.weights(cal or warm)
            risk = mixed.ratio.fit(hist)
        change = mixed.source(raw, q, geom)
        np.testing.assert_allclose(q[t, :, :2], before[:, :, :2], rtol=0, atol=5e-7)
        p = q.copy()
        p[:, :, :2] += s * change
        sampled = before.copy()
        cal.append(mixed.source_stats(before, y[t], change[t]))
        sampled[:, :, :2] += s * change[t]
        v, st = mixed.ratio.risk_states(raw, np.arange(len(raw)), geom)
        z = mixed.design(v[t], st[t], mixed.centered(sampled), risk)[0]
        hist.append(
            tuple(
                a[ok]
                for a in (v[t], st[t], mixed.centered(sampled), mixed.centered(y[t]))
            )
        )
        if d >= 8 and d % 2 == 0:
            sizes = [mixed.price.sizing(records) for records in risk_stats]
        after = p[:, :, :2].copy()
        if d >= 8:
            for i in range(0, len(p), 512):
                sl = slice(i, i + 512)
                block = mixed.design(v[sl], st[sl], mixed.centered(p[sl]), risk)[0]
                for j, col in enumerate((3, 1)):
                    after[sl, :, j] += sizes[j][col] * block[:, :, col, j]
        for j in range(2):
            risk_stats[j].append(mixed.stat(sampled, y[t], z[:, :, :, j], j))
        with np.load(medium.fast.reference.OUT / f"day_{d:02d}.npz") as source:
            np.testing.assert_allclose(after[t], source["baseline"], rtol=0, atol=8e-7)
        np.save(OUT / f"short_{d:02d}.npy", after)
        print("Verified full-grid short tape", d + 1, flush=True)


def final():
    difficulty, fast = medium.prior, medium.fast
    price = medium.v.price
    fast_pick = json.loads((fast.OUT / "results.json").read_text())["pick"]
    margin_pick = json.loads((difficulty.OUT / "results.json").read_text())["pick"]
    hist, fast_hist, fast_cal, margin_cal, medium_cal = [], [], [], [], []
    errors = []
    for d, raw, q, y, geom in difficulty.training_panels():
        parts = difficulty.events(raw, q, geom)
        with np.load(fast.prior.tape.OUT / f"day_{d:02d}.npz") as f:
            original, t = f["prediction"], f["t"]
        p = q.copy()
        if d >= 6:
            full = np.arange(len(p))
            short = np.load(OUT / f"short_{d:02d}.npy")
            if d % 2 == 0:
                state = difficulty.fit(hist)
                op = fast.prior.axes(fast_hist)
                sf = price.sizing(fast_cal)[fast_pick] if fast_cal else 0.0
                sm = price.sizing(margin_cal)[margin_pick] if margin_cal else 0.0
                sy = price.sizing(medium_cal)[0] if medium_cal else 0.0
            p[:, :, :2] = short
            zx = fast.design(raw, original, full, geom, op)
            fbase = original[t].copy()
            fbase[:, :, :2] = short[t]
            fast_cal.append(fast.mixed.stat(fbase, y[t], zx[t], 0))
            p[:, :, 0] += sf * zx[:, :, fast_pick]
            zm = difficulty.design(parts, state)
            margin_cal.append(medium.v.price.rscore(q, y, zm))
            p[:, :, 2] += sm * zm[:, :, margin_pick]
            zy = medium.proposals(medium.gate(q, state), p)
            medium_cal.append(fast.mixed.stat(p[t], y[t], zy[t], 1))
            p[:, :, 1] += sy * zy[:, :, 0]
        np.save(OUT / f"forecast_{d:02d}.npy", p)
        errors.append(dict(day=d + 1, full_rows=len(p) * 50))
        hist.append(difficulty.statistics(parts))
        fast_hist.append(fast.geometry(raw, original, geom, 0))
        print(errors[-1], flush=True)
    (OUT / "verified.json").write_text(
        json.dumps(
            dict(
                dates=errors,
                release="023d814",
                purpose="Rolling training references only; each head follows its original chronological schedule.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    if not all((OUT / f"short_{d:02d}.npy").is_file() for d in range(6, 16)):
        dense_short()
    final()
