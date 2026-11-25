"""Price relative Y3 by expected pairwise prediction difficulty, with or without realized variation."""

import json
import numpy as np

import experiment_y3_peer_settled_medium_errors_v1 as previous
from experiment_causal_cross_horizon_ridge_v1 import labels

v = previous.v
OUT = previous.OUT.parent / "y3_forecast_matchup_difficulty_v1"


def pairs(x):
    return x[..., :, None] - x[..., None, :]


def training_panels():
    history, cal = [], []
    geom = v.feedback.geometry()
    for d in (4, 5):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        with np.load(previous.tape.OUT / f"day_{d:02d}.npz") as file:
            q = file["prediction"]
        history.append(previous.statistics(raw, q, geom))
        yield d, raw, q, labels(d).reshape(q.shape), geom
    for d, raw, q, _, y, geom, _ in previous.training_panels():
        if d % 2 == 0:
            weights = previous.fit(history)
            size = v.price.sizing(cal)[0] if cal else 0.0
        delta = previous.design(raw, q, np.arange(len(q)), geom, weights)[:, :, 0]
        p = q.copy()
        p[:, :, 2] += size * delta
        yield d, raw, p, y, geom
        cal.append(v.price.rscore(q, y, delta[:, :, None]))
        history.append(previous.statistics(raw, q, geom))


ARMS = (
    "expected correctness with observed amplitude",
    "expected correctness before cash arrives",
    "observed amplitude alone",
    "uncalibrated forecast margins",
    "reassigned receiver",
    "stale expected correctness",
    "no outcome weights",
    "individual forecast magnitude",
)


def events(raw, q, geom):
    k, coef, row = geom
    parts = []
    for a, b in v.price.segments(len(raw)):
        p = q[a:b]
        s = raw[a:b]
        marks = np.arange(20, len(p), 5)
        orig = marks - 20
        path = v.feedback.cash.compound(s, k)[1:]
        gaps = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(s, row)).cumsum(0)))
        ok = gaps[marks + 1] == gaps[orig]
        real = coef[1, 1] * np.expm1(path[marks] - path[orig]) / k
        observed = pairs(real).astype(np.float32)
        forecast = pairs(p[orig, :, 1]).astype(np.float32)
        valid = ok[:, :, None] & ok[:, None, :]
        bins = np.digitize(abs(forecast), [0.10, 0.25, 0.5, 0.9]).astype(np.uint8)
        fw = np.tanh(abs(forecast) / 0.25)
        rw = np.tanh(abs(observed) / 0.4)
        far = p[:, :, 2] - p[:, :, 1] * (coef[2, 1] / coef[1, 1])
        far -= far.mean(1, keepdims=True)
        parts.append((far, marks, forecast, observed, valid, bins, fw, rw))
    return parts


def statistics(parts):
    sums = np.zeros((5, 3))
    for far, marks, p, y, ok, cat, fw, rw in parts:
        mass = fw * rw * ok
        for i, val in enumerate((mass * np.sign(p * y), mass, fw * ok)):
            sums[:, i] += np.bincount(cat.ravel(), weights=val.ravel(), minlength=5)
    return sums / sums[:, 2].sum()


def fit(hist):
    wins, mass, count = np.mean(hist, axis=0).T
    return wins / mass.clip(1e-6), mass / count.clip(1e-6)


def design(parts, state):
    rate, amp = state
    out = []
    for far, marks, p, y, ok, cat, fw, rw in parts:
        mem = np.zeros((50, 8))
        z = np.zeros((*far.shape, len(ARMS)), np.float32)
        saved = []
        pos = 0
        for t in range(0, len(far), 10):
            while pos < len(marks) and marks[pos] <= t:
                idx = cat[pos]
                prob = rate[idx]
                guessed = fw[pos] * amp[idx]
                actual = fw[pos] * rw[pos] * ok[pos]
                guessed = guessed * (~np.eye(50, dtype=bool))
                rawmargin = np.tanh(abs(p[pos]) / 0.5)
                valid = ok[pos]
                values = (
                    actual * prob,
                    actual,
                    guessed * prob,
                    guessed,
                    actual,
                    rawmargin,
                    prob * (~np.eye(50, dtype=bool)),
                    valid,
                )
                row = np.stack([vv.sum(1) / 49 for vv in values], axis=1)
                # Update at observation time, integrating four overlapping schedules with unit total exposure.
                mem = np.exp(-5 / 360.0) * mem + 0.25 * row
                pos += 1

            def dev(x):
                return x - x.mean()

            obs = dev(mem[:, 0] / (2 + mem[:, 1]))
            issued = dev(mem[:, 2] / (2 + mem[:, 3]))
            mass = dev(mem[:, 4] / (2 + mem[:, 7]))
            rawmargin = dev(mem[:, 5] / 20.0)
            plain = dev(mem[:, 6] / 20.0)
            saved.append(obs.copy())
            old = saved[-19] if len(saved) >= 19 else np.zeros(50)
            i = min(len(far), t + 10)
            gates = (obs, issued, mass, rawmargin, np.roll(obs, 13), old, plain)
            for col, gate in enumerate(gates):
                val = far[t:i] * gate
                z[t:i, :, col] = val - val.mean(1, keepdims=True)
            val = far[t:i] * (abs(far[t:i]) - abs(far[t:i]).mean(1, keepdims=True))
            z[t:i, :, -1] = val - val.mean(1, keepdims=True)
        out.append(z)
    return np.concatenate(out)


def train():
    hist, cal, days, offsets = [], [], [], []
    for d, raw, q, y, geom in training_panels():
        parts = events(raw, q, geom)
        if d >= 6:
            if d % 2 == 0:
                state = fit(hist)
                if cal:
                    size = v.price.sizing(cal)
            z = design(parts, state)
            if d >= 8:
                gs = [v.price.rscore(q[i::5], y[i::5], z[i::5] * size) for i in range(5)]
                days.append(sum(gs))
                offsets.append(gs)
                if d == 8:
                    n = 2493
                    np.testing.assert_allclose(
                        z[:n], design(events(raw[:n], q[:n], geom), state), rtol=0, atol=5e-7
                    )
            cal.append(v.price.rscore(q, y, z))
        hist.append(statistics(parts))
        print("day", d + 1, flush=True)
    diff = lambda g: v.price.diff(g)[1:]
    halves = np.stack([diff(sum(rr)) for rr in (days[:4], days[4:])])
    phases = np.array([[diff(sum(gg)) for gg in zip(*rr)] for rr in (offsets[:4], offsets[4:])])
    daily = np.stack([diff(g) for g in days])

    def passes(j):
        return (
            (halves[:, j] > np.maximum(0, halves[:, 4:6].max(1))).all()
            and (daily[:, j] > 0).sum() >= 5
            and (phases[:, :, j] > 0).sum(1).min() >= 4
        )

    pick = 0 if passes(0) else 1 if passes(1) else None
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                arms=ARMS,
                pick=pick,
                halves=halves.tolist(),
                phases=phases.tolist(),
                daily=daily.tolist(),
                rates=fit(hist)[0].tolist(),
            ),
            indent=2,
        )
    )
    if pick is not None:
        rate, amp = fit(hist)
        np.savez(OUT / "model.npz", rate=rate, amp=amp, col=pick, size=v.price.sizing(cal)[pick])
    print("matchup calibration", halves, "positive dates", (daily > 0).sum(0), "picked", pick, flush=True)


def issued_change(q, ratio, rate, amp, size):
    sections = []
    for a, b in v.price.segments(len(q)):
        p = q[a:b]
        far = p[:, :, 2] - p[:, :, 1] * ratio
        far -= far.mean(1, keepdims=True)
        mem = np.zeros((50, 2))
        pos = 20
        out = np.empty_like(far, dtype=np.float32)
        for t in range(0, len(p), 10):
            while pos <= t:
                h = p[pos - 20, :, 1]
                distance = abs(h[:, None] - h[None, :]).astype(np.float32)
                group = np.digitize(distance, [0.10, 0.25, 0.5, 0.9])
                mass = np.tanh(distance / 0.25) * amp[group]
                row = np.stack(((mass * rate[group]).sum(1) / 49, mass.sum(1) / 49), axis=1)
                mem = np.exp(-5 / 360.0) * mem + 0.25 * row
                pos += 5
            gate = mem[:, 0] / (2 + mem[:, 1])
            val = far[t : t + 10] * (gate - gate.mean())
            out[t : t + 10] = val - val.mean(1, keepdims=True)
        sections.append(out)
    return np.concatenate(sections) * size


def load_change():
    with np.load(OUT / "model.npz") as f:
        rate, amp, col, size = f["rate"], f["amp"], int(f["col"]), float(f["size"])
    geom = v.feedback.geometry()
    if col == 1:
        ratio = geom[1][2, 1] / geom[1][1, 1]
        return lambda raw, q: issued_change(q, ratio, rate, amp, size)
    return lambda raw, q: design(events(raw, q, geom), (rate, amp))[:, :, col] * size


def correction():
    old, change = previous.correction(), load_change()

    def revise(raw, p):
        delta = old(raw, p)
        q = p.copy()
        q[:, :, 2] += delta
        return delta + change(raw, q)

    return revise


if __name__ == "__main__":
    import argparse
    import torch
    import experiment_short_medium_variance_ratio_v1 as mixed
    import experiment_y1_source_specific_peer_response_v1 as fast
    import y3_portfolio_walkforward as runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        runner.replay(
            OUT, correction, load_final_y2=lambda: mixed.final_head(1), load_final_y1=fast.final_head
        )
