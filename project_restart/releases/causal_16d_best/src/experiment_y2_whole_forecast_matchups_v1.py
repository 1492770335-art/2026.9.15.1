"""Recalibrate the whole medium forecast with stock-relative difficulty on independently shifted issuance schedules."""

import json

import numpy as np

import experiment_y1_source_specific_peer_response_v1 as fast
import experiment_y3_forecast_matchup_difficulty_v1 as prior

v = prior.v
OUT = prior.OUT.parent / "y2_whole_forecast_matchups_v1"
ARMS = (
    "mean of two shifted schedules",
    "first schedule",
    "second schedule",
    "wrong receivers",
    "stale gate",
    "global forecast extremeness",
    "market part instead",
)


def gate(q, state):
    # Shift the origin grid within each continuous session and retain both estimates.
    parts = []
    for a, b in v.price.segments(len(q)):
        p = q[a:b, :, 1]
        n = len(p)
        gates = []
        for shift in (1, 3):
            weight = np.zeros((50, 2))
            pos = 20 + shift
            dst = np.zeros((n, 50), np.float32)
            for t in range(0, n, 10):
                while pos <= t:
                    h = p[pos - 20]
                    d = abs(h[:, None] - h[None, :])
                    bins = np.digitize(d, [0.10, 0.25, 0.50, 0.90])
                    prob, amp = state
                    mass = np.tanh(d / 0.25) * amp[bins]
                    obs = np.stack(((mass * prob[bins]).sum(1) / 49, mass.sum(1) / 49), axis=1)
                    weight = np.exp(-5 / 360.0) * weight + 0.25 * obs
                    pos += 5
                rate = weight[:, 0] / (2 + weight[:, 1])
                dst[t : t + 10] = rate - rate.mean()
            gates.append(dst)
        x, y = gates
        avg = (x + y) * 0.5
        parts.append(np.stack((avg, x, y, np.roll(avg, 13, axis=1), avg[(np.arange(n) - 160).clip(0)]), 2))
    return np.concatenate(parts)


def proposals(w, p):
    latest = p[:, :, 1]
    avg = latest.mean(1, keepdims=True)
    rel = latest - avg
    z = np.concatenate(
        (
            rel[:, :, None] * w,
            ((abs(rel) - abs(rel).mean(1, keepdims=True)) * rel)[:, :, None],
            (avg * w[:, :, 0])[:, :, None],
        ),
        axis=2,
    )
    return z - z.mean(1, keepdims=True)


def train():
    hist, cal, days = [], [], []
    for d, raw, q, y, geom in prior.training_panels():
        parts = prior.events(raw, q, geom)
        if d >= 6:
            if d % 2 == 0:
                state = prior.fit(hist)
                if cal:
                    size = v.price.sizing(cal)
            with np.load(fast.reference.OUT / f"day_{d:02d}.npz") as f:
                t = f["t"]
                p = q[t].copy()
                p[:, :, :2] = f["baseline"]
            w = gate(q, state)
            z = proposals(w[t], p)
            if d >= 8:
                days.append(fast.mixed.stat(p, y[t], z * size, 1))
                if d == 8:
                    n = 2491
                    np.testing.assert_array_equal(w[:n], gate(q[:n], state))
            cal.append(fast.mixed.stat(p, y[t], z, 1))
        hist.append(prior.statistics(parts))
    diff = lambda g: v.price.diff(g)[1:]
    halves = np.stack([diff(sum(gg)) for gg in (days[:4], days[4:])])
    daily = np.stack([diff(g) for g in days])
    pick = (
        0
        if (
            (halves[:, 0] > np.maximum(0, halves[:, 3])).all()
            and (halves[:, 1:3] > 0).all()
            and (daily[:, 0] > 0).sum() >= 6
            and halves.mean(0)[0] > halves.mean(0)[4]
        )
        else None
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(arms=ARMS, pick=pick, halves=halves.tolist(), daily=daily.tolist(), stride=5), indent=2
        )
    )
    if pick is not None:
        rate, amp = prior.fit(hist)
        np.savez(OUT / "model.npz", rate=rate, amp=amp, size=v.price.sizing(cal)[0])
    print(
        "Y2 shifted matchup gates", halves, "positive days", (daily[:, 0] > 0).sum(), "pick", pick, flush=True
    )


def load_change():
    with np.load(OUT / "model.npz") as f:
        state = (f["rate"], f["amp"])
        size = float(f["size"])

    def adjust(q, final):
        weight = gate(q, state)[:, :, 0]
        medium = final[:, :, 1]
        z = (medium - medium.mean(1, keepdims=True)) * weight
        return (z - z.mean(1, keepdims=True)) * size

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
            OUT,
            prior.correction,
            load_final_y2=lambda: fast.mixed.final_head(1),
            load_final_y1=fast.final_head,
            load_post_y2=load_change,
        )
