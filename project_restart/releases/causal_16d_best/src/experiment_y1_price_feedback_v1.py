"""Calibrate the price-only feedback head for Y1 and compare its stock identity with matched controls."""

import argparse
import json
import numpy as np
import experiment_y2_matured_feedback_readout_v1 as parent

feedback = parent.parent
price = feedback.price
OUT = parent.OUT.parent / "y1_price_feedback_v1"
ARMS = ("all lag lengths", "median lag effect", "shared sign")
MODE = ("own stock", "reassigned feedback", "fixed slope")


def training_lags(raw, q, t, geometry):
    k, coeff, row = geometry
    result = []
    for lo, hi in price.segments(len(raw)):
        ix = t[(t >= lo) & (t < hi)] - lo
        if not len(ix):
            continue
        r = raw[lo:hi]
        i = np.arange(len(r))
        levels = feedback.cash.compound(r, k)[1:] / k * coeff[2, 1]
        missing = np.vstack((np.zeros((1, 50), int), (~feedback.cash.present(r, row)).cumsum(0)))
        short = (
            np.log1p(k * (q[lo:hi, :, :2] - coeff[None, None, :2, 0]) / coeff[None, None, :2, 1])
            / k
            * coeff[2, 1]
        )
        at = (i - 5).clip(0)
        values = []
        for h in feedback.LAGS:
            past = (i - h).clip(0)
            valid = (missing[i + 1] == missing[past]) & (i >= h)[:, None]
            ok = valid[at] & (missing[i + 1] == missing[at]) & (i >= h + 5)[:, None]
            obs = feedback.centered(levels - levels[at] - short[at, :, 0], ok)
            px = feedback.centered(levels - levels[past], valid)
            z = np.tanh(px)
            old = z[at] * ok
            slope = (feedback.running(old * obs) / (0.012 + feedback.running(old * old))).clip(-1.5, 1.5)
            values.append(np.stack((z[ix] * slope[ix], z[ix] * np.roll(slope[ix],17,axis=1), z[ix]),2))
        v = np.stack(values, 3)
        result.append((v - v.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(result)


def features(raw, q, t, geom):
    allz = training_lags(raw, q, t, geom)[:, :, [0, 1, 2], :]
    avg = allz.mean(3)
    med = np.median(allz, axis=3)
    agree = np.where(
        np.all(allz * med[:, :, :, None] > 0, axis=3), np.sign(med) * np.min(abs(allz), axis=3), 0
    )
    z = np.stack((avg, med, agree), 3).reshape(*avg.shape[:2], 9)
    return z - z.mean(1, keepdims=True)


def train():
    geom = feedback.geometry()
    cal, days = [], []
    for d, raw, t, q, before, y, *extra in feedback.training_panels():
        z = features(raw, q, t, geom)
        p = before[:, :, [1, 2, 0]]
        truth = y[t][:, :, [1, 2, 0]]
        if d >= 8:
            if d % 2 == 0:
                size = price.sizing(cal)
            days.append(price.rscore(p, truth, z * size))
            if d == 8:
                end = 2461
                keep = t < end
                np.testing.assert_allclose(
                    features(raw[:end], q[:end], t[keep], geom), z[keep], atol=1e-7, rtol=0
                )
        cal.append(price.rscore(p, truth, z))
    h = np.stack([price.diff(sum(days[k : k + 4]))[1:].reshape(3, 3) for k in (0, 4)])
    ds = np.stack([price.diff(g)[1:].reshape(3, 3) for g in days])
    eligible = [
        j
        for j in range(3)
        if np.all(h[:, 0, j] > np.maximum.reduce([np.zeros(2), h[:, 1, j], h[:, 2, j]]))
        and (ds[:, 0, j] > 0).sum() >= 5
    ]
    pick = max(eligible, key=lambda j: min(h[:, 0, j])) if eligible else None
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(arms=ARMS, mode=MODE, halves=h.tolist(), daily=ds.tolist(), pick=pick), indent=2)
    )
    print("Y1 price", h, "pick", pick, flush=True)
    if pick is not None:
        np.savez(OUT / "model.npz", size=price.sizing(cal)[pick], pick=pick)
    return pick


def live(raw, q, t, geom, pick):
    k, coeff, row = geom
    rows = []
    for lo, hi in price.segments(len(q)):
        ix = t[(t >= lo) & (t < hi)] - lo
        if not len(ix):
            continue
        r = raw[lo:hi]
        i = np.arange(len(r))
        past = (i - 5).clip(0)
        level = feedback.cash.compound(r, k)[1:] / k * coeff[2, 1]
        missing = np.vstack((np.zeros((1, 50), int), (~feedback.cash.present(r, row)).cumsum(0)))
        short = np.log1p(k * (q[lo:hi, :, 0] - coeff[0, 0]) / coeff[0, 1]) / k * coeff[2, 1]
        moves = []
        for lag in feedback.LAGS:
            prev = (i - lag).clip(0)
            valid = (missing[i + 1] == missing[prev]) & (i >= lag)[:, None]
            known = valid[past] & (missing[i + 1] == missing[past]) & (i >= lag + 5)[:, None]
            z = np.tanh(feedback.centered(level - level[prev], valid))
            old = z[past] * known
            err = feedback.centered(level - level[past] - short[past], known)
            rate = (feedback.running(old * err) / (0.012 + feedback.running(old * old))).clip(-1.5, 1.5)
            v = z[ix] * rate[ix]
            moves.append((v - v.mean(1, keepdims=True)).astype(np.float32))
        lag_scores = np.stack(moves, 2)
        if pick == 0:
            v = lag_scores.mean(2)
        else:
            median = np.median(lag_scores, axis=2)
            v = (
                median
                if pick == 1
                else np.where(
                    np.all(lag_scores * median[:, :, None] > 0, axis=2),
                    np.sign(median) * np.min(abs(lag_scores), axis=2),
                    0,
                )
            )
        rows.append(v - v.mean(1, keepdims=True))
    return np.concatenate(rows)


def load():
    geom = feedback.geometry()
    with np.load(OUT / "model.npz") as f:
        size, pick = float(f["size"]), int(f["pick"])
    return lambda raw, q: size * live(raw, q, np.arange(len(q)), geom, pick)


if __name__ == "__main__":
    import torch
    import y3_portfolio_walkforward as runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        runner.replay(OUT, feedback.correction, load_final_y2=parent.load, load_final_y1=load)
