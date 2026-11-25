"""Interpolate risk-conditioned return magnitudes with the causal probability of future high volatility."""

import argparse
import json

import numpy as np

import experiment_y3_incremental_magnitude_ratio_v1 as parent

price = parent.price
OUT = parent.OUT.parent / "y3_expected_volatility_occupancy_v1"


def occupancy(vol, t, threshold):
    category = []
    outcome = []
    for lo, hi in price.segments(len(vol)):
        u = t[(lo <= t) & (t < hi)] - lo
        if not len(u):
            continue
        assert np.max(u) + 60 < hi - lo
        flag = vol[lo:hi] > threshold
        ticks = np.arange(len(flag))[:, None]
        changed = np.vstack((np.ones_like(flag[:1]), flag[1:] != flag[:-1]))
        age = ticks - np.maximum.accumulate(np.where(changed, ticks, 0), axis=0)
        dev = np.log((vol[lo + u] + 0.015) / (threshold + 0.015))
        # Forecast high-state occupancy from the current margin and the ongoing episode's age.
        category.append(
            np.digitize(dev, [-0.6, -0.24, -0.08, 0, 0.08, 0.24, 0.6]) * 4 + np.digitize(age[u], [8, 32, 96])
        )
        prefix = np.concatenate((np.zeros((1, 50)), flag.cumsum(0)))
        outcome.append((prefix[u + 61] - prefix[u + 21]) / 40)
    return np.concatenate(category), np.concatenate(outcome)


def fit_teacher(records, threshold):
    counts = np.zeros(32)
    total = np.zeros(32)
    for vol, t in records:
        c, target = occupancy(vol, t, threshold)
        counts += np.bincount(c.ravel(), minlength=32)
        total += np.bincount(c.ravel(), weights=target.ravel(), minlength=32)
    return (total + 75) / (counts + 150)


ARMS = (
    "current magnitude model",
    "expected future occupancy",
    "adjust distal ratio",
    "adjust near ratio",
    "transition adjustment across horizons",
    "changes in the first 60 ticks of a regime",
    "reassigned occupancy",
)


def past(vol, t, ref):
    category = []
    age = []
    for lo, hi in price.segments(len(vol)):
        u = t[(t >= lo) & (t < hi)] - lo
        if not len(u):
            continue
        flag = vol[lo:hi] > ref
        ticks = np.arange(len(flag))[:, None]
        changed = np.vstack((np.ones_like(flag[:1]), flag[1:] != flag[:-1]))
        elapsed = (ticks - np.maximum.accumulate(np.where(changed, ticks, 0), axis=0))[u]
        margin = np.log((vol[lo + u] + 0.015) / (ref + 0.015))
        category.append(
            np.digitize(margin, [-0.6, -0.24, -0.08, 0, 0.08, 0.24, 0.6]) * 4
            + np.digitize(elapsed, [8, 32, 96])
        )
        age.append(elapsed)
    return np.concatenate(category), np.concatenate(age)


def signals(vol, q, t, ratio, state, rates):
    ref, scale, tables = state
    p = parent.bands(q[t], ratio)
    c, age = past(vol, t, ref)
    prob = rates[c]
    cur = vol[t] > ref
    idx = np.digitize(abs(p) / scale, [0.45, 1.15, 2.0])

    def mix(pair, w):
        low = pair[np.arange(50)[None, :, None], 0, idx, np.arange(3)]
        high = pair[np.arange(50)[None, :, None], 1, idx, np.arange(3)]
        return low + w[:, :, None] * (high - low)

    tab, broad = tables[0]
    refm = mix(broad, cur)

    def delta(weight):
        return p * np.log(mix(tab, weight) / refm).clip(-0.65, 0.65)

    actual = delta(cur)
    expected = delta(prob)
    diff = expected - actual
    fake = delta(np.roll(prob, 19, axis=1)) - actual
    z = np.stack(
        (
            expected[:, :, 2],
            diff[:, :, 2],
            diff[:, :, 1],
            diff[:, :, 1] + diff[:, :, 2],
            diff[:, :, 2] * (age < 60),
            fake[:, :, 2],
        ),
        axis=2,
    )
    return (z - z.mean(1, keepdims=True)).astype(np.float32)


def train():
    hist, regimes, oldcal, cal, days, legs = [], [], [], [], [], []
    for d, raw, t, q, y, ok, geo in parent.training_panels():
        vol = parent.risk(raw, np.arange(len(q)))
        p = parent.bands(q[t], geo[-1])
        yy = parent.bands(y[t], geo[-1])
        if d >= 6:
            if d % 2 == 0:
                state = parent.fit(hist)
                rates = fit_teacher(regimes, state[0])
            old = parent.signals(vol[t], q[t], geo[-1], state)[0][:, :, 2]
            z = signals(vol, q, t, geo[-1], state, rates)
            previous = q[t].copy()
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                    oldsize = price.sizing(oldcal)[0]
                previous[:, :, 2] += oldsize * old
                dz = size * z
                days.append(price.rscore(previous, y[t], dz))
                legs.append(
                    [
                        parent.trajectory.cohort.motion.parent.gram(previous, y[t], a, geo[-1])
                        for a in dz.transpose(2, 0, 1)
                    ]
                )
                if d == 8:
                    take = t < 2460
                    short = signals(
                        parent.risk(raw[:2460], np.arange(2460)), q[:2460], t[take], geo[-1], state, rates
                    )
                    np.testing.assert_allclose(short, z[take], rtol=0, atol=3e-6)
            cal.append(price.rscore(previous, y[t], z))
            oldcal.append(price.rscore(q[t], y[t], old[:, :, None]))
        hist.append((vol[t[ok]], p[ok], yy[ok]))
        use = ok.copy()
        for lo, hi in price.segments(len(q)):
            use &= ~((t >= hi - 60) & (t < hi))
        regimes.append((vol, t[use]))
    h = np.stack([price.diff(sum(v)) for v in (days[:4], days[4:])])
    ds = np.stack([price.diff(v) for v in days])
    attr = [
        [parent.trajectory.cohort.motion.parent.decompose(g) for g in np.sum(v, axis=0)]
        for v in (legs[:4], legs[4:])
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
        json.dumps(dict(arms=ARMS, selected=pick, halves=h.tolist(), daily=ds.tolist(), legs=attr), indent=2)
    )
    print(h, "selected", pick, flush=True)
    if pick:
        st = parent.fit(hist)
        np.savez(
            OUT / "model.npz",
            rates=fit_teacher(regimes, st[0]),
            size=price.sizing(cal)[pick - 1],
            pick=pick - 1,
        )
    return pick


def load():
    old = parent.trajectory.load()
    with np.load(parent.OUT / "model.npz") as f:
        st = tuple(f[k] for k in ("r", "scale", "table"))
        ratio, oldsize = float(f["ratio"]), float(f["size"])
        assert int(f["pick"]) == 2
    with np.load(OUT / "model.npz") as f:
        rates, size, pick = f["rates"], float(f["size"]), int(f["pick"])

    def apply(raw, p):
        before = old(raw, p)
        q = p.copy()
        q[:, :, 2] += before
        t = np.arange(len(q))
        vol = parent.risk(raw, t)
        earlier = oldsize * parent.signals(vol, q, ratio, st)[0][:, :, 2]
        z = signals(vol, q, t, ratio, st, rates)
        return before + earlier + size * z[:, :, pick]

    return apply


def correction():
    prior = parent.trajectory.cohort.motion.current.base.memory.correction()
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
        parent.trajectory.cohort.motion.parent.replay(OUT, correction)
