"""Pool multiple recent price/forecast revisions, then check fast feedback on disjoint intraday blocks."""

import argparse
import json
import numpy as np
from scipy.signal import lfilter

import experiment_compounded_return_audit_v1 as cash
import experiment_y3_expected_volatility_occupancy_v1 as parent

price = parent.price
LAGS = (8, 15, 30)
OUT = parent.OUT.parent / "y3_pooled_revision_feedback_v1"
MODE = ("individual slopes", "common slope", "agreement across blocks", "above recent forecast change")
FAMILY = ("own", "wrong stock", "fixed", "price only", "forecast only")


def centered(x, valid):
    return (x - (x * valid).sum(1, keepdims=True) / valid.sum(1, keepdims=True).clip(1)) * valid


def running(totals):
    return lfilter([1 / 200], [1, -199 / 200], totals, axis=0)


def geometry():
    with np.load(cash.OUT / "model.npz") as f:
        return float(f["k"]), f["coeff"], f["row"]


def training_panels():
    hist, regimes, magcal, occal = [], [], [], []
    for d, raw, t, q, y, ok, geo in parent.parent.training_panels():
        vol = parent.parent.risk(raw, np.arange(len(q)))
        p = parent.parent.bands(q[t], geo[-1])
        yy = parent.parent.bands(y[t], geo[-1])
        if d >= 6:
            if d % 2 == 0:
                state = parent.parent.fit(hist)
                rates = parent.fit_teacher(regimes, state[0])
            mag = parent.parent.signals(vol[t], q[t], geo[-1], state)[0][:, :, 2]
            occ = parent.signals(vol, q, t, geo[-1], state, rates)[:, :, 3]
            before = q[t].copy()
            if d >= 8:
                if d % 2 == 0:
                    mag_size = price.sizing(magcal)[0]
                    occ_size = price.sizing(occal)[0]
                before[:, :, 2] += mag_size * mag
            occ_stat = price.rscore(before, y[t], occ[:, :, None])
            if d >= 8:
                before[:, :, 2] += occ_size * occ
            yield d, raw, t, q, before, y, ok, geo, vol, state, rates, regimes
            occal.append(occ_stat)
            magcal.append(price.rscore(q[t], y[t], mag[:, :, None]))
        hist.append((vol[t[ok]], p[ok], yy[ok]))
        use = ok.copy()
        for lo, hi in price.segments(len(q)):
            use &= ~((t >= hi - 60) & (t < hi))
        regimes.append((vol, t[use]))


def design(raw, q, t, geom, selected=None):
    k, coeff, row = geom
    parts = []
    for lo, hi in price.segments(len(raw)):
        ix = t[(t >= lo) & (t < hi)] - lo
        if not len(ix):
            continue
        r = raw[lo:hi]
        i = np.arange(len(r))
        at = (i - 5).clip(0)
        level = cash.compound(r, k)[1:] / k * coeff[2, 1]
        missing = np.vstack((np.zeros((1, 50), int), (~cash.present(r, row)).cumsum(0)))
        short = (
            np.log1p(k * (q[lo:hi, :, :2] - coeff[None, None, :2, 0]) / coeff[None, None, :2, 1])
            / k
            * coeff[2, 1]
        )
        observed = level - level[at] - short[at, :, 0]

        def family(legs, wrong=False, flat=False):
            vals = []
            for h in LAGS:
                past = (i - h).clip(0)
                valid = (missing[i + 1] == missing[past]) & (i >= h)[:, None]
                for j in (0, 1):
                    px = level - level[past] if legs != "forecast" else 0.0
                    fc = short[:, :, 0] - short[past, :, j] if legs != "price" else 0.0
                    z = np.tanh(centered(px + fc, valid))
                    ok = valid[at] & (missing[i + 1] == missing[at]) & (i >= h + 5)[:, None]
                    old = z[at] * ok
                    truth = centered(observed, ok)
                    cov = running(old * truth)[ix]
                    variance = running(old * old)[ix]
                    if selected in (0, 3):
                        vals.append(z[ix] * (cov / (variance + 0.012)).clip(-1.5, 1.5))
                        continue
                    # Every completed five-tick interval must belong entirely to one of the two block sets.
                    g = (i % 80 >= 5)[:, None]
                    block = (i // 80 % 2 == 0)[:, None]
                    blocked = [
                        (running(old * truth * g * b)[ix], running(old * old * g * b)[ix])
                        for b in (block, ~block)
                    ]
                    vals.append((z[ix], cov, variance, *blocked))
            if selected in (0, 3):
                return np.mean(vals, 0)
            zs, covs, variances, left, right = zip(*vals)
            z = np.mean(zs, 0)

            def coef(c, v, reg):
                if flat:
                    return np.ones_like(c)
                a = (c / (v + reg)).clip(-1.5, 1.5)
                return np.roll(a, 17, axis=1) if wrong else a

            ind = np.mean([a * coef(c, v, 0.012) for a, c, v in zip(zs, covs, variances)], 0)
            pooled = z * coef(np.mean(covs, 0), np.mean(variances, 0), 0.012)
            slopes = [coef(*[np.mean(u, 0) for u in zip(*group)], 0.006) for group in (left, right)]
            a, b = slopes
            agree = z * np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0)
            return np.stack((ind, pooled, agree), 2)

        own = family("all")
        if selected in (0, 3):
            z = own - family("forecast") if selected == 3 else own
            parts.append((z - z.mean(1, keepdims=True)).astype(np.float32))
            continue
        wrong = family("all", wrong=True)
        flat = family("all", flat=True)
        px = family("price")
        fc = family("forecast")
        wrong_fc = family("forecast", wrong=True)
        flat_fc = family("forecast", flat=True)
        z = np.concatenate(
            [
                np.concatenate((v, (v[:, :, 0] - f)[:, :, None]), 2)
                for v, f in (
                    (own, fc[:, :, 0]),
                    (wrong, wrong_fc[:, :, 0]),
                    (flat, flat_fc[:, :, 0]),
                    (px, fc[:, :, 0]),
                    (fc, np.zeros_like(fc[:, :, 0])),
                )
            ],
            2,
        )
        centered_z = (z - z.mean(1, keepdims=True)).astype(np.float32)
        parts.append(centered_z if selected is None else centered_z[:, :, selected])
    return np.concatenate(parts)


def train():
    geom = geometry()
    cal, days, legs = [], [], []
    for d, raw, t, q, before, y, ok, geo, vol, state, rates, history in training_panels():
        z = design(raw, q, t, geom)
        if d >= 8:
            if d % 2 == 0:
                sizes = price.sizing(cal)
            dz = z * sizes
            days.append(price.rscore(before, y[t], dz))
            legs.append(
                [
                    parent.parent.trajectory.cohort.motion.parent.gram(before, y[t], u, geo[-1])
                    for u in dz[:, :, :4].transpose(2, 0, 1)
                ]
            )
            if d == 8:
                np.testing.assert_allclose(design(raw, q, t, geom, selected=3), z[:, :, 3], rtol=0, atol=1e-7)
                end = 2461
                use = t < end
                np.testing.assert_allclose(
                    design(raw[:end], q[:end], t[use], geom), z[use], atol=2e-6, rtol=0
                )
        cal.append(price.rscore(before, y[t], z))
    h = np.stack([price.diff(sum(days[k : k + 4]))[1:].reshape(5, 4) for k in (0, 4)])
    ds = np.stack([price.diff(g)[1:].reshape(5, 4) for g in days])
    at = [
        [parent.parent.trajectory.cohort.motion.parent.decompose(g) for g in np.sum(legs[k : k + 4], axis=0)]
        for k in (0, 4)
    ]
    valid = [
        j
        for j in range(4)
        if np.all(h[:, 0, j] > np.maximum.reduce([np.zeros(2), h[:, 1, j], h[:, 2, j]]))
        and min(h[:, 0, j]) > max(min(h[:, 3, j]), min(h[:, 4, j]))
        and (ds[:, 0, j] > 0).sum() >= 5
        and all(v[j]["near"] > 0 for v in at)
    ]
    pick = max(valid, key=lambda j: min(h[:, 0, j])) if valid else None
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(mode=MODE, family=FAMILY, halves=h.tolist(), daily=ds.tolist(), legs=at, pick=pick), indent=2
        )
    )
    print("halves", h, "pick", pick, flush=True)
    if pick is not None:
        np.savez(OUT / "model.npz", size=price.sizing(cal)[pick], pick=pick)
    return pick


def load():
    old = parent.load()
    with np.load(OUT / "model.npz") as f:
        size, col = float(f["size"]), int(f["pick"])
    geom = geometry()

    def apply(raw, p):
        return old(raw, p) + size * design(raw, p, np.arange(len(p)), geom, selected=col)

    return apply


def correction():
    prior = parent.parent.trajectory.cohort.motion.current.base.memory.correction()
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
        parent.parent.trajectory.cohort.motion.parent.replay(OUT, correction)
