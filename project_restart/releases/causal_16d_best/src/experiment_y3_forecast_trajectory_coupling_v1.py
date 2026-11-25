"""Match disjoint stock baskets on the entire recent short-forecast trajectory with balanced peer weights."""

import argparse
import json

import numpy as np
from scipy.stats import rankdata
from scipy.signal import lfilter

import experiment_y3_pressure_cohort_projection_v1 as cohort

price = cohort.price
OUT = cohort.OUT.parent / "y3_forecast_trajectory_coupling_v1"
LEFT, RIGHT = np.random.default_rng(97421).permutation(50).reshape(2, 25)
ARMS = (
    "incumbent",
    "current peers",
    "entire matched path",
    "path added to current peers",
    "displacement since matching",
    "unbalanced path matching",
    "reassigned older path",
    "old peers only",
)


def distance(x):
    dx = x[:, LEFT, None] - x[:, None, RIGHT]
    return np.mean(dx**2, axis=3)


def matrix(cost, balanced=True):
    kernel = np.exp(-cost / 0.042).clip(0.0002)
    row = kernel / kernel.sum(2, keepdims=True)
    col = kernel / kernel.sum(1, keepdims=True)
    if not balanced:
        return row, col
    for _ in range(14):
        row /= row.sum(1, keepdims=True)
        row /= row.sum(2, keepdims=True)
    return row, row / row.sum(1, keepdims=True)


def spread(x, w):
    row, col = w
    z = np.empty_like(x)
    z[:, LEFT] = x[:, LEFT] - row @ x[:, RIGHT]
    z[:, RIGHT] = x[:, RIGHT] - col.transpose(0, 2, 1) @ x[:, LEFT]
    return z - z.mean(1, keepdims=True)


def design(raw, q, t, geo, y=None):
    rows = []
    diagnostics = np.zeros((5, 3))
    k, sd, axis, ratio = geo
    for lo, hi in price.segments(len(q)):
        u = t[(lo <= t) & (t < hi)] - lo
        if not len(u):
            continue
        part = raw[lo:hi]
        level, m = cohort.motion.parent.score_inputs(part, k, sd, axis)
        r = abs(level - level[(np.arange(len(level)) - 5).clip(0)]).clip(0, 10)
        vol = lfilter([0.02], [1, -0.98], r, axis=0)
        vals = np.concatenate((q[lo:hi, :, :2], vol[:, :, None]), axis=2)
        ranks = (rankdata(vals, axis=1) / 50).astype(np.float32)
        book = cohort.motion.parent.pressure_fields(part)
        slow = lfilter([0.05], [1, -0.95], book - book[:1], axis=0) + book[:1]
        for k0 in range(0, len(u), 80):
            ix = u[k0 : k0 + 80]
            now = ranks[ix]
            old = ranks[(ix - 12).clip(0)]
            oldest = ranks[(ix - 48).clip(0)]
            prior = 0.6 * distance(old) + 0.4 * distance(oldest)
            cost = distance(now)
            wrong = distance(np.roll(old, 17, axis=1)) * 0.6 + distance(np.roll(oldest, 17, axis=1)) * 0.4
            cur = matrix(cost)
            path = matrix(0.5 * (cost + prior))
            then = matrix(prior)
            free = matrix(0.5 * (cost + prior), False)
            fake = matrix(0.5 * (cost + wrong))
            px = q[lo + ix]
            p = px - px.mean(1, keepdims=True)
            v = np.concatenate((book[ix], slow[ix], m[ix, :, 1:]), 2).astype(np.float32)
            current = spread(v, cur)
            matched = spread(v, path)
            left = spread(v, then)
            z = np.concatenate(
                (p, np.tanh(3 * p), current, matched, left - current, spread(v, free), spread(v, fake), left),
                2,
            )
            rows.append(z.astype(np.float32))
            if y is not None:
                good = ix < hi - lo - 60
                yy = y[lo + ix]
                late = yy[:, :, 2] - ratio * yy[:, :, 1]
                edge = (late[:, LEFT, None] - late[:, None, RIGHT]) ** 2
                for h, w in enumerate((cur, path, then, free, fake)):
                    # Each observation contributes its probability-weighted pairwise target distance.
                    diagnostics[h] += [
                        (edge[good] * w[0][good]).sum(),
                        (cost[good] * w[0][good]).sum(),
                        w[0][good].sum(),
                    ]
    return np.concatenate(rows), diagnostics


def fit(hist):
    mats = np.stack(hist)
    g = mats.mean(0)[:-2, :-2]
    xy = mats[:, :-2, -2:]
    cur, matched, displace, free, fake, old = np.split(np.arange(6, len(g)), 6)
    out = []
    for j, use in enumerate((cur, matched, matched, displace, free, fake, old)):
        controls = np.arange(6) if j < 2 else np.r_[np.arange(6), cur]
        b = np.eye(len(g))[:, use]
        a = g[np.ix_(controls, controls)]
        b[controls] = -np.linalg.solve(
            a + 0.04 * np.diag(a.diagonal().clip(0.00001)), g[np.ix_(controls, use)]
        )
        cov = b.T @ g @ b
        daily = np.einsum("dhk,hi->dik", xy, b)
        mu = daily.mean(0)
        mu *= abs(mu) / (abs(mu) + daily.std(0) / np.sqrt(len(hist)) + 0.00001)
        full, late = mu.T
        stable = np.where(full * late > 0, np.sign(full) * np.minimum(abs(full), abs(late)), 0)
        out.append(b @ np.linalg.solve(cov + 0.9 * np.diag(cov.diagonal().clip(0.00001)), stable))
    return np.stack(out, axis=1).astype(np.float32)


def training_panels():
    cohort_hist, cohort_cal = [], []
    for d, raw, t, q, y, ok, geo in cohort.training_panels():
        v = cohort.design(raw, q, t, *geo)
        baseline = q[t].copy()
        if d >= 6:
            if d % 2 == 0:
                w = cohort.fit(cohort_hist)[:, 1]
            step = v @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cohort_cal)[0]
                baseline[:, :, 2] += size * step
            cohort_cal.append(price.rscore(q[t], y[t], step[:, :, None]))
        cohort_hist.append(cohort.motion.parent.stat(v[ok], q[t[ok]], y[t[ok]], geo[-1]))
        q[t] = baseline
        yield d, raw, t, q, y, ok, geo


def train():
    hist, days, cal, tracking, terms = [], [], [], [], []
    for d, raw, t, q, y, ok, geo in training_panels():
        x, e = design(raw, q, t, geo, y if d >= 8 else None)
        if d >= 6:
            if d % 2 == 0:
                w = fit(hist)
            z = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                dz = z * size
                days.append(price.rscore(q[t], y[t], dz))
                terms.append(
                    [cohort.motion.parent.gram(q[t], y[t], dz[:, :, j], geo[-1]) for j in range(dz.shape[2])]
                )
                tracking.append(e)
                if d == 8:
                    n = 2517
                    np.testing.assert_allclose(
                        x[t < n], design(raw[:n], q[:n], t[t < n], geo)[0], rtol=0, atol=2e-6
                    )
            cal.append(price.rscore(q[t], y[t], z))
        hist.append(cohort.motion.parent.stat(x[ok], q[t[ok]], y[t[ok]], geo[-1]))
    h = np.stack([price.diff(sum(f)) for f in (days[:4], days[4:])])
    ds = np.stack([price.diff(f) for f in days])
    legs = [
        [cohort.motion.parent.decompose(g) for g in np.sum(part, axis=0)] for part in (terms[:4], terms[4:])
    ]
    distances = [(m[:, :2] / m[:, 2:]).tolist() for m in (sum(tracking[:4]), sum(tracking[4:]))]
    eligible = [
        j
        for j in (2, 3, 4, 7)
        if h[:, j].min() > max(0, h[:, 6].min())
        and (ds[:, j] > 0).sum() >= 5
        and all(row[j - 1]["late"] > 0 for row in legs)
    ]
    pick = max(eligible, key=lambda j: h[:, j].min()) if eligible else 0
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                selected=pick,
                arms=ARMS,
                halves=h.tolist(),
                daily=ds.tolist(),
                legs=legs,
                weighted_pair_mse=distances,
                tracking_rows=["current", "path", "past", "unbalanced", "shuffled"],
                tracking_cols=["future late payoff", "current rank state"],
            ),
            indent=2,
        )
    )
    print(h, "selected", pick, "match losses", distances, flush=True)
    if pick:
        np.savez(
            OUT / "model.npz",
            k=geo[0],
            sd=geo[1],
            axis=geo[2],
            ratio=geo[3],
            w=fit(hist)[:, pick - 1] * price.sizing(cal)[pick - 1],
        )
    return pick


def load():
    old = cohort.load()
    with np.load(OUT / "model.npz") as f:
        geo = (float(f["k"]), f["sd"], f["axis"], float(f["ratio"]))
        w = f["w"]

    def apply(raw, p):
        before = old(raw, p)
        q = p.copy()
        q[:, :, 2] += before
        return before + design(raw, q, np.arange(len(q)), geo)[0] @ w

    return apply


def correction():
    prior = cohort.motion.current.base.memory.correction()
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
        cohort.motion.parent.replay(OUT, correction)
