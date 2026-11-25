"""Validate recent market-disagreement forecasts against matured cash before applying them again."""

import json
import numpy as np
from scipy.signal import lfilter

import experiment_y3_variance_ratio_magnitude_v1 as source
import experiment_short_medium_variance_ratio_v1 as mixed

price = source.price
rev = source.feedback
NAXES = 6
OUT = source.OUT.parent / "y3_prequential_disagreement_gate_v1"
ARMS = (
    "medium-model on full outcomes",
    "distal-model on full outcomes",
    "joint votes on full outcomes",
    "medium-model on distal outcomes",
    "distal-model on distal outcomes",
    "joint votes on distal outcomes",
)
MODES = ("recent gradient", "agreement across separate blocks", "misdated evidence", "fixed direction")


def centered(x):
    return x - x.mean(axis=1, keepdims=True)


def update_geometry(q):
    v = centered(q[::8, :, 1].astype(float))
    return (v.T @ v) / len(v)


def factors(history):
    cov = np.mean(history, 0)
    sd = np.sqrt(cov.diagonal().clip(0.0001))
    corr = cov / sd[:, None] / sd[None, :]
    eig, vec = np.linalg.eigh(corr)
    return vec[:, -NAXES:] / sd[:, None] / np.sqrt(eig[-NAXES:].clip(0.05))


def flow(raw, q, t, axis, geom, rho):
    k, coef, row = geom
    outputs = []
    for a, b in price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        r, p = raw[a:b], q[a:b]
        mids = centered(p[:, :, 1].astype(float))
        x = mids @ axis
        shuffled = mids @ np.roll(axis, 17, axis=0)
        x, shuffled = [3 * np.tanh(v / 3) for v in (x, shuffled)]
        obs = np.arange(0, len(r), 8)
        prices = rev.cash.compound(r, k)[1:] / k * coef[2, 1]
        miss = np.vstack((np.zeros((1, 50), int), (~rev.cash.present(r, row)).cumsum(0)))

        def smooth(v):
            return lfilter([1 / 70], [1, -69 / 70], v, axis=0)

        def regression(age, offset):
            issue = (obs - age).clip(0)
            at = (obs - age + offset).clip(0)
            good = miss[obs + 1] == miss[issue]
            count = good.sum(1)
            keep = (count >= 35) & (obs >= age)
            # The source forecast and all price changes are already observable at obs.
            expected = rho * p[issue, :, 1] if age == 20 else p[issue, :, 2] - rho * p[issue, :, 1]
            y = ((prices[obs] - prices[at] - expected) * good).sum(1) / count.clip(1)
            y = (y - smooth(y * keep)).clip(-2.5, 2.5)
            v = x[issue]
            cov = smooth(v[:, :, None] * v[:, None, :] * keep[:, None, None])
            cp = smooth(v * y[:, None] * keep[:, None])
            eye = np.eye(NAXES)[None, :, :]
            scale = 0.8 + 0.5 * cov.diagonal(axis1=1, axis2=2)
            ridge = cov + eye * scale[:, None, :]
            w = np.linalg.solve(ridge, cp[:, :, None])[:, :, 0]
            diag = cp / (cov.diagonal(axis1=1, axis2=2) + scale)
            return w, diag

        near, ndiag = regression(20, 0)
        far, fdiag = regression(60, 20)
        ids = ix // 8
        v = x[ix]

        def bank(v, wn, wf):
            n = (v * wn).sum(1)
            f = (v * wf).sum(1)
            agree = np.where(n * f > 0, np.sign(n) * np.minimum(abs(n), abs(f)), 0)
            # Scale today's medium signal by the fraction of forecast dispersion in the far horizon.
            medium = centered(p[ix, :, 1]) * rho
            distant = centered(p[ix, :, 2]) - medium
            ratio = np.sqrt((0.004 + (distant * distant).mean(1)) / (0.004 + (medium * medium).mean(1))).clip(
                0.4, 2
            )
            return np.column_stack((n, f, ratio * n, agree, f - n, (n + f) / 2))

        old = (ids - 32).clip(0)
        outputs.append(
            np.column_stack(
                (
                    bank(v, near[ids], far[ids]),
                    bank(shuffled[ix], near[ids], far[ids]),
                    bank(v, near[old], far[old]),
                    bank(v, ndiag[ids], fdiag[ids]),
                )
            ).astype(np.float32)
        )
    return np.concatenate(outputs)


def gram(before, y, z):
    # Each scalar is shared by all 50 stocks. Expand only the five necessary cross moments.
    x, target = before[:, :, 2].astype(float), y[:, :, 2].astype(float)
    v = np.stack((np.ones_like(x), x, target), 2)
    scalar = z.astype(float)
    g = np.empty((3 + z.shape[1], 3 + z.shape[1]))
    g[:3, :3] = v.reshape(-1, 3).T @ v.reshape(-1, 3)
    g[:3, 3:] = v.sum(1).T @ scalar
    g[3:, :3] = g[:3, 3:].T
    g[3:, 3:] = 50 * scalar.T @ scalar
    return g


def design(raw, q, t, axis, geom, rho):
    k, coef, row = geom
    result = []
    for a, b in price.segments(len(raw)):
        select = (t >= a) & (t < b)
        if not select.any():
            continue
        # The underlying market model resets at session boundaries in its full-day input.
        ticks = np.arange(a, b, 4)
        z = flow(raw, q, ticks, axis, geom, rho)[:, [0, 1]].astype(float)
        z = np.column_stack((z, z.mean(1)))
        r = raw[a:b]
        p = q[ticks, :, 2].astype(float)
        level = rev.cash.compound(r, k)[1:] / k * coef[2, 1]
        miss = np.vstack((np.zeros((1, 50), int), (~rev.cash.present(r, row)).cumsum(0)))
        now = ticks - a
        issue = (np.arange(len(ticks)) - 15).clip(0)
        valid = (now >= 60)[:, None] & (miss[now + 1] == miss[now[issue]])
        count = valid.sum(1).clip(1)
        ok = count >= 35

        def avg(arr):
            return (arr * valid).sum(1) / count

        def moments(y):
            pp = p[issue]
            # Include the full panel moment to project off the incumbent's global Pearson slope.
            zz = z[issue]
            return np.column_stack(
                (
                    avg(pp),
                    avg(y),
                    avg(pp * pp),
                    avg(pp * y),
                    zz,
                    zz * avg(pp)[:, None],
                    zz * avg(y)[:, None],
                    zz**2,
                )
            )

        def fit(v, weights):
            def ew(x):
                return lfilter([1 / 180], [1, -179 / 180], x, axis=0)

            mass = ew(weights)
            g = ew(v * weights[:, None]) / (mass[:, None] + 0.0001)
            xp, yt, pp, py = g[:, :4].T
            zz, zp, zy, z2 = np.split(g[:, 4:], 4, axis=1)
            beta = ((py - xp * yt) / (0.016 + pp - xp * xp)).clip(-0.5, 2.5)
            cz = zy - zz * yt[:, None] - (zp - zz * xp[:, None]) * beta[:, None]
            # A diffuse zero prior limits leverage when very few independent future windows have matured.
            return (0.46 * cz / (0.035 + (z2 - zz * zz).clip(0)) * (mass / (mass + 0.3))[:, None]).clip(
                -1.5, 1.5
            )

        full = level[now] - level[now[issue]]
        far = level[now] - level[(now - 40).clip(0)]
        wi = np.arange(len(ticks))
        slots = np.searchsorted(ticks, t[select], side="right") - 1
        states = []
        for label in (full, far):
            v = moments(label)
            w = fit(v, ok)
            x, y = [fit(v, ok * ((wi // 36) % 2 == i)) for i in (0, 1)]
            paired = np.where(x * y > 0, np.sign(x) * np.minimum(abs(x), abs(y)), 0)
            states.append((w[slots], paired[slots], w[(slots - 100).clip(0)], np.ones_like(w[slots])))
        live = flow(raw, q, t[select], axis, geom, rho)[:, [0, 1]].astype(float)
        live = np.column_stack((live, live.mean(1)))
        out = np.column_stack([v * live for mode in zip(*states) for v in mode])
        result.append(out.astype(np.float32))
    return np.concatenate(result)


def train():
    geo = rev.geometry()
    history, cal, days = [], [], []
    for d, raw, t, q, y, ok, fg in source.prior.training_panels():
        if d >= 6:
            break
        history.append(update_geometry(q))
    for d, raw, t, q, before, y, ok, fg in source.training_panels():
        if d % 2 == 0:
            axes = factors(history)
        z = design(raw, q, t, axes, geo, fg[-1])
        if d >= 8:
            if d % 2 == 0:
                size = price.sizing(cal)
            days.append(gram(before, y[t], z * size))
            if d == 8:
                cut = 2481
                use = t < cut
                np.testing.assert_allclose(
                    z[use], design(raw[:cut], q[:cut], t[use], axes, geo, fg[-1]), atol=3e-7, rtol=0
                )
        cal.append(gram(before, y[t], z))
        history.append(update_geometry(q))
    h = np.stack([price.diff(sum(v))[1:].reshape(4, 6) for v in (days[:4], days[4:])])
    ds = np.stack([price.diff(v)[1:].reshape(4, 6) for v in days])
    pool = [(i, j) for i in (0, 1) for j in range(6) if h[0, i, j] > max(0, h[0, 2:, j].max())]
    first = max(pool, key=lambda p: h[0, p[0], p[1]]) if pool else None
    pick = None
    if first is not None:
        i, j = first
        if h[1, i, j] > max(0, h[1, 2:, j].max()) and (ds[:, i, j] > 0).sum() >= 5:
            pick = first
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(arms=ARMS, modes=MODES, first=first, pick=pick, halves=h.tolist(), daily=ds.tolist()),
            indent=2,
        )
    )
    if pick is not None:
        j = 6 * pick[0] + pick[1]
        np.savez(OUT / "model.npz", pick=j, axis=factors(history), size=price.sizing(cal)[j], rho=fg[-1])
    print("prequential calibration", h[:, :2], "first", first, "selected", pick, flush=True)


def load_change():
    geom = rev.geometry()
    with np.load(OUT / "model.npz") as f:
        axis, size, j, rho = f["axis"], float(f["size"]), int(f["pick"]), float(f["rho"])

    def apply(raw, q):
        return size * design(raw, q, np.arange(len(q)), axis, geom, rho)[:, j, None]

    return apply


def correction():
    upstream = source.prior.trajectory.cohort.motion.current.base.memory.correction()
    feedback = rev.load()
    calibrate_size = source.load_change()
    apply_change = load_change()

    def update(raw, p):
        initial = upstream(raw, p)
        q = p.copy()
        q[:, :, 2] += initial
        first = feedback(raw, q)
        before = q.copy()
        before[:, :, 2] += first
        previous = first + calibrate_size(raw, before)
        before = q.copy()
        before[:, :, 2] += previous
        return initial + (previous + apply_change(raw, before))

    return update


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
            correction,
            load_final_y2=lambda: mixed.final_head(1),
            load_final_y1=lambda: mixed.final_head(0),
        )
