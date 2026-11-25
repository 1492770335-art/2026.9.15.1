"""Use forecasts for the unexpired part of a return to test revision skill before full maturity."""

import json


import numpy as np
from scipy.signal import lfilter

import experiment_y3_intraday_vintage_orientation_v1 as prior

v = prior.v
OUT = prior.OUT.parent / "y3_partial_maturity_orientation_v1"


def design(raw, q, t, geom):
    k, co, row = geom
    rho = co[2, 1] / co[1, 1]
    out = []
    for a, b in v.price.segments(len(raw)):
        ii = t[(t >= a) & (t < b)] - a
        if not len(ii):
            continue
        ticks = np.arange(0, b - a, 4)
        u = np.union1d(ticks, ii)
        f = v.features(raw[a:b], q[a:b], u, geom).reshape(len(u), 50, len(v.LAGS), 4)
        basis = np.stack(
            (
                f[:, :, :3, 0].mean(2),
                f[:, :, 3:, 0].mean(2),
                f[:, :, :3, 1].mean(2),
                f[:, :, 3:, 1].mean(2),
                (f[:, :, :, 0] - f[:, :, :, 2]).mean(2),
            ),
            axis=2,
        )
        z = basis[np.searchsorted(u, ticks)]
        query = basis[np.searchsorted(u, ii)]
        p = q[a + ticks]
        term = p[:, :, 1] * rho
        level = v.feedback.cash.compound(raw[a:b], k)
        missing = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(raw[a:b], row)).cumsum(0)))
        idx = np.arange(len(ticks))

        def observe(h, teacher):
            issue = (idx - h // 4).clip(0)
            source = ticks[issue]
            ok = (ticks >= h)[:, None] & (missing[ticks + 1] == missing[source])
            count = ok.sum(1).clip(1)
            y = co[2, 1] * np.expm1(level[ticks + 1] - level[source + 1]) / k
            if teacher:
                y += term + (40 - h) / 40 * (p[:, :, 2] - term)

            def avg(x):
                return (x * ok[:, :, None]).sum(1) / count[:, None]

            def center(x):
                return x - avg(x)[:, None, :]

            x, base, response = center(z[issue]), center(p[issue, :, 2:]), center(y[:, :, None])
            moments = np.column_stack(
                (avg(base * base), avg(base * response), avg(x * base), avg(x * response), avg(x * x))
            )

            def slope(mask):
                mask = mask & (count >= 32)
                lam = 1 / 75
                mass = lfilter([lam], [1, lam - 1], mask.astype(float))
                g = lfilter([lam], [1, lam - 1], moments * mask[:, None], axis=0) / (mass[:, None] + 0.00001)
                xp, xy, xx = np.split(g[:, 2:], 3, axis=1)
                grad = xy - xp * (g[:, 1] / (g[:, 0] + 0.0001)).clip(0, 3)[:, None]
                return (0.35 * grad / (0.014 + xx) * mass[:, None] / (0.35 + mass[:, None])).clip(-2, 2)

            s0, s1 = [slope((idx // 20) % 2 == j) for j in range(2)]
            return np.where(s0 * s1 > 0, np.sign(s0) * np.minimum(abs(s0), abs(s1)), 0)

        early, mid = [observe(h, True) for h in (20, 40)]
        measured = observe(20, False)
        old = observe(60, False)
        agreed = np.where(early * mid > 0, np.sign(early) * np.minimum(abs(early), abs(mid)), 0)
        residual = agreed - old
        states = (early, mid, agreed, residual, measured, np.roll(agreed, 31, axis=1))
        out.append(
            np.stack([(query * s[ii // 4, None, :]).mean(2) for s in states], axis=2).astype(np.float32)
        )
    return np.concatenate(out)


def train():
    history, parent_cal, cal, days = [], [], [], []
    geom = v.feedback.geometry()
    from experiment_causal_cross_horizon_ridge_v1 import labels

    for d in range(4, 16):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        y = labels(d).reshape(-1, 50, 3)
        with np.load(v.TAPE / f"day_{d:02d}.npz") as f:
            t = f["t"]
            q = np.concatenate((f["short"], f["dense"][:, :, None]), axis=2)
        if d >= 6:
            if d % 2 == 0:
                w = v.fit(history)
            parent_shift = prior.design(raw, q, t, geom, w)[:, :, 3:4]
            parent_stat = v.price.rscore(q[t], y[t], parent_shift)
            if d >= 8:
                if d % 2 == 0:
                    oldsize = v.price.sizing(parent_cal)[0]
                full = prior.design(raw, q, np.arange(len(q)), geom, w)[:, :, 3]
                q = q.copy()
                q[:, :, 2] += oldsize * full
            parent_cal.append(parent_stat)
            z = design(raw, q, t, geom)
            if d >= 8:
                if d % 2 == 0:
                    size = v.price.sizing(cal)
                days.append(v.price.rscore(q[t], y[t], z * size))
                if d == 8:
                    n = 2507
                    use = t < n
                    np.testing.assert_allclose(
                        z[use], design(raw[:n], q[:n], t[use], geom), atol=3e-7, rtol=0
                    )
            cal.append(v.price.rscore(q[t], y[t], z))
        # The previous readout's feature weights are fitted on its own input.
        with np.load(v.TAPE / f"day_{d:02d}.npz") as f:
            q[:, :, 2] = f["dense"]
        valid = ((t < 2401 - 60) | (t >= 2401)) & (t < len(q) - 60)
        x = v.features(raw, q, t[valid], geom)
        history.append(v.moments(x, q[t[valid], :, 2], y[t[valid], :, 2]))
    half = np.stack([v.price.diff(sum(g))[1:] for g in (days[:4], days[4:])])
    daily = np.stack([v.price.diff(g)[1:] for g in days])
    first = int(np.argmax(half[0, :4]))
    pick = (
        first
        if (half[:, first] > np.maximum(0, half[:, 4:].max(1))).all() and (daily[:, first] > 0).sum() >= 5
        else None
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(first=first, pick=pick, half=half.tolist(), daily=daily.tolist()), indent=2)
    )
    if pick is not None:
        np.savez(OUT / "model.npz", size=v.price.sizing(cal)[pick], pick=pick)
    print("partial maturity", half, "selected", pick, flush=True)


def load_change():
    with np.load(OUT / "model.npz") as f:
        size, col = float(f["size"]), int(f["pick"])
    geom = v.feedback.geometry()
    return lambda raw, q: size * design(raw, q, np.arange(len(q)), geom)[:, :, col]


def correction():
    previous, change = prior.correction(), load_change()

    def apply(raw, p):
        shift = previous(raw, p)
        q = p.copy()
        q[:, :, 2] += shift
        return shift + change(raw, q)

    return apply


if __name__ == "__main__":
    import argparse
    import torch
    import y3_portfolio_walkforward as runner
    import experiment_short_medium_variance_ratio_v1 as mixed

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
