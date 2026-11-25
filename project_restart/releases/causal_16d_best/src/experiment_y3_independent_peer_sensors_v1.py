"""Use two disjoint stock baskets as independent observations of a common leading signal."""

import json
import numpy as np
from scipy.signal import lfilter
import build_y3_revision_transport_training_tape_v1 as tape

v = tape.v
OUT = tape.OUT.parent / "y3_independent_peer_sensors_v1"
GROUPS = np.array_split(np.random.default_rng(84073).permutation(50), 3)


def source(raw, q, geom):
    k, co, row = geom
    level = v.feedback.cash.compound(raw, k)
    miss = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(raw, row)).cumsum(0)))
    t = np.arange(len(raw))
    o = (t - 8).clip(0)
    x = (level[t + 1] - level[o + 1]) * co[2, 1] / k + (q[t, :, 1] - q[o, :, 1]) * co[2, 1] / co[1, 1]
    ok = (t >= 8)[:, None] & (miss[t + 1] == miss[o])
    x = np.tanh(2 * x) * ok
    return x, level, miss


def geometry(raw, q, geom):
    samples = []
    for a, b in v.price.segments(len(raw)):
        x = source(raw[a:b], q[a:b], geom)[0][::5]
        samples.append(x)
    x = np.concatenate(samples)
    return x.T @ x / len(x)


def axes(history):
    g = np.mean(history, 0)
    models = []
    for j in range(3):
        a, b = (GROUPS[(j + h) % 3] for h in (1, 2))

        def inverse_root(group):
            cov = g[np.ix_(group, group)]
            e, u = np.linalg.eigh(cov + 0.45 * np.diag(cov.diagonal().clip(0.002)))
            return (u / e.clip(0.0001) ** 0.5) @ u.T

        ra, rb = inverse_root(a), inverse_root(b)
        left, strength, right = np.linalg.svd(ra @ g[np.ix_(a, b)] @ rb)
        models.append((ra @ left[:, :4], rb @ right[:4].T, strength[:4]))
    return models


def design(raw, q, t, geom, operators):
    k, co, _ = geom
    pieces = []
    for a, b in v.price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        x, level, miss = source(raw[a:b], q[a:b], geom)
        tick = np.arange(0, b - a, 4)
        pos = np.arange(len(tick))
        out = np.zeros((len(ix), 50, 7))
        for j, recipient in enumerate(GROUPS):
            donor_a, donor_b = (GROUPS[(j + h) % 3] for h in (1, 2))
            a_map, b_map, quality = operators[j]
            aa = 2 * np.tanh(x[:, donor_a] @ a_map / 2)
            bb = 2 * np.tanh(x[:, donor_b] @ b_map / 2)

            def agree(x, y):
                return np.where(x * y > 0, np.sign(x) * np.minimum(abs(x), abs(y)), 0)

            both = agree(aa, bb) * quality

            def transfer(h):
                issue = tick[(pos - h // 4).clip(0)]
                ok = (tick >= h)[:, None] & (miss[tick + 1][:, recipient] == miss[issue][:, recipient])
                n = ok.sum(1).clip(1)

                def center(z):
                    return (z - (z * ok).sum(1, keepdims=True) / n[:, None]) * ok

                price = (
                    co[2, 1] * np.expm1(level[tick + 1][:, recipient] - level[issue + 1][:, recipient]) / k
                )
                if h == 40:
                    price += q[a + tick][:, recipient, 1] * co[2, 1] / co[1, 1]
                y = center(price)
                old = center(q[a + issue][:, recipient, 2])

                def ew(v):
                    return lfilter([1 / 100], [1, -0.99], v, axis=0)

                slope = (ew((y * old).sum(1) / n) / (0.004 + ew((old**2).sum(1) / n))).clip(0, 2)
                error = y - old * slope[:, None]
                val = n >= int(0.65 * len(recipient))

                def learn(sensor):
                    s = sensor[issue]
                    xy = s[:, :, None] * error[:, None, :]
                    xx = s * s

                    def coef(mask):
                        m = val * mask
                        return ew(xy * m[:, None, None]) / (0.12 + ew(xx * m[:, None]))[:, :, None]

                    l, r = [coef((pos // 24) % 2 == i) for i in (0, 1)]
                    return agree(l, r)

                def forecast(sensor, w):
                    return np.einsum("ti,tij->tj", sensor[ix], w[ix // 4], optimize=True)

                ma, mb = learn(aa), learn(bb)
                single = forecast(aa, ma)
                cross = 0.5 * (forecast(aa, mb) + forecast(bb, ma))
                reliable = forecast(both, agree(ma, mb))
                matched = agree(single, forecast(bb, mb))
                return single, cross, reliable, matched

            single, cross, reliable, matched = transfer(60)
            _, early, er, em = transfer(40)
            for col, arr in enumerate(
                (
                    cross,
                    reliable,
                    matched,
                    agree(early, cross),
                    agree(em, matched),
                    single,
                    np.roll(matched, 5, axis=1),
                )
            ):
                out[:, recipient, col] = arr
        pieces.append((out - out.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(pieces)


def train():
    history, cal, days = [], [], []
    geom = v.feedback.geometry()
    from experiment_causal_cross_horizon_ridge_v1 import labels

    for d in range(4, 16):
        with np.load(tape.OUT / f"day_{d:02d}.npz") as f:
            q, t = f["prediction"], f["t"]
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        if d >= 6:
            if d % 2 == 0:
                op = axes(history)
            y = labels(d).reshape(-1, 50, 3)
            z = design(raw, q, t, geom, op)
            if d >= 8:
                if d % 2 == 0:
                    size = v.price.sizing(cal)
                days.append(v.price.rscore(q[t], y[t], z * size))
                if d == 8:
                    cut = 2509
                    use = t < cut
                    np.testing.assert_allclose(
                        z[use], design(raw[:cut], q[:cut], t[use], geom, op), atol=3e-7, rtol=0
                    )
            cal.append(v.price.rscore(q[t], y[t], z))
        history.append(geometry(raw, q, geom))
    halves = np.stack([v.price.diff(sum(g))[1:] for g in (days[:4], days[4:])])
    daily = np.stack([v.price.diff(g)[1:] for g in days])
    j = int(np.argmax(halves[0, :5]))
    pick = (
        j
        if (halves[:, j] > np.maximum(0, halves[:, 5:].max(1))).all() and (daily[:, j] > 0).sum() >= 5
        else None
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(halves=halves.tolist(), daily=daily.tolist(), first=j, pick=pick), indent=2)
    )
    if pick is not None:
        np.savez(OUT / "model.npz", cov=np.mean(history, 0), col=j, size=v.price.sizing(cal)[j])
    print("independent donor baskets", halves, "days", (daily[:, j] > 0).sum(), "selected", pick, flush=True)


def load_change():
    with np.load(OUT / "model.npz") as f:
        op = axes([f["cov"]])
        col = int(f["col"])
        size = float(f["size"])
    geom = v.feedback.geometry()
    return lambda raw, q: size * design(raw, q, np.arange(len(q)), geom, op)[:, :, col]


def correction():
    parent, change = tape.transport.correction(), load_change()

    def apply(raw, q):
        delta = parent(raw, q)
        p = q.copy()
        p[:, :, 2] += delta
        return delta + change(raw, p)

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
