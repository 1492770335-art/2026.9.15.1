"""Compare the first absolute moment of a stock's backward five-tick move to RMS, stale and swapped motion."""

import argparse
import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_bipower_phase_pool_v1 as current

parent, price = current.parent, current.price
OUT = price.OUT.parent / "y3_absolute_block_motion_v1"
ARMS = (
    "retained",
    "absolute move",
    "rms move",
    "absolute beyond rms",
    "another stock's absolute move",
    "delayed absolute move",
)


def design(raw, q, t, k, sd, axis):
    rows = []
    for a, b in price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        level, x = parent.score_inputs(raw[a:b], k, sd, axis)
        times = np.arange(len(level))
        v = abs(level - level[(times - 5).clip(0)]).clip(0, 10)
        vals = np.stack((v, v * v), axis=2)
        p = q[a + ix]
        slow = lfilter([0.05], [1, -0.95], x[:, :, 0], axis=0)[ix]
        marks = np.concatenate(
            (x[ix, :, :1], slow[:, :, None], x[ix, :, 1:], p - p.mean(1, keepdims=True)), axis=2
        )
        views = []
        for span in (50, 250):
            alpha = -np.expm1(-1 / span)
            m = lfilter([alpha], [1, alpha - 1], vals, axis=0)
            absolute = np.tanh(m[ix, :, 0])
            rms = np.tanh(np.sqrt(m[ix, :, 1]))
            views.append((absolute, rms, np.roll(absolute, 17, axis=1), np.tanh(m[(ix - 100).clip(0), :, 0])))
        z = np.concatenate(
            [
                marks,
                *[
                    np.concatenate([gate[:, :, None] * marks for gate in group], axis=2)
                    for group in zip(*views)
                ],
            ],
            axis=2,
        )
        rows.append((z - z.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(rows)


def weights(history):
    g = np.mean(history, axis=0)[:-2, :-2]
    xy = np.stack(history)[:, :-2, -2:]
    marks = np.arange(6)
    absolute, rms, swap, stale = np.split(np.arange(6, len(g)), 4)
    out = []
    for sel, remove in (
        (absolute, marks),
        (rms, marks),
        (absolute, np.r_[marks, rms]),
        (swap, marks),
        (stale, marks),
    ):
        basis = np.eye(len(g))[:, sel]
        control = g[np.ix_(remove, remove)]
        basis[remove] = -np.linalg.solve(
            control + 0.025 * np.diag(control.diagonal().clip(1e-6)), g[np.ix_(remove, sel)]
        )
        cov = basis.T @ g @ basis
        grad = np.einsum("dkh,ki->dih", xy, basis)
        avg = grad.mean(0)
        avg *= abs(avg) / (abs(avg) + grad.std(0) / np.sqrt(len(grad)) + 1e-5)
        stable = np.where(
            avg[:, 0] * avg[:, 1] > 0, np.sign(avg[:, 0]) * np.minimum(abs(avg[:, 0]), abs(avg[:, 1])), 0
        )
        out.append(basis @ np.linalg.solve(cov + 0.9 * np.diag(cov.diagonal().clip(1e-5)), stable))
    return np.stack(out, axis=1).astype(np.float32)


def records():
    k, co, _ = price.previous_stage.early_slopes.geometry()
    ratio = co[2, 1] / co[1, 1]
    axis = parent.direction(k)
    history, sizing = [], []
    for d, _, p, y, model in current.base.records():
        raw = np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        t = np.arange(d % 5, len(p), 5)
        x = parent.design(raw, p, t, k, model[2], axis)
        q = p
        if d >= 6:
            if d % 2 == 0:
                w = parent.weights(history)[:, 3]
            z = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(sizing)[0]
                panel = current.features(raw, p, k, model[2], axis, w * size)
                np.testing.assert_allclose(z * size, panel[t, :, 0], atol=1e-6, rtol=0)
                q = p.copy()
                q[:, :, 2] += panel[:, :, 0] + panel[:, :, 2]
            sizing.append(price.rscore(p[t], y[t], z[:, :, None]))
        ok = ((t < 2401 - 60) | (t >= 2401)) & (t < len(p) - 60)
        history.append(parent.stat(x[ok], p[t[ok]], y[t[ok]], ratio))
        yield d, raw, q, y


def train():
    k, co, _ = price.previous_stage.early_slopes.geometry()
    ratio = co[2, 1] / co[1, 1]
    axis = parent.direction(k)
    sd = price.innovation_model(
        [
            price.observe(price.tape(np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r"), k))
            for d in range(4)
        ]
    )[2]
    hist, cal, days, parts = [], [], [], []
    for d, raw, q, y in records():
        t = np.arange(d % 5, len(q), 5)
        x = design(raw, q, t, k, sd, axis)
        if d >= 6:
            if d % 2 == 0:
                w = weights(hist)
            z = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                days.append(price.rscore(q[t], y[t], z * size))
                parts.append(
                    [parent.gram(q[t], y[t], z[:, :, j] * size[j], ratio) for j in (0, 1, 2)]
                )
                if d == 8:
                    n = 2473
                    np.testing.assert_allclose(
                        x[t < n], design(raw[:n], q[:n], t[t < n], k, sd, axis), rtol=0, atol=3e-6
                    )
            cal.append(price.rscore(q[t], y[t], z))
        ok = ((t < 2401 - 60) | (t >= 2401)) & (t < len(q) - 60)
        hist.append(parent.stat(x[ok], q[t[ok]], y[t[ok]], ratio))
    h = np.stack([price.diff(sum(f)) for f in (days[:4], days[4:])])
    ds = np.stack([price.diff(v) for v in days])
    attr = [[parent.decompose(g) for g in np.sum(v, axis=0)] for v in (parts[:4], parts[4:])]
    # Absolute motion is the primary replication. Controls share its coefficient fitting recipe.
    distal_ok = (
        h[:, 1].min() > max(0, h[:, 4].min(), h[:, 5].min())
        and (ds[:, 1] > 0).sum() >= 5
        and all(row[0]["late"] > 0 for row in attr)
    )
    # Eight positive dates and both contemporaneous stock controls beaten on each fold
    # also qualify the full endpoint, even when different legs drive the two folds.
    endpoint_ok = (ds[:, 1] > 0).all() and (h[:, 1, None] > h[:, [4, 5]]).all()
    ok = distal_ok or endpoint_ok
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(
                selected=int(ok),
                distal_qualified=bool(distal_ok),
                endpoint_qualified=bool(endpoint_ok),
                arms=ARMS,
                halves=h.tolist(),
                daily=ds.tolist(),
                attribution=attr,
            ),
            indent=2,
        )
    )
    if ok:
        np.savez(
            OUT / "model.npz", k=k, sd=sd, axis=axis, w=weights(hist)[:, 0] * price.sizing(cal)[0]
        )
    print("rolling Y3", h, "selected", bool(ok), flush=True)
    return ok


def load():
    prev = current.load()
    with np.load(OUT / "model.npz") as f:
        k, sd, axis, w = float(f["k"]), f["sd"], f["axis"], f["w"]

    def apply(raw, p):
        before = prev(raw, p)
        q = p.copy()
        q[:, :, 2] += before
        return before + design(raw, q, np.arange(len(q)), k, sd, axis) @ w

    return apply


def correction():
    prior=current.base.memory.correction()
    last=load()
    def apply(raw,p):
        shift=prior(raw,p)
        q=p.copy()
        q[:,:,2]+=shift
        return shift+last(raw,q)
    return apply


if __name__=='__main__':
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('train','verify'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=='train':
        train()
    else:
        parent.replay(OUT,correction)
