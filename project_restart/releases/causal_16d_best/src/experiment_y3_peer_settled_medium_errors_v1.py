"""Test whether the stock-specific pattern of other stocks' already settled 20-tick forecast errors predicts Y3."""

import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_independent_peer_sensors_v1 as prior
from experiment_causal_cross_horizon_ridge_v1 import labels

v, tape = prior.v, prior.tape
OUT = prior.OUT.parent / "y3_peer_settled_medium_errors_v1"
DECAYS = np.exp(-2.5 / np.array((64.0, 128.0, 256.0)))
ARMS = (
    "peer settlement error",
    "cash only",
    "issued forecast only",
    "wrong receiver",
    "stale errors",
    "own error",
    "unfiltered error",
    "misaligned forecast",
)


def training_panels():
    history, cal = [], []
    geom = v.feedback.geometry()
    for d in range(4, 16):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        with np.load(tape.OUT / f"day_{d:02d}.npz") as f:
            q, t = f["prediction"], f["t"]
        if d >= 6:
            if d % 2 == 0:
                op = prior.axes(history)
                size = v.price.sizing(cal)[0] if cal else 0
            old = prior.design(raw, q, np.arange(len(q)), geom, op)[:, :, 0]
            y = labels(d).reshape(-1, 50, 3)
            cal.append(v.price.rscore(q[t], y[t], old[t, :, None]))
            p = q.copy()
            p[:, :, 2] += size * old
            yield d, raw, p, t, y, geom, op
        history.append(prior.geometry(raw, q, geom))


def inputs(raw, q, geom):
    k, co, row = geom
    tick = np.arange(len(q))
    issue = (tick - 20).clip(0)
    level = v.feedback.cash.compound(raw, k)
    missing = np.vstack((np.zeros((1, 50), int), (~v.feedback.cash.present(raw, row)).cumsum(0)))
    ok = (tick >= 20)[:, None] & (missing[tick + 1] == missing[issue])
    cash = (level[tick + 1] - level[issue + 1]) * co[2, 1] / k
    pred = q[issue, :, 1] * co[2, 1] / co[1, 1]
    wrong = q[(issue - 20).clip(0), :, 1] * co[2, 1] / co[1, 1]
    return np.tanh(2 * np.stack((cash - pred, cash, pred, cash - wrong), axis=2)) * ok[:, :, None]


def statistics(raw, q, geom):
    sections = []
    for a, b in v.price.segments(len(raw)):
        x = inputs(raw[a:b], q[a:b], geom)
        mats = []
        for decay in (*DECAYS, 0.0):
            f = lfilter([1 - decay], [1, -decay], x, axis=0)[::4]
            mats.append([s.T @ s / len(s) for s in f.transpose(2, 0, 1)])
        sections.append(mats)
    return np.mean(sections, axis=0)


def fit(hist):
    cov = np.mean(hist, axis=0)
    weights = np.empty_like(cov)
    for i, scale in enumerate(cov):
        for j, g in enumerate(scale):
            prec = np.linalg.inv(g + 0.65 * np.diag(g.diagonal().clip(1e-6)))
            w = -prec / prec.diagonal()[None, :]
            np.fill_diagonal(w, 0)
            weights[i, j] = w - (1 - np.eye(50)) * w.sum(0) / 49
    np.testing.assert_allclose(weights.diagonal(axis1=2, axis2=3), 0, atol=1e-12)
    np.testing.assert_allclose(weights.sum(2), 0, atol=1e-12)
    return weights


def design(raw, q, t, geom, w):
    parts = []
    ix_stocks = np.roll(np.arange(50), 13)
    for a, b in v.price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        x = inputs(raw[a:b], q[a:b], geom)
        views, own, wrong = [], [], []
        for decay, maps in zip(DECAYS, w):
            f = lfilter([1 - decay], [1, -decay], x, axis=0)
            views.append(np.stack([s @ m for s, m in zip(f.transpose(2, 0, 1), maps)], axis=-1))
            own.append(f[:, :, 0])
            wrong.append(f[:, :, 0] @ maps[0][np.ix_(ix_stocks, ix_stocks)])
        residual, cash, pred, misaligned = np.mean(views, axis=0).transpose(2, 0, 1)
        cols = (
            residual[ix],
            cash[ix],
            pred[ix],
            np.mean(wrong, axis=0)[ix],
            residual[(ix - 120).clip(0)] * (ix >= 120)[:, None],
            np.mean(own, axis=0)[ix],
            x[ix, :, 0] @ w[3, 0],
            misaligned[ix],
        )
        z = np.stack(cols, axis=2)
        parts.append((z - z.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(parts)


def train():
    hist, cal, days, offsets = [], [], [], []
    geom = v.feedback.geometry()
    for d in (4, 5):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        with np.load(tape.OUT / f"day_{d:02d}.npz") as f:
            hist.append(statistics(raw, f["prediction"], geom))
    for d, raw, q, _, y, geom, _ in training_panels():
        if d % 2 == 0:
            w = fit(hist)
            if cal:
                size = v.price.sizing(cal)
        t = np.arange(len(q))
        z = design(raw, q, t, geom, w)
        if d >= 8:
            gs = [v.price.rscore(q[j::5], y[j::5], z[j::5] * size) for j in range(5)]
            days.append(sum(gs))
            offsets.append(gs)
            if d == 8:
                cut = 2483
                np.testing.assert_allclose(
                    z[:cut], design(raw[:cut], q[:cut], t[:cut], geom, w), atol=5e-7, rtol=0
                )
        cal.append(v.price.rscore(q, y, z))
        hist.append(statistics(raw, q, geom))
    diff = lambda g: v.price.diff(g)[1:]
    halves = np.stack([diff(sum(chunk)) for chunk in (days[:4], days[4:])])
    phases = np.array([[diff(sum(g)) for g in zip(*chunk)] for chunk in (offsets[:4], offsets[4:])])
    daily = np.stack([diff(g) for g in days])
    # The primary contrast is fixed in advance. Require evidence against source identity, age and own-stock controls
    # in both folds, and against absence of the forecast-error pairing on average.
    pick = (
        0
        if (
            (halves[:, 0] > np.maximum(0, halves[:, 3:6].max(1))).all()
            and halves.mean(0)[0] > halves.mean(0)[[1, 2, 6, 7]].max()
            and (daily[:, 0] > 0).sum() >= 5
            and (phases[:, :, 0] > 0).sum(1).min() >= 4
        )
        else None
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(arms=ARMS, pick=pick, halves=halves.tolist(), phases=phases.tolist(), daily=daily.tolist()),
            indent=2,
        )
    )
    if pick is not None:
        np.savez(OUT / "model.npz", weights=fit(hist), size=v.price.sizing(cal)[0], col=pick)
    print(
        "cross-stock propagation of settled 20-tick errors",
        halves,
        "dates",
        int((daily[:, 0] > 0).sum()),
        "chosen",
        pick,
        flush=True,
    )


def load_change():
    with np.load(OUT / "model.npz") as f:
        weights = f["weights"][:3, 0]
        size = float(f["size"])
        assert int(f["col"]) == 0
    geom = v.feedback.geometry()

    def apply(raw, q):
        sections = []
        for a, b in v.price.segments(len(raw)):
            x = inputs(raw[a:b], q[a:b], geom)[:, :, 0]
            pred = sum(lfilter([1 - p], [1, -p], x, axis=0) @ w for p, w in zip(DECAYS, weights)) / 3
            sections.append((pred - pred.mean(1, keepdims=True)).astype(np.float32))
        return size * np.concatenate(sections)

    return apply


def correction():
    previous, change = prior.correction(), load_change()

    def revise(raw, p):
        old = previous(raw, p)
        q = p.copy()
        q[:, :, 2] += old
        return old + change(raw, q)

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
