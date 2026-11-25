"""Fit a five-tick bipower variation gate to the stock-relative Y3 prediction."""

import argparse
import json
import numpy as np
from scipy.signal import lfilter
import torch

import y3_reversal_training_days as base
from y3_portfolio_walkforward import replay

price = base.price
OUT = price.OUT.parent / "y3_local_jump_variation_v1"
SPANS = (50, 250)
FIELDS = (11, 12, 14, 15, 26, 38, 41, 42, 50, 51, 54, 55, 77, 78)
ARMS = (
    "incumbent",
    "current signals",
    "total volatility",
    "isolated share",
    "continuous component",
    "other stock share",
    "four-block separation",
)


def pressure_fields(v):
    z = np.tanh(0.7 * np.arcsinh(v[:, :, FIELDS].astype(np.float32)))
    return z[:, :, ::2] - z[:, :, 1::2]


def direction(k):
    grams = []
    for d in range(4):
        raw = np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        for a, b in price.segments(len(raw)):
            v = raw[a:b]
            f = pressure_fields(v)
            p = (np.log1p(k * (v[:, :, 0].astype(float) - 0.00136)) / k).cumsum(0)
            t = np.arange(10, len(v) - 5, 5)
            x = np.concatenate((f[t], f[t] - f[t - 5]), 2)
            x -= x.mean(1, keepdims=True)
            y = p[t + 5] - p[t]
            y -= y.mean(1, keepdims=True)
            z = np.concatenate((x, y[:, :, None]), 2).reshape(-1, 15).astype(float)
            grams.append(z.T @ z / len(z))
    g = np.mean(grams, 0)
    w = np.linalg.solve(g[:-1, :-1] + 0.8 * np.diag(g.diagonal()[:-1].clip(0.0001)), g[:-1, -1])
    return w / np.sqrt(w @ g[:-1, :-1] @ w + 0.0001)


def score_inputs(raw, k, sd, axis):
    t = np.arange(len(raw))
    book = pressure_fields(raw)
    book_value = book @ axis[:7] + (book - book[(t - 5).clip(0)]) @ axis[7:]
    book_value -= book_value.mean(1, keepdims=True)
    move = np.log1p(k * (raw[:, :, 0].astype(float) - 0.00136)) / (k * sd)
    level = move.cumsum(0)
    past = level - level[(t - 5).clip(0)]
    value = np.tanh(book_value)
    past = np.tanh(past / 2)
    return level, np.stack((value, past), axis=2)


def stat(x, p, y, ratio):
    q = p.astype(float)
    q -= q.mean((0, 1))
    y = y.astype(float)
    y -= y.mean((0, 1))
    third = q[:, :, 2]
    truth = y[:, :, 2]
    slope = np.clip((third * truth).sum() / (third * third).sum(), 0.4, 2.5)
    near = y[:, :, 1] * ratio

    def adjust(a):
        return (a - third * ((third * a).sum() / (third * third).sum())) / slope

    labels = np.stack((adjust(truth), adjust(truth - near)), 2)
    matrix = np.concatenate((x, labels.clip(-5, 5)), 2).reshape(-1, x.shape[-1] + 2).astype(float)
    return matrix.T @ matrix / len(matrix)


def gram(q, y, z, ratio):
    feat = np.stack(
        (np.ones_like(z), q[:, :, 2], z, ratio * y[:, :, 1], y[:, :, 2] - ratio * y[:, :, 1]), axis=2
    )
    flat = feat.reshape(-1, 5).astype(float)
    return flat.T @ flat


def decompose(g):
    c = g[1:, 1:] / g[0, 0] - np.outer(g[0, 1:], g[0, 1:]) / g[0, 0] ** 2
    sy = c[2:, 2:].sum() ** 0.5
    s0 = c[0, 0] ** 0.5
    s1 = c[:2, :2].sum() ** 0.5
    change = ((c[0, 2:] + c[1, 2:]) / s1 - c[0, 2:] / s0) / sy
    return dict(near=float(change[0]), late=float(change[1]), total=float(change.sum()))


def variation(level):
    states = []
    for span in SPANS:
        alpha = -np.expm1(-5 / span)
        state = np.zeros((*level.shape, 4), np.float32)
        for offset in range(min(5, len(level))):
            v = np.abs(np.diff(level[offset::5], axis=0)).clip(0, 10)
            if not len(v):
                continue

            def ema(arr):
                return lfilter([alpha], [1.0, -(1 - alpha)], arr, axis=0)

            rv = ema(v * v)
            bips = []
            for lag in (1, 4):
                paired = np.zeros_like(v)
                paired[lag:] = v[lag:] * v[:-lag]
                bips.append(np.pi / 2 * ema(paired))
            # Signed excess is retained: adjacent variation can exceed quadratic variation.
            jump, stale = [np.tanh((rv - bp) / (0.8 + rv)) for bp in bips]
            burn = np.minimum(np.arange(len(v)) / 10, 1.0)[:, None]
            state[offset + 5 :: 5] = np.stack(
                (np.tanh(np.sqrt(rv)), jump * burn, np.tanh(np.sqrt(bips[0])), stale * burn), axis=2
            )
        states.append(state)
    return np.stack(states, axis=2)


def design(raw, p, ticks, k, sd, axis):
    rows = []
    for a, b in price.segments(len(raw)):
        ix = ticks[(ticks >= a) & (ticks < b)] - a
        if not len(ix):
            continue
        level, x = score_inputs(raw[a:b], k, sd, axis)
        v, jump, bip, stale = variation(level)[ix].transpose(3, 0, 1, 2)
        slow = lfilter([0.05], [1.0, -0.95], x[:, :, 0], axis=0)[ix]
        q = p[a + ix]
        marks = np.concatenate(
            (x[ix, :, :1], slow[:, :, None], x[ix, :, 1:], q - q.mean(1, keepdims=True)), axis=2
        )
        blocks = [marks]
        for state in (v, jump, bip, np.roll(jump, 17, axis=1), stale):
            blocks.append((marks[:, :, None, :] * state[:, :, :, None]).reshape(len(ix), 50, -1))
        z = np.concatenate(blocks, axis=2)
        rows.append((z - z.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(rows)


def weights(hist):
    g = np.mean(hist, 0)[:-2, :-2]
    grads = np.stack(hist)[:, :-2, -2:]
    groups = [np.arange(6), *np.split(np.arange(6, len(g)), 5)]
    out = []
    for j, ix in enumerate(groups):
        basis = np.eye(len(g))[:, ix]
        if j:
            n = 6 if j == 1 else 18
            basis[:n] = -np.linalg.solve(g[:n, :n] + 0.025 * np.diag(g.diagonal()[:n].clip(1e-6)), g[:n, ix])
        sub = basis.T @ g @ basis
        grad = np.einsum("dkh,ki->dih", grads, basis)
        avg = grad.mean(0)
        avg *= abs(avg) / (abs(avg) + grad.std(0) / np.sqrt(len(grad)) + 1e-5)
        # Only pay for a distal signal whose direction also improves the full target.
        stable = np.where(
            avg[:, 0] * avg[:, 1] > 0, np.sign(avg[:, 0]) * np.minimum(abs(avg[:, 0]), abs(avg[:, 1])), 0.0
        )
        out.append(basis @ np.linalg.solve(sub + 0.9 * np.diag(sub.diagonal().clip(1e-5)), stable))
    return np.stack(out, axis=1).astype(np.float32)


def train():
    k, co, _ = price.previous_stage.early_slopes.geometry()
    ratio = co[2, 1] / co[1, 1]
    axis = direction(k)
    hist, cal, days, parts = [], [], [], []
    for d, _, p, y, model in base.records():
        raw = np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        t = np.arange(d % 5, len(p), 5)
        x = design(raw, p, t, k, model[2], axis)
        if d >= 6:
            if d % 2 == 0:
                w = weights(hist)
            z = x @ w
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                days.append(price.rscore(p[t], y[t], z * size))
                parts.append([gram(p[t], y[t], z[:, :, j] * size[j], ratio) for j in (2, 3)])
                if d == 8:
                    n = 2549
                    np.testing.assert_allclose(
                        x[t < n], design(raw[:n], p[:n], t[t < n], k, model[2], axis), atol=4e-6, rtol=0
                    )
            cal.append(price.rscore(p[t], y[t], z))
        ok = ((t < 2401 - 60) | (t >= 2401)) & (t < len(p) - 60)
        hist.append(stat(x[ok], p[t[ok]], y[t[ok]], ratio))
    halves = np.stack([price.diff(sum(v)) for v in (days[:4], days[4:])])
    daily = np.stack([price.diff(v) for v in days])
    attr = [[decompose(g) for g in np.sum(v, axis=0)] for v in (parts[:4], parts[4:])]
    eligible = [
        j
        for j in (3, 4)
        if (halves[:, j] > 0).all()
        and (daily[:, j] > 0).sum() >= 5
        and all(v[j - 3]["late"] > 0 for v in attr)
        and halves[:, j].min() > max(halves[:, 5].min(), halves[:, 6].min())
    ]
    pick = max(eligible, key=lambda j: halves[:, j].min()) if eligible else 0
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            dict(arms=ARMS, selected=pick, halves=halves.tolist(), daily=daily.tolist(), attribution=attr),
            indent=2,
        )
    )
    if pick:
        with np.load(base.INCUMBENT / "model.npz") as f:
            sd = f["sd"]
        np.savez(
            OUT / "model.npz",
            k=k,
            sd=sd,
            axis=axis,
            w=weights(hist)[:, pick - 1] * price.sizing(cal)[pick - 1],
        )
    print("selected", pick, "Y3 changes", halves, flush=True)
    return bool(pick)


def load():
    with np.load(OUT / "model.npz") as f:
        k, sd, axis, w = float(f["k"]), f["sd"], f["axis"], f["w"]

    def apply(raw, p):
        return design(raw, p, np.arange(len(p)), k, sd, axis) @ w

    return apply


def correction():
    before = base.memory.correction()
    change = load()

    def apply(raw, p):
        previous = before(raw, p)
        q = p.copy()
        q[:, :, 2] += previous
        return previous + change(raw, q)

    return apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        replay(OUT, correction)
