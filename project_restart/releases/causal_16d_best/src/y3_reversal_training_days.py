"""Chronological Y3 targets and frozen price-memory predictions for days five through sixteen."""

import json

import numpy as np
from scipy.signal import lfilter

import experiment_y3_reversal_conditioned_memory_v1 as memory

price = memory.price
OUT = price.OUT.parent / "y3_reversal_training_days_v1"
INCUMBENT = memory.OUT


def evidence(move, lag, model):
    w, _, sd = model
    h = memory.H
    base = 1 - 1 / h
    norm = 1 / np.sqrt(h)
    shocks = move - lag @ w
    trend = np.stack([lfilter([n], [1, -b], shocks, axis=0) for n, b in zip(norm, base)], axis=2)
    previous = np.concatenate((np.zeros_like(trend[:1]), trend[:-1]))
    tilt = np.tanh(shocks[:, :, None] * previous / sd[None, :, None] ** 2)
    energy = np.minimum(12, (shocks / sd) ** 2)[:, :, None]
    rates = energy * (1 - tilt)
    decay = base ** (0.30 + 0.70 * rates)
    state = np.zeros((50, len(h)))
    seq = np.empty_like(trend, dtype=np.float32)
    for t, row in enumerate(shocks):
        state = decay[t] * state + row[:, None] * norm
        seq[t] = state
    return seq, rates, shocks


def features(seq, sd):
    return np.concatenate([memory.summarize(seq[:, :, i], sd)[1] for i in range(4)], axis=2)


def build():
    OUT.mkdir(exist_ok=True)
    k = price.previous_stage.early_slopes.geometry()[0]
    gs = [
        price.observe(price.tape(np.load(price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r"), k))
        for d in range(4)
    ]
    hist, cal, days = [], [], []
    for d in range(4, 16):
        raw, _, y = price.read_day(d)
        q = np.load(memory.reference.OUT / f"day_{d}.npy")
        if d % 2 == 0:
            model = price.innovation_model(gs)
            np.savez(OUT / f"filter_{d}.npz", w=model[0], pr=model[1], sd=model[2])
        chunks = price.tape(raw, k)
        x = [np.concatenate([features(evidence(r, lag, model)[0], model[2]) for r, lag in chunks])]
        record = memory.row(x, q, y, d)
        if d >= 6:
            if d % 2 == 0:
                w = memory.weights(hist)
            z = memory.signals(x, q, w)
            if d >= 8 and d % 2 == 0:
                size = price.sizing(cal)[0]
            cal.append(price.rscore(q, y, z))
            if d >= 8:
                days.append(price.rscore(q, y, z * size))
                q[:, :, 2] += z[:, :, 0] * size
        hist.append(record)
        gs.append(price.observe(chunks))
        dest = OUT / f"day_{d}.npy"
        tmp = dest.with_suffix(".tmp.npy")
        np.save(tmp, q)
        tmp.replace(dest)
    actual = [price.diff(sum(v))[1] for v in (days[:4], days[4:])]
    expected = np.array(json.loads((INCUMBENT / "results.json").read_text())["halves"])[:, 3]
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    (OUT / "verified.json").write_text(json.dumps(dict(Y3_gains=actual, days=list(range(5, 17))), indent=2))
    print("matched rolling Y3 gains", actual)


def records():
    if not (OUT / "verified.json").is_file():
        build()
    k = price.previous_stage.early_slopes.geometry()[0]
    for d in range(4, 16):
        raw, _, y = price.read_day(d)
        with np.load(OUT / f"filter_{d - d % 2}.npz") as p:
            model = p["w"], p["pr"], p["sd"]
        q = np.load(OUT / f"day_{d}.npy", mmap_mode="r")
        yield d, price.tape(raw, k), q, y, model


if __name__ == "__main__":
    build()
