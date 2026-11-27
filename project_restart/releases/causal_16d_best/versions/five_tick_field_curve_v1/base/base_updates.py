"""Apply each trained research readout once to its immediate predecessor."""

import numpy as np

import experiment_categorical_suffix_responses_v1 as suffix

import experiment_conditional_tail_payouts_v1 as tail

import experiment_supervised_gradient_kernel_v1 as kernel

import experiment_rolling_drift_strength_v1 as drift

import experiment_propagated_return_kernel_v1 as propagated


def suffix_update():
    with np.load(suffix.OUT / "model.npz") as st:
        trip = st["trip"].copy()
        gate = st["gate"].copy()
        tables = [
            [st[f"{i}_{j}"].copy() for j in range(len(suffix.LEVELS))]
            for i in range(len(trip))
        ]

    def refine(raw, p):
        q = p
        a, b = suffix.predict(raw, trip, tables)
        return (q + a * gate[0] + b * gate[1]).astype("float32")

    return (suffix.OUT, refine)


def tail_update():
    net = tail.load_model(16, "cpu")
    with np.load(tail.OUT / "model.npz") as st:
        wm = st["market"].copy()
        wr = st["relative"].copy()

    def refine(raw, p):
        base = p
        m, r = tail.design(net, raw, base)
        return (base + (m @ wm)[:, None] + r @ wr).astype("float32")

    return (tail.OUT, refine)


def kernel_update():
    geo, w, coef = kernel.load_fit(kernel.OUT)

    def refine(raw, p):
        base = p
        m, r = kernel.readout.basis(base, kernel.predict(raw, geo, w))
        return (base + (m @ coef[0])[:, None] + r @ coef[1]).astype("float32")

    return (kernel.OUT, refine)


def drift_update():
    geo, w, coef = drift.kernel.load_fit(drift.drift.OUT)
    mul = np.load(drift.OUT / "multiplier.npy")

    def refine(raw, p):
        base = p
        side = drift.kernel.predict(raw, geo, w)
        m = side.mean(1)
        r = drift.pair.innovation(base, side - m[:, None])
        return (base + (m @ coef[0])[:, None] * mul[0] + r @ coef[1] * mul[1]).astype(
            "float32"
        )

    return (drift.OUT, refine)


def propagated_update():
    geo, w, coef = propagated.kernel.load_fit(propagated.OUT)
    with np.load(propagated.OUT / "model.npz") as st:
        operators = [
            [tuple((st[f"op{h}_{i}_{j}"].copy() for j in range(3))) for i in range(2)]
            for h in range(len(propagated.LAGS))
        ]

    def refine(raw, p):
        base = p
        m, r = propagated.readout.basis(
            base, propagated.predict(raw, base, geo, w, operators)
        )
        return (base + (m @ coef[0])[:, None] + r @ coef[1]).astype("float32")

    return (propagated.OUT, refine)
