"""Use past forecasts and intervening realized returns to estimate today's sixty-tick return."""

import numpy as np
import experiment_y3_prequential_disagreement_gate_v1 as gate


price = gate.price
feedback = gate.rev
OUT = gate.OUT.parent / "y3_dense_forecast_vintages_v1"
TAPE = OUT.parent / "y3_dense_training_tape_v1"
LAGS = (1, 2, 4, 8, 16, 32)
MODES = ("dense", "sparse")


def features(raw, q, t, geom):
    k, co, row = geom
    parts = []
    for a, b in price.segments(len(raw)):
        ix = t[(t >= a) & (t < b)] - a
        if not len(ix):
            continue
        quotes = raw[a:b]
        forecasts = q[a:b]
        cash = feedback.cash.compound(quotes, k)[1:] / k * co[2, 1]
        miss = np.vstack((np.zeros((1, 50), int), (~feedback.cash.present(quotes, row)).cumsum(0)))
        far = forecasts[:, :, 2]
        near = forecasts[:, :, 1] * (co[2, 1] / co[1, 1])
        now = far[ix]
        output = []
        for lag in LAGS:
            old = (ix - lag).clip(0)
            valid = (ix >= lag)[:, None] & (miss[ix + 1] == miss[old])
            realized = cash[ix] - cash[old]
            # Roll the vintage forward by the time elapsed using its predicted distal drift.
            roll = far[old] - realized + lag / 40 * (far[old] - near[old])
            revision = np.tanh((roll - now) / 1.5) * valid
            near_update = np.tanh((near[ix] - near[old] + realized) / 1.5) * valid
            distal = np.tanh((far[old] - near[old]) * 2)
            output.extend(
                (
                    revision,
                    near_update,
                    revision * np.abs(distal),
                    revision * np.tanh(6 * near_update * revision),
                )
            )
        x = np.stack(output, axis=2)
        parts.append((x - x.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(parts)


def moments(x, p, y):
    v = np.column_stack((np.ones(p.size), p.ravel(), y.ravel(), x.reshape(p.size, -1))).astype(float)
    return (v.T @ v) / len(v)


def fit(stats):
    g = np.mean(stats, axis=0)
    c = g - np.outer(g[0], g[0])
    # Remove the direction of the existing forecast from the correlation gradient.
    grad = c[3:, 2] - c[3:, 1] * c[1, 2] / c[1, 1]
    var = c[3:, 3:]
    w = np.linalg.solve(var + np.diag(0.65 * var.diagonal().clip(0.0001)), grad)
    return w * (c[1, 2] / c[2, 2])
