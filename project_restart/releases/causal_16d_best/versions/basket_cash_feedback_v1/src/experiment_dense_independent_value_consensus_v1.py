"""Confirm a relative stock correction with both a peer-network return estimate and a noise-controlled linear estimate."""

import numpy as np
import experiment_dense_peer_noise_control_v1 as hedge
import experiment_dense_hierarchical_value_net_v1 as net
import experiment_dense_neural_value_agreement_v1 as agreement

OUT = net.ref.base.ROOT / "artifacts/dense_independent_value_consensus_v1"
ARMS = ("two_model_average", "confirmed_returns", "confirmed_peer_increment")
USES_FORECAST = True


def train(days):
    if not (net.OUT / f"net_{days}.pt").is_file():
        net.train(days)
    return np.r_[days, hedge.train(days)[:, 1]]


def views(raw, grid, code, weights, p):
    nonlinear = net.views(raw, grid, code, weights[:1], p).astype(float)
    linear = hedge.views(raw, grid, code, weights[1:, None]).astype(float)
    s = np.concatenate((linear, nonlinear), 2)
    q = p.astype(float) - p.mean(1, keepdims=True, dtype=float)
    g = q.transpose(0, 2, 1) @ q
    beta = np.linalg.solve(
        g + 0.03 * np.eye(3) * g.diagonal(axis1=1, axis2=2)[:, None, :].clip(0.005),
        q.transpose(0, 2, 1) @ s,
    )
    residual = s - q @ beta
    rms = np.sqrt((residual**2).mean(1, keepdims=True))
    # The linear fit is normalized during training; neural output units are normalized label errors.
    floor = np.array([0.15, 0.015, 0.015, 0.015])
    standardized = residual / (rms + floor)
    linear, own, group, teacher = standardized.transpose(2, 0, 1)
    direct = 0.5 * (own + group)
    peer = agreement.consensus(group - own, teacher - own)
    result = np.stack(
        (
            0.5 * (linear + direct),
            agreement.consensus(linear, direct),
            agreement.consensus(linear, peer),
        ),
        -1,
    )
    result *= np.sqrt((q[:, :, 2:3] ** 2).mean(1, keepdims=True) + 0.01)
    return result - result.mean(1, keepdims=True)


if __name__ == "__main__":
    net.rel.main(OUT, ARMS, train, views, with_prediction=True)
