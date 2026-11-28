"""Select contemporaneously independent value innovations validated by the separate shorter-horizon teacher."""

import numpy as np
import experiment_dense_hierarchical_value_net_v1 as net

OUT = net.ref.base.ROOT / "artifacts/dense_neural_value_agreement_v1"
ARMS = ("panel_residual", "return_teacher_agreement", "peer_specific_agreement")
USES_FORECAST = True


def train(days):
    if not (net.OUT / f"net_{days}.pt").is_file():
        net.train(days)
    return np.array([days])


def consensus(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def views(raw, grid, code, bank, p):
    signals = net.views(raw, grid, code, bank, p).astype(float)
    q = p.astype(float) - p.mean(1, keepdims=True, dtype=float)
    gram = q.transpose(0, 2, 1) @ q
    reg = 0.1 * np.eye(3) * gram.diagonal(axis1=1, axis2=2)[:, None, :].clip(0.005)
    coef = np.linalg.solve(gram + reg, q.transpose(0, 2, 1) @ signals)
    independent = signals - q @ coef
    # Normalize across the observable 50-name panel before comparing heads trained on different targets.
    unit = np.sqrt(np.mean(independent**2, axis=1, keepdims=True) + 0.0004)
    z = independent / unit
    first = z[..., 0]
    group = z[..., 1]
    teacher = z[..., 2]
    out = np.stack(
        (
            independent[..., 0],
            consensus(0.5 * (first + group), teacher),
            consensus(group - first, teacher - first),
        ),
        -1,
    )
    return out - out.mean(1, keepdims=True)


if __name__ == "__main__":
    net.rel.main(OUT, ARMS, train, views, with_prediction=True)
