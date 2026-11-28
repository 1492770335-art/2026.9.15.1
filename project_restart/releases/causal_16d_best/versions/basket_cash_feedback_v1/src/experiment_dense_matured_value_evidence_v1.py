"""Price two independent stock-value estimates using already matured, feature-observable cross-sectional returns."""

import numpy as np
from scipy.signal import lfilter
import experiment_dense_independent_value_consensus_v1 as consensus
from experiment_dense_session_slope_v1 import GEO

OUT = consensus.OUT.parent / "dense_matured_value_evidence_v1"
ARMS = ("observed_sixty_tick_skill", "distal_skill", "replicated_expert_skill")
USES_FORECAST = True
train = consensus.train


def check_template():
    samples = []
    for d in range(4):
        raw = np.load(consensus.net.ref.base.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        samples.append(raw[d::21].reshape(-1, 100))
    rows, counts = np.unique(np.concatenate(samples), axis=0, return_counts=True)
    with np.load(GEO) as g:
        np.testing.assert_array_equal(rows[counts.argmax()], g["row"])


def update(x, grid, p, z):
    out = np.zeros_like(z, dtype=float)
    with np.load(GEO) as f:
        k, co, row = float(f["k"]), f["coeff"][2, 1], f["row"]
    for a, b in consensus.net.rel.fields.parent.sessions(grid):
        raw = x[a:b]
        q = p[a:b]
        signal = z[a:b]
        path = np.vstack(
            (
                np.zeros(50),
                np.cumsum(np.log1p(k * (raw[:, :, 0].astype(float) - 0.00136)), axis=0),
            )
        )
        missed = np.vstack(
            (np.zeros(50, int), np.cumsum(np.all(raw == row, axis=2), axis=0))
        )
        stops = np.arange(60, len(q), 5)
        if not len(stops):
            continue
        origin = stops - 60
        ok = missed[stops + 1] == missed[origin]
        mass = ok.sum(1, keepdims=True).clip(1)

        def center(v):
            return (
                v - (v * ok[..., None]).sum(1, keepdims=True) / mass[..., None]
            ) * ok[..., None]

        s = center(signal[origin])
        controls = center(q[origin].astype(float))
        g = controls.transpose(0, 2, 1) @ controls
        beta = np.linalg.solve(
            g + 0.002 * np.eye(3) * g.diagonal(axis1=1, axis2=2)[:, None, :].clip(0.01),
            controls.transpose(0, 2, 1) @ s,
        )
        s -= controls @ beta
        truth = co / k * np.expm1(path[stops + 1] - path[origin + 1])
        late = co / k * np.expm1(path[stops + 1] - path[origin + 21])
        y = center(np.stack((truth, late), axis=-1))
        sd = np.sqrt((y**2).sum(1) / mass + 0.08)
        y = np.clip(y / sd[:, None, :], -3, 3)
        cov = np.einsum("tsj,tsh->tjh", s, y) / mass[..., None]
        exposure = (s**2).sum(1) / mass
        alpha = 1 - np.exp(-5 / 320)

        def ema(v):
            return lfilter([alpha], [1, alpha - 1], v, axis=0)

        # Cross-sectional observations are correlated; shrink each update toward zero.
        eligible = mass[:, 0] >= 25

        def skill(mask):
            return (
                ema(cov * mask[:, None, None])
                / (0.065 + ema(exposure * mask[:, None]))[:, :, None]
            )

        w = skill(eligible)
        halves = [skill(eligible & ((origin // 60) % 2 == j)) for j in (0, 1)]
        stable = consensus.agreement.consensus(halves[0][:, :, 0], halves[1][:, :, 0])
        t = np.arange(stops[0], len(q))
        ix = (t - stops[0]) // 5

        def read(weight):
            return (
                np.einsum("tsj,tj->ts", signal[t], weight[ix].clip(-1.0, 1.0))
                / signal.shape[2]
            )

        out[a + t] = np.stack((read(w[:, :, 0]), read(w[:, :, 1]), read(stable)), -1)
    return out - out.mean(1, keepdims=True)


def views(raw, grid, code, bank, p):
    z = consensus.views(raw, grid, code, bank, p)
    return update(raw, grid, p, z)


if __name__ == "__main__":
    check_template()
    consensus.net.rel.main(OUT, ARMS, train, views, with_prediction=True)
