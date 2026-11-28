"""Use unpredictable shorter-horizon peer returns as training-only controls for the stock-relative label."""

import json
import numpy as np
import experiment_dense_future_expert_value_v1 as future
import experiment_dense_peer_shrunk_field_prices_v1 as peers

rel, ref, panel = future.rel, future.ref, future.panel
OUT = ref.base.ROOT / "artifacts/dense_peer_noise_control_v1"
ARMS = ("book_peer_hedge", "shock_peer_hedge", "own_unshared_shock")
views = panel.views


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4, 16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = load("x"), load("time").astype(int), load("y").astype(float)
        x = panel.book(raw, grid, code)
        p = future.forecast(d).astype(float)
        p -= p.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        scale = ((p * y).sum((0, 1)) / (p * p).sum((0, 1))).clip(0.2, 4)
        err = y[..., 2] / scale[2] - p[..., 2]
        short = (y[..., 0] - scale[0] * p[..., 0]) * ref.base.UNITS[0] / scale[2]
        t = np.concatenate(
            [np.arange(a, b - 60, 4) for a, b in rel.fields.parent.sessions(grid)]
        )
        u = sum(short[t + i] for i in range(0, 60, 5))
        xx = np.concatenate((p[t], x[t]), axis=2)
        # Store the contribution of every sender's future shock to each receiver's training gradient.
        xu = xx.transpose(1, 2, 0) @ u / len(t)
        np.savez(
            OUT / f"day_{d:02d}.npz",
            xu=xu,
            uu=u.T @ u / len(t),
            ue=u.T @ err[t] / len(t),
            ee=err[t].T @ err[t] / len(t),
        )


def control(stats):
    uu, ue = np.mean([v[0] for v in stats], 0), np.mean([v[1] for v in stats], 0)
    corr = uu / np.sqrt(np.outer(uu.diagonal(), uu.diagonal())).clip(1e-9)
    np.fill_diagonal(corr, -np.inf)
    book = np.load(peers.OUT / "peers.npy")
    names = np.arange(50)
    banks = []
    for geometry in (book, corr):
        near = np.argsort(-geometry, axis=1)[:, :6]
        w = np.zeros_like(uu)
        for i, neighbors in enumerate(near):
            assert i not in neighbors
            cov = uu[np.ix_(neighbors, neighbors)]
            w[neighbors, i] = np.linalg.solve(
                cov + 0.55 * np.diag(cov.diagonal().clip(0.01)), ue[neighbors, i]
            )
        banks.append(w)
    # Estimate the part of the same-name short error orthogonal to its neighbors.
    w = np.zeros_like(uu)
    for i, neighbors in enumerate(np.argsort(-corr, axis=1)[:, :6]):
        cov = uu[np.ix_(neighbors, neighbors)]
        w[i, i] = 1
        w[neighbors, i] = -np.linalg.solve(
            cov + 0.55 * np.diag(cov.diagonal().clip(0.01)), uu[neighbors, i]
        )
    beta = np.diag(w.T @ ue) / (np.diag(w.T @ uu @ w) + 0.3 * uu.diagonal()).clip(1e-6)
    banks.append(w * beta[None, :])
    return np.stack(banks)


def train(days):
    data = []
    for d in range(4, days):
        with (
            np.load(OUT / f"day_{d:02d}.npz") as r,
            np.load(future.OUT / f"day_{d:02d}.npz") as f,
        ):
            data.append((r["uu"], r["ue"], r["xu"], f["gram"], f["cross"][:, 2]))
    covariance = []
    gradients = []
    for side in (0, 1):
        w = control(data[1 - side :: 2])
        for uu, ue, xu, g, truth in data[side::2]:
            c = g[:3, :3]
            proj = np.linalg.solve(c + 0.01 * np.diag(c.diagonal()), g[:3, 3:])
            cov = g[3:, 3:] - proj.T @ g[:3, 3:] - g[3:, :3] @ proj + proj.T @ c @ proj
            rhs = truth - g[:, 2]
            hedge = np.einsum("ifj,aji->fa", xu, w, optimize=True) / 50
            gradient = rhs[:, None] - hedge
            covariance.append(cov)
            gradients.append(gradient[3:] - proj.T @ gradient[:3])
    g = np.mean(covariance, 0)
    sd = np.sqrt(g.diagonal()).clip(0.015)
    e, u = np.linalg.eigh(g / np.outer(sd, sd))
    daily = u.T @ (np.stack(gradients) / sd[None, :, None])
    avg = daily.mean(0)
    noise = daily.std(0, ddof=1) / np.sqrt(len(daily))
    bank = (
        u @ (avg * abs(avg) / (abs(avg) + noise + 1e-9) / (e[:, None].clip(0) + 0.55))
    ) / sd[:, None]
    denom = np.mean([r[3][3:, 3:] for r in data], 0)
    var = np.einsum("ij,ik,kj->j", bank, denom, bank)
    return bank / np.sqrt(var.clip(1e-12))


def audit():
    all_days = []
    metrics = []
    for d in range(4, 16):
        with np.load(OUT / f"day_{d:02d}.npz") as f:
            uu, ue, ee = f["uu"], f["ue"], f["ee"]
        if d in (8, 12):
            bank = control(all_days)
        if d >= 8:
            metrics.append(
                [
                    (2 * np.trace(w.T @ ue) - np.trace(w.T @ uu @ w)) / np.trace(ee)
                    for w in bank
                ]
            )
        all_days.append((uu, ue))
    arr = np.array(metrics)
    (OUT / "label_noise_audit.json").write_text(
        json.dumps(
            dict(
                arms=ARMS,
                days=arr.tolist(),
                folds=np.stack((arr[:4].mean(0), arr[4:].mean(0))).tolist(),
            ),
            indent=2,
        )
    )
    print("Variance reductions", arr.mean(0), flush=True)


if __name__ == "__main__":
    collect()
    audit()
    rel.main(OUT, ARMS, train, views)
