"""Fit stock-level book corrections with separable cross-stock and overlapping-horizon noise models."""

import numpy as np
import experiment_dense_relative_path_inputs_v1 as rel

ref, fields, fine = rel.ref, rel.fields, rel.fine
OUT = ref.base.ROOT / "artifacts/dense_panel_noise_readout_v1"
ARMS = ("stock_precision", "overlap_precision", "joint_precision")
LAG = 10


def noise_prior():
    current = np.zeros((50, 50))
    past = current.copy()
    for d in range(4):
        y = np.load(ref.base.CACHE / f"day_{d:02d}_y.npy", mmap_mode="r")[
            ..., 2
        ].astype(float)
        y -= y.mean(1, keepdims=True)
        grid = np.load(ref.base.CACHE / f"day_{d:02d}_time.npy").astype(int)
        for a, b in fields.parent.sessions(grid):
            t = np.arange(a + LAG, b - 61, 4)
            now = y[t]
            before = y[t - LAG]
            current += 0.5 * (now.T @ now + before.T @ before)
            past += before.T @ now
    rho = np.clip(np.trace(past) / np.trace(current), 0, 0.90)
    innovations = (1 + rho**2) * current - rho * (past + past.T)

    def precision(c):
        c = c / np.mean(c.diagonal())
        k = 0.5 * c + 0.25 * np.diag(c.diagonal()) + 0.25 * np.eye(50)
        v = np.linalg.inv(k)
        return v / np.mean(v.diagonal())

    return rho, precision(current), precision(innovations)


def book(raw, grid, code):
    x = fields.parent.state(raw, grid, code)
    # The readout can assign different coefficients to slow and fast changes in the same fields.
    out = np.empty((*x.shape[:2], x.shape[2] + 38), dtype=np.float32)
    out[:, :, : x.shape[2]] = x
    for a, b in fields.parent.sessions(grid):
        block = x[a:b, :, :38]
        out[a:b, :, x.shape[2] :] = block - fields.parent.smooth(block, 32)
    return out


def collect():
    OUT.mkdir(exist_ok=True)
    rho, stock, whitened = noise_prior()
    np.savez(OUT / "noise.npz", rho=rho, stock=stock, whitened=whitened)
    code = fields.parent.codec()
    for d in range(4, 16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = (
            load("x"),
            load("time").astype(int),
            load("y")[..., 2].astype(float),
        )
        x = book(raw, grid, code)
        if d < 6:
            p = np.load(ref.market.TAPE / f"forecast_{d:02d}.npy")[..., 2]
        else:
            with np.load(ref.tape.OUT / f"day_{d:02d}.npz") as f:
                p = f["reference"][..., 2]
            if d >= 8:
                for m in (
                    ref.precision,
                    ref.nonlinear,
                    ref.clock,
                    fine.memory,
                    fine.news,
                    fine.path,
                    fine,
                ):
                    p += np.load(m.OUT / f"oof_{d:02d}.npy")[:, 2, None]
        p = p.astype(float)
        p -= p.mean()
        y -= y.mean()
        slope = (p * y).sum() / (p * p).sum()
        error = y / slope.clip(0.25, 4) - p
        error -= error.mean(1, keepdims=True)
        width = x.shape[2]
        stats = np.zeros((4, width, width + 1))
        n = 0
        for a, b in fields.parent.sessions(grid):
            for t in np.array_split(np.arange(a + LAG, b - 61, 4), 8):
                if not len(t):
                    continue
                v = x[t].astype(float)
                e = error[t]
                dv = v - rho * x[t - LAG]
                de = e - rho * error[t - LAG]
                for j, (src, out, w) in enumerate(
                    ((v, e, None), (v, e, stock), (dv, de, None), (dv, de, whitened))
                ):
                    z = np.concatenate((src, out[..., None]), 2)
                    if w is not None:
                        z = w @ z
                    stats[j] += src.reshape(-1, width).T @ z.reshape(-1, width + 1)
                n += len(t) * 50
        np.save(OUT / f"day_{d:02d}.npy", stats / n)


def train(days):
    stats = np.stack([np.load(OUT / f"day_{d:02d}.npy") for d in range(4, days)])
    weights = []
    for j in range(1, 4):
        gram = stats[:, j, :, :-1].mean(0)
        gradient = stats[:, j, :, -1]
        sd = np.sqrt(gram.diagonal()).clip(0.01)
        cov = gram / np.outer(sd, sd)
        val, u = np.linalg.eigh((cov + cov.T) * 0.5)
        daily = (gradient / sd) @ u
        mean = daily.mean(0)
        noise = daily.var(0, ddof=1) / len(daily)
        gain = mean**2 / (mean**2 + noise + 1e-10)
        weights.append(u @ (gain * mean / (val.clip(0) + 0.65)) / sd)
    bank = np.column_stack(weights)
    var = np.einsum("ij,ik,kj->j", bank, stats[:, 0, :, :-1].mean(0), bank)
    return bank / np.sqrt(var.clip(1e-12))


def views(raw, grid, code, bank):
    v = book(raw, grid, code) @ bank
    return v - v.mean(1, keepdims=True)


if __name__ == "__main__":
    collect()
    rel.main(OUT, ARMS, train, views)
