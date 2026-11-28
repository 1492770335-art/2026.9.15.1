"""Distill the current book's continuation value from shorter forecasts issued later on training dates."""

import numpy as np
import experiment_dense_panel_noise_readout_v1 as panel

rel, ref = panel.rel, panel.ref
OUT = ref.base.ROOT / "artifacts/dense_future_expert_value_v1"
ARMS = ("three_twenty_tick_values", "twelve_five_tick_values", "replicated_surprise")


def forecast(day):
    if day < 6:
        return np.load(ref.market.TAPE / f"forecast_{day:02d}.npy")
    with np.load(ref.tape.OUT / f"day_{day:02d}.npz") as f:
        p = f["reference"]
    if day >= 8:
        for m in (
            ref.precision,
            ref.nonlinear,
            ref.clock,
            rel.fine.memory,
            rel.fine.news,
            rel.fine.path,
            rel.fine,
        ):
            p += np.load(m.OUT / f"oof_{day:02d}.npy")[:, None]
    return p


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4, 16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = load("x"), load("time").astype(int), load("y").astype(float)
        x = panel.book(raw, grid, code)
        p = forecast(d).astype(float)
        p -= p.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        scale = ((p * y).sum((0, 1)) / (p * p).sum((0, 1))).clip(0.2, 4)
        gram = 0
        cross = 0
        count = 0
        for a, b in rel.fields.parent.sessions(grid):
            for t in np.array_split(np.arange(a, b - 60, 4), 6):
                if not len(t):
                    continue
                medium = (
                    sum(p[t + h, :, 1] for h in range(0, 60, 20))
                    * ref.base.UNITS[1]
                    * scale[1]
                )
                short = (
                    sum(p[t + h, :, 0] for h in range(0, 60, 5))
                    * ref.base.UNITS[0]
                    * scale[0]
                )
                read = np.stack((medium, short, y[t, :, 2]), -1) / scale[2]
                xx = (
                    np.concatenate((p[t], x[t]), 2)
                    .reshape(-1, 3 + x.shape[-1])
                    .astype(float)
                )
                gram += xx.T @ xx
                cross += xx.T @ read.reshape(-1, 3)
                count += len(xx)
        np.savez(OUT / f"day_{d:02d}.npz", gram=gram / count, cross=cross / count)


def train(days):
    rows = []
    for d in range(4, days):
        with np.load(OUT / f"day_{d:02d}.npz") as f:
            rows.append((f["gram"], f["cross"]))
    gs, xy = map(np.stack, zip(*rows))
    # Identify information outside the current three forecasts separately on each training date.
    beta = np.linalg.solve(
        gs[:, :3, :3]
        + 0.01 * np.eye(3) * gs[:, :3, :3].diagonal(axis1=1, axis2=2)[:, None],
        gs[:, :3, 3:],
    )
    controls = beta.transpose(0, 2, 1)
    g = (
        gs[:, 3:, 3:]
        - controls @ gs[:, :3, 3:]
        - gs[:, 3:, :3] @ beta
        + controls @ gs[:, :3, :3] @ beta
    ).mean(0)
    rhs = xy[:, 3:] - controls @ xy[:, :3]
    sd = np.sqrt(g.diagonal()).clip(0.015)
    val, u = np.linalg.eigh(g / np.outer(sd, sd))
    signal = u.T @ (rhs / sd[None, :, None])
    mean = signal.mean(0)
    # The two future experts are independent supervision signals for the delayed part of the return.
    common = 0.5 * (mean[:, 0] + mean[:, 1])
    errors = signal[:, :, 2] - 0.5 * (signal[:, :, 0] + signal[:, :, 1])
    bias = errors.mean(0)
    se = errors.std(0, ddof=1) / np.sqrt(len(errors))
    replicated = common + bias * abs(bias) / (abs(bias) + se + 1e-8)
    basis = np.column_stack((mean[:, 0], mean[:, 1], replicated))
    bank = (u @ (basis / (val[:, None].clip(0) + 0.55))) / sd[:, None]
    risk = np.einsum("ij,ik,kj->j", bank, gs.mean(0)[3:, 3:], bank)
    return bank / np.sqrt(risk.clip(1e-12))


if __name__ == "__main__":
    collect()
    rel.main(OUT, ARMS, train, panel.views)
