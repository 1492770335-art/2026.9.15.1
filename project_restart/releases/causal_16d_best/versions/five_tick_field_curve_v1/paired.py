"""Estimate the small innovation signal with true aligned history/future pairs rather than stationary lag substitution."""

import numpy as np
import history_head as path
from history_head import fields, news, memory, ref

OUT = ref.base.ROOT / "artifacts/dense_paired_field_innovations_v1"


def sample(z, grid, core):
    total = 0
    count = 0
    for a, b in fields.parent.sessions(grid):
        for t in np.array_split(np.arange(a + 20, b - 61, 8), 6):
            if not len(t):
                continue
            # The same instruments and forecast origins enter all five terms of the cancellation.
            target = np.concatenate(
                (
                    z[t + 20, ..., core:] + z[t + 40, ..., core:],
                    z[t + 20, ..., :core] + z[t + 40, ..., :core],
                    z[t, ..., :core] + z[t + 20, ..., :core],
                    z[t - 20, ..., :core] + z[t, ..., :core],
                ),
                axis=-1,
            )
            current = z[t, ..., :core].reshape(-1, core).astype(float)
            total += current.T @ target.reshape(-1, target.shape[-1]).astype(float)
            count += len(current)
    return total / count


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        load = lambda k: np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")
        raw, grid = load("x"), load("time").astype(int)
        x, av = fields.parent.state(raw, grid, fields.parent.codec(), with_market=True)
        core = x.shape[-1]
        m = fields.book_inputs(raw, grid, ref.market.market_state(av, grid), {})
        omitted = np.tanh(0.65 * np.arcsinh(raw[:, :, fields.COLUMNS].astype(float)))
        omitted -= omitted.mean(1, keepdims=True)
        z = np.concatenate((x, omitted.astype(np.float32)), axis=-1)
        np.savez(
            OUT / f"day_{d:02d}.npz",
            stock=sample(z, grid, core),
            market=sample(m, grid, core),
        )
