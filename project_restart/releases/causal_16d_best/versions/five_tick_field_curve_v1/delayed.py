"""Find delayed return information in a historical book that is absent from the forecast of the future book."""

import numpy as np
from memory import ref

parent = ref.market.parent
OUT = ref.base.ROOT / "artifacts/dense_hidden_payoff_memory_v1"


def collect():
    OUT.mkdir(exist_ok=True)
    code = parent.codec()
    for day in range(16):
        load = lambda key: np.load(
            ref.base.CACHE / f"day_{day:02d}_{key}.npy", mmap_mode="r"
        )
        raw, grid, y = load("x"), load("time").astype(int), load("y")[..., 1]
        x, av = parent.state(raw, grid, code, with_market=True)
        m = ref.market.market_state(av, grid)
        moments = np.zeros((2, 2, 2, m.shape[1]))
        count = 0
        for a, b in parent.sessions(grid):
            t = np.arange(a + 40, b - 61, 6)
            if not len(t):
                continue
            for j, lag in enumerate((20, 40)):
                label = y[t + lag].astype(float)
                common = label.mean(1)
                rel = (label - common[:, None]).ravel()
                for i, idx in enumerate((t, t - lag)):
                    moments[0, j, i] += x[idx].reshape(-1, m.shape[1]).T @ rel / 50
                    moments[1, j, i] += m[idx].T @ common
            count += len(t)
        np.save(OUT / f"payoffs_{day:02d}.npy", moments / count)
