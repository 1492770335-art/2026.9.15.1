"""Pool stock field-response parameters over names with repeatable early book behavior."""

import numpy as np
import experiment_dense_stock_field_pricing_v1 as stock

OUT = stock.ref.base.ROOT / "artifacts/dense_peer_shrunk_field_prices_v1"
ARMS = ("peer_price_and_speed", "name_residual", "peer_plus_name")


def profiles():
    code = stock.fields.parent.codec()
    hist = []
    for day in range(4):
        grid = np.load(stock.ref.base.CACHE / f"day_{day:02d}_time.npy").astype(int)
        raw = np.load(stock.ref.base.CACHE / f"day_{day:02d}_x.npy", mmap_mode="r")
        state = stock.fields.parent.state(raw, grid, code)
        g = 0
        count = 0
        for a, b in stock.fields.parent.sessions(grid):
            t = np.arange(a + 40, b, 4)
            x = state[t].astype(float)
            g += np.stack(
                [(x * x).sum(0), (x * state[t - 5]).sum(0), (x * state[t - 40]).sum(0)]
            )
            count += len(t)
        var, near, far = g / count
        hist.append(
            np.concatenate(
                (np.log(var.clip(1e-6)), near / var.clip(1e-6), far / var.clip(1e-6)), 1
            )
        )
    hist = np.stack(hist)
    # Standardize across stocks separately each day; emphasize features reproducible on alternating dates.
    hist -= hist.mean(1, keepdims=True)
    hist /= hist.std(1, keepdims=True).clip(0.015)
    left, right = hist[::2].mean(0), hist[1::2].mean(0)
    agreement = (left * right).mean(0).clip(0, 1)
    means = hist.mean(0)
    distance = ((means[:, None] - means[None, :]) ** 2 * agreement).mean(2)
    np.fill_diagonal(distance, np.inf)
    scale = np.partition(distance, 7, axis=1)[:, 7].clip(1e-8)
    weights = np.exp(-2 * distance / scale[:, None])
    return weights / weights.sum(1, keepdims=True)


def train(days):
    g = np.stack([np.load(stock.OUT / f"day_{d:02d}.npy") for d in range(4, days)])
    v = g.diagonal(axis1=2, axis2=3)
    peer = np.load(OUT / "peers.npy")

    def coefficients(moments):
        denom = moments.diagonal(axis1=2, axis2=3)
        slope = moments[:, :, (0, 1), (2, 3)] / (
            denom[:, :, :2] + 0.3 * v[:, :, :2].mean(1, keepdims=True)
        ).clip(1e-12)
        price = moments[:, :, 4, 5] / (
            denom[:, :, 4] + 0.3 * v[:, :, 4].mean(1, keepdims=True)
        ).clip(1e-12)
        return slope * price[:, :, None]

    own = coefficients(g)
    group = coefficients(np.einsum("ij,djkl->dikl", peer, g, optimize=True))
    # Cross-sectional deviations alone carry stock allocation information.
    beta = group.mean(0)
    beta -= beta.mean(0)
    resid = stock.shrink(own - group)
    bank = np.stack((beta, resid, beta + resid), -1)
    risk = np.einsum("ihj,ihk,ikj->j", bank, g.mean(0)[:, :2, :2], bank) / len(bank)
    return bank / np.sqrt(risk.clip(1e-12))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "peers.npy", profiles())
    stock.relative.main(OUT, ARMS, train, stock.views)
