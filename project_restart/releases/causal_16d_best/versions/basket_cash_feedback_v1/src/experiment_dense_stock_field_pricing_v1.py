"""Separate stock-specific dynamics of a future book innovation from its stock-specific payoff."""

import numpy as np
import experiment_dense_relative_path_inputs_v1 as relative

fine, fields, ref = relative.fine, relative.fields, relative.ref
OUT = ref.base.ROOT / "artifacts/dense_stock_field_pricing_v1"
ARMS = ("stock_transition", "stock_price", "price_and_transition")


def basis():
    grams = []
    cross = []
    import experiment_dense_exact_cover_fields_v1 as curves

    for d in range(4):
        with (
            np.load(fields.OUT / f"moments_{d:02d}.npz") as f,
            np.load(curves.OUT / f"day_{d:02d}.npz") as g,
        ):
            grams.append(f["cov"][0, 0])
            cross.append(g["cross"][0])
    cov = np.mean(grams, 0)
    sd = np.sqrt(cov.diagonal()).clip(0.03)
    n = len(sd) - len(fields.COLUMNS)
    k = np.mean(cross, 0) / np.outer(sd[:n], sd)
    gram = cov[:n, :n] / np.outer(sd[:n], sd[:n])
    decoder = fine.news.train(range(4), stock_fraction=1.0, teacher_decoder=True)[
        "bank"
    ]
    nf = len(fields.COLUMNS)
    price = decoder[:nf]
    lags = decoder[nf:].reshape(3, n)

    def lag(h):
        return k[curves.LAGS.index(h)]

    forecasts = []
    for h in range(20, 60, 5):
        gradient = lag(h)[:, n:] @ price + sum(
            lag(h - 20 * j)[:, :n] @ w for j, w in enumerate(lags)
        )
        forecasts.append(np.linalg.solve(gram + 0.22 * np.eye(n), gradient) / sd[:n])
    return np.column_stack(
        (np.mean(forecasts[:4], 0), np.mean(forecasts[4:], 0))
    ), decoder / np.r_[sd[n:], np.tile(sd[:n], 3)]


def collect():
    OUT.mkdir(exist_ok=True)
    bank, decoder = basis()
    np.save(OUT / "basis.npy", bank)
    code = fields.parent.codec()
    nf = len(fields.COLUMNS)
    for d in range(4, 16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = load("x"), load("time").astype(int), load("y")[..., 1]
        x = fields.parent.state(raw, grid, code)
        forecast = x @ bank
        field = np.tanh(0.65 * np.arcsinh(raw[:, :, fields.COLUMNS].astype(float)))
        field -= field.mean(1, keepdims=True)
        vals = field @ decoder[:nf]
        stats = 0
        count = 0
        for a, b in fields.parent.sessions(grid):
            for j, w in enumerate(decoder[nf:].reshape(3, -1)):
                t = np.arange(a + 40, b)
                vals[t] += x[t - 20 * j] @ w
            t = np.arange(a + 40, b - 61, 5)
            observed = np.stack(
                [
                    sum(vals[t + h] for h in steps) / 4
                    for steps in (range(20, 40, 5), range(40, 60, 5))
                ],
                -1,
            )
            yrel = y[t] - y[t].mean(1, keepdims=True)
            z = np.concatenate(
                (forecast[t], observed, vals[t, :, None], yrel[..., None]), -1
            )
            stats += np.einsum("tij,tik->ijk", z, z, optimize=True)
            count += len(t)
        np.save(OUT / f"day_{d:02d}.npy", stats / count)


def shrink(observations):
    # Estimate each receiver's persistent deviation using variation between training dates.
    pool = observations.mean(1, keepdims=True)
    centered = observations - pool
    mean = centered.mean(0)
    noise = centered.var(0, ddof=1).mean(0) / len(observations)
    overall = (mean**2).mean(0)
    reliability = (1 - noise / overall.clip(1e-14)).clip(0, 1)
    return mean * reliability


def train(days):
    g = np.stack([np.load(OUT / f"day_{d:02d}.npy") for d in range(4, days)])
    v = g.diagonal(axis1=2, axis2=3)
    # Fit the value of a field innovation and how much of its future realization a name's current book predicts.
    slopes = np.stack(
        [
            g[:, :, i, i + 2]
            / (v[:, :, i] + 0.3 * v[:, :, i].mean(1, keepdims=True)).clip(1e-12)
            for i in (0, 1)
        ],
        2,
    )
    price = g[:, :, 4, 5] / (v[:, :, 4] + 0.3 * v[:, :, 4].mean(1, keepdims=True)).clip(
        1e-12
    )
    ds = shrink(slopes)
    dp = shrink(price[:, :, None])
    s0 = slopes.mean((0, 1))
    p0 = price.mean()
    return np.stack((ds * p0, dp * s0, ds * p0 + dp * s0 + dp * ds), -1)


def views(raw, grid, code, weights):
    x = fields.parent.state(raw, grid, code)
    forecast = x @ np.load(OUT / "basis.npy")
    z = np.einsum("tij,ijk->tik", forecast, weights)
    z -= z.mean(1, keepdims=True)
    for a, b in fields.parent.sessions(grid):
        z[a : min(a + 40, b)] = 0
    return z


if __name__ == "__main__":
    collect()
    relative.main(OUT, ARMS, train, views)
