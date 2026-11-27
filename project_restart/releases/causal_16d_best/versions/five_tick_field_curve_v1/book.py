import numpy as np
from scipy.signal import lfilter
import settings as base

COLS = np.arange(77, 83)


def metric(p, y, z):
    flat = np.stack([p.ravel(), y.ravel(), z.ravel()]).astype(float)
    return np.pad(flat @ flat.T, ((1, 0), (1, 0))) + np.block(
        [
            [np.array([[flat.shape[1]]]), flat.sum(1)[None]],
            [flat.sum(1)[:, None], np.zeros((3, 3))],
        ]
    )


def gain(g):
    cov = g[1:, 1:] / g[0, 0] - np.outer(g[0, 1:], g[0, 1:]) / g[0, 0] ** 2
    p, y, z = cov.diagonal()
    return float(
        (cov[0, 1] + cov[1, 2]) / np.sqrt((p + 2 * cov[0, 2] + z) * y)
        - cov[0, 1] / np.sqrt(p * y)
    )


def sessions(grid):
    cuts = (
        np.flatnonzero((np.diff(grid) != 1) | ((grid[:-1] < 2401) & (grid[1:] >= 2401)))
        + 1
    )
    return zip(np.r_[0, cuts], np.r_[cuts, len(grid)])


def smooth(x, tau):
    alpha = 1 - np.exp(-1 / tau)
    return lfilter([alpha], [1, alpha - 1], x - x[:1], axis=0) + x[:1]


def codec():
    raw = np.load(base.CACHE / "day_00_x.npy", mmap_mode="r")
    rank = np.column_stack((np.ones(32), np.log1p(3 * np.arange(32))))
    return np.stack(
        [
            np.linalg.lstsq(
                rank, np.unique(raw[:, :, j])[:32].astype(float), rcond=None
            )[0]
            for j in COLS
        ]
    )


def decode(s, code):
    return np.expm1((s.astype(float) - code[:, 0]) / code[:, 1]) / 3


def history(x, tau):
    alpha = 1 - np.exp(-1 / tau)
    return lfilter([alpha], [1, alpha - 1], x, axis=0)


def sources(raw, grid, code):
    chunks = []
    for a, b in sessions(grid):
        s = raw[a:b]
        price = np.tanh(s[:, :, 0].astype(float) - 0.00136)
        book = s[:, :, [26, 38, 41, 42, 54, 55]].astype(float)
        imb = np.tanh(0.65 * (book[:, :, ::2] - book[:, :, 1::2]))
        counts = np.rint(decode(s[:, :, 77:81], code[:4])).clip(0, 512)
        left, right, u, v = np.moveaxis(counts, 2, 0)

        def direction(l, r):
            return np.tanh((l - r) / np.sqrt(1 + l + r + 0.025 * (l + r) ** 2))

        levels = np.stack(
            (
                direction(u, v),
                direction((left - u).clip(0), (right - v).clip(0)),
                price,
            ),
            2,
        )
        src = np.concatenate((imb, levels), 2)
        fast = smooth(src, 8)
        slow = smooth(src, 50)
        chunks.append(
            np.concatenate((src - fast, fast - slow, slow), 2).astype("float32")
        )
    return np.concatenate(chunks)
