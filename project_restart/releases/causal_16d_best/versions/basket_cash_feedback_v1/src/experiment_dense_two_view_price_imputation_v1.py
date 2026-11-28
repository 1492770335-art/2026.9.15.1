"""Learn to recover a deliberately hidden price change from disjoint stock and book measurements."""
import numpy as np
import experiment_dense_asynchronous_next_print_v1 as previous

onset, rel, book, REFERENCE = previous.onset, previous.rel, previous.book, previous.REFERENCE
OUT = REFERENCE.parent / "dense_two_view_price_imputation_v1"
ARMS = ("other_stocks_only", "own_book_only", "two_views_agree")
FIELDS = [5, 10, 14, 24, 39, 40, 41, 42, 52, 53, 54, 55]
WIDTH = 51 + len(FIELDS)
WINDOW = np.array([1, 3, 7, 15, 31, 63])


def inputs(level, view, t, origin, stock):
    norm = np.sqrt((t - origin).clip(1))[:, None]
    donors = (level[t] - level[origin]) / norm
    donors[:, stock] = 0
    common = donors.sum(1) / 49
    donors -= common[:, None]
    donors[:, stock] = 0
    quotes = (view[t, stock] - view[origin, stock]) / norm
    return np.column_stack((common, donors, quotes))


def collect():
    OUT.mkdir(exist_ok=True)
    for day in range(16):
        raw, grid = [onset.settled.load(day, key) for key in ("x", "time")]
        gram = np.zeros((50, WIDTH + 1, WIDTH + 1))
        for start, end, valid, _, level, _ in previous.sessions(raw, grid):
            view = np.arcsinh(raw[start:end][:, :, FIELDS].astype(np.float32))
            missing = (~valid).cumsum(0)
            t = np.arange(64 + day % 3, len(level), 3)
            for stock in range(50):
                # The hidden interval is sampled independently of observed price movements.
                length = WINDOW[(np.arange(len(t)) + stock + day) % len(WINDOW)]
                origin = t - length
                x = inputs(level, view, t, origin, stock)
                y = (level[t, stock] - level[origin, stock]) / np.sqrt(length)
                ok = (missing[t, stock] == missing[origin, stock]) & valid[origin, stock]
                z = np.column_stack((x, y))[ok]
                gram[stock] += z.T @ z
        np.save(OUT / f"gram_{day:02d}.npy", gram)


def train(end):
    g = sum(np.load(OUT / f"gram_{d:02d}.npy") for d in range(end))
    w = np.zeros((50, WIDTH, 2))
    for stock in range(50):
        for arm, idx in enumerate((slice(0, 51), slice(51, WIDTH))):
            cov = g[stock, idx, idx]
            rhs = g[stock, idx, -1]
            # The price scale is in raw accumulated return units for every stock.
            w[stock, idx, arm] = previous.base.own.solve(cov, rhs, 1.)
    return w


def design(raw, grid, code, pred, bank):
    out = np.zeros((*pred.shape[:2], len(ARMS), 3))
    for start, end, observed, _, level, origin in previous.sessions(raw, grid):
        view = np.arcsinh(raw[start:end][:, :, FIELDS].astype(np.float32))
        t = np.arange(len(level))
        for stock in range(50):
            # Limit unobserved time to the range covered by training.
            old = np.maximum(origin[:, stock], t - WINDOW[-1])
            age = t - old
            z = inputs(level, view, t, old, stock) @ bank[stock]
            z *= (np.sqrt(age) * observed[:, stock])[:, None]
            other, own = z.T
            agree = np.where(other * own > 0, np.sign(other) * np.minimum(abs(other), abs(own)), 0.)
            out[start:end, stock] = np.column_stack((other, own, agree))[:, :, None]
    return out - out.mean(1, keepdims=True)


def predict(raw, grid, p, params):
    return book.path.corrected(p, design(raw, grid, params["code"], p, params["bank"]), params)


if __name__ == "__main__":
    collect()
    onset.rolling(OUT, ARMS, train, design, REFERENCE)
