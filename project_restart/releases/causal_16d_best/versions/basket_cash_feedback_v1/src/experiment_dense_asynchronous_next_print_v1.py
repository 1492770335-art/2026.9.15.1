"""Resolve the return before and after a stock's next price change from peer moves during its stale interval."""
import numpy as np
import experiment_dense_factor_error_covariance_v1 as base

onset, rel, book, REFERENCE = base.onset, base.rel, base.book, base.REFERENCE
OUT = REFERENCE.parent / "dense_asynchronous_next_print_v1"
ARMS = ("until_next_print", "after_next_print", "whole_interval")
HORIZONS = (5, 20, 60)
# Intercept, current forecast, leave-one-out mean, then individual peers.
CONTROL = 4
WIDTH = CONTROL + 51
AGES = 3


def sessions(raw, grid):
    with np.load(onset.settled.GEO) as codec:
        k = float(codec["k"])
        coefficient = codec["coeff"][2, 1]
        missing = codec["row"]
    for start, end in rel.fields.parent.sessions(grid):
        rows = raw[start:end]
        observed = ~np.all(rows == missing, axis=2)
        step = np.log1p(k * (rows[:, :, 0].astype(float) - .00136)) * coefficient / k
        step *= observed
        t = np.arange(end - start)[:, None]
        changes = np.abs(step) > .003
        origin = np.maximum.accumulate(np.where(changes | ~observed, t, 0), axis=0)
        yield start, end, observed, changes, step.cumsum(0), origin


def panel(level, origin, t, stock, pred):
    age = t - origin[t, stock]
    x = np.zeros((len(t), WIDTH))
    x[:, 0] = 1
    x[:, 1:CONTROL] = pred[t, stock]
    change = (level[t] - level[origin[t, stock]]) / np.sqrt(1 + age[:, None])
    change[:, stock] = 0
    other = change.sum(1) / 49
    x[:, CONTROL] = other
    x[:, CONTROL + 1:] = change - other[:, None]
    x[:, CONTROL + 1 + stock] = 0
    age_bin = (age >= 4).astype(int) + (age >= 12)
    return x, age, age_bin


def collect():
    OUT.mkdir(exist_ok=True)
    for day in range(4, 16):
        raw, grid = [onset.settled.load(day, field) for field in ("x", "time")]
        p = (np.load(REFERENCE / f"rolling_{day:02d}.npy") if day >= 6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(day))
        n = WIDTH + 2 * len(HORIZONS)
        grams = np.zeros((50, AGES, n, n))
        for start, end, valid, changes, level, origin in sessions(raw, grid):
            steps = np.arange(len(level))
            # All forward lookups here are targets, computed on the fitting day only.
            future = np.minimum.accumulate(np.where(changes, steps[:, None], len(level))[::-1], axis=0)[::-1]
            missing = (~valid).cumsum(0)
            t = np.arange(day % 3, len(level) - max(HORIZONS), 3)
            for stock in range(50):
                x, age, bins = panel(level, origin, t, stock, p[start:end])
                ok = (age > 0) & valid[t, stock] & (missing[t + max(HORIZONS), stock] == missing[t, stock])
                stop = future[t + 1, stock]
                # Subtract contemporaneous market moves over exactly the same lengths.
                relative = level[:, stock] - level.mean(1)
                first, rest = [], []
                for horizon in HORIZONS:
                    expiry = t + horizon
                    middle = np.minimum(expiry, stop)
                    first.append(relative[middle] - relative[t])
                    rest.append(relative[expiry] - relative[middle])
                row = np.column_stack((x, *first, *rest))
                for group in range(AGES):
                    z = row[ok & (bins == group)]
                    grams[stock, group] += z.T @ z
        np.save(OUT / f"gram_{day:02d}.npy", grams)


def train(end):
    grams = sum(np.load(OUT / f"gram_{d:02d}.npy") for d in range(4, end))
    weights = np.zeros((50, AGES, WIDTH, 6))
    for stock in range(50):
        per_name = grams[stock].sum(0)
        for group, local in enumerate(grams[stock]):
            cov = local + .3 * (local[0, 0] + 1) * per_name / max(per_name[0, 0], 1)
            cov = cov / max(cov[0, 0], 1)
            # Partial out the existing forecast, then fit the price-change payoff.
            transform = np.eye(WIDTH)[:, CONTROL:]
            transform[:CONTROL] = -base.own.solve(cov[:CONTROL, :CONTROL],
                                                    cov[:CONTROL, CONTROL:WIDTH], .06)
            clean = transform.T @ cov[:WIDTH, :WIDTH] @ transform
            rhs = transform.T @ cov[:WIDTH, WIDTH:]
            weights[stock, group] = transform @ base.own.solve(clean, rhs, 2.5)
    return weights


def design(raw, grid, code, pred, bank):
    out = np.zeros((*pred.shape[:2], len(ARMS), 3))
    for start, end, observed, _, level, origin in sessions(raw, grid):
        t = np.arange(len(level))
        for stock in range(50):
            x, age, bins = panel(level, origin, t, stock, pred[start:end])
            for group in range(AGES):
                ok = (bins == group) & (age > 0) & observed[:, stock]
                result = x[ok] @ bank[stock, group]
                first, rest = np.split(result, 2, axis=1)
                out[start + t[ok], stock] = np.stack((first, rest, first + rest), axis=1)
    return out - out.mean(1, keepdims=True)


def predict(raw, grid, p, params):
    return book.path.corrected(p, design(raw, grid, params["code"], p, params["bank"]), params)


if __name__ == "__main__":
    collect()
    onset.rolling(OUT, ARMS, train, design, REFERENCE)
