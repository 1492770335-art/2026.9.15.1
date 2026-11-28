"""Use peer pressure and the activity of each endpoint to predict later individual book states."""
import numpy as np

import experiment_dense_distal_activity_decomposition_v1 as distal

activity, rel, REFERENCE = distal.activity, distal.rel, distal.REFERENCE
OUT = REFERENCE.parent / "dense_peer_activity_bridge_v1"
ARMS = ("peer_pressure", "sender_activity", "receiver_activity")


def load(day, key):
    return np.load(rel.ref.base.CACHE / f"day_{day:02d}_{key}.npy", mmap_mode="r")


def graph():
    correlations = []
    for day in range(4):
        price = load(day, "x")[:, :, 0].astype(float) - .00136
        blocks = []
        for start, stop in activity.source.sessions(load(day, "time")):
            sums = np.vstack((np.zeros(50), price[start:stop].cumsum(0)))
            returns = sums[20::5] - sums[:-20:5]
            blocks.append(returns - returns.mean(1, keepdims=True))
        correlations.append(np.corrcoef(np.vstack(blocks).T))
    left, right = np.mean(correlations[::2], 0), np.mean(correlations[1::2], 0)
    # Preserve only positive covariation present on both sets of dates.
    agreement = np.minimum(left, right).clip(0)
    np.fill_diagonal(agreement, 0)
    weights = agreement ** 2
    return weights / weights.sum(1, keepdims=True).clip(1e-8)


def state(raw, grid, code, peers):
    own = activity.state(raw, grid, code).astype(np.float32)
    width = own.shape[-1] // 3
    values = np.empty((*own.shape[:2], 6 * width), dtype=np.float32)
    for start, stop in activity.source.sessions(grid):
        moved = (np.abs(raw[start:stop, :, 0].astype(float) - .00136) > .0001).astype(float)
        fast = activity.source.smooth(moved, 24)
        slow = activity.source.smooth(moved, 240)
        active = np.tanh(3 * (fast - .5 * (slow + fast.mean(1, keepdims=True))))
        for t in range(start, stop, 128):
            sl = slice(t, min(t + 128, stop))
            out = values[sl]
            out[:, :, :3*width] = own[sl]
            pressure = peers @ own[sl, :, :width]
            sender = peers @ own[sl, :, width:2*width]
            receiver = pressure * active[t-start:sl.stop-start, :, None]
            for j, signal in enumerate((pressure, sender, receiver), 3):
                out[:, :, j*width:(j+1)*width] = signal - signal.mean(1, keepdims=True)
    return values


def collect():
    OUT.mkdir(exist_ok=True)
    peers = graph()
    np.save(OUT / "peers.npy", peers)
    code = activity.source.codec()
    for day in range(16):
        grid = load(day, "time").astype(int)
        x = state(load(day, "x"), grid, code, peers)
        n = x.shape[-1]
        local = n // 2
        gram = np.zeros((n, n))
        future = np.zeros((n, local))
        payoff = np.zeros(local)
        count = 0
        y = load(day, "y")[:, :, 1].astype(float)
        y -= y.mean(1, keepdims=True)
        for start, stop in activity.source.sessions(grid):
            idx = np.arange(start, stop - 60, 4)
            now = x[idx].reshape(-1, n).astype(float)
            later = x[idx+20, :, :local] + x[idx+40, :, :local]
            gram += now.T @ now
            future += now.T @ later.reshape(-1, local)
            payoff += now[:, :local].T @ y[idx].ravel()
            count += len(now)
        np.savez(OUT / f"day_{day:02d}.npz", gram=gram/count, later=future/count, payoff=payoff/count)


def train(end):
    days = []
    for day in range(end):
        with np.load(OUT / f"day_{day:02d}.npz") as f:
            days.append(tuple(f[k] for k in ("gram", "later", "payoff")))
    grams, later, payoffs = map(np.stack, zip(*days))
    g = grams.mean(0)
    w = len(g) // 6
    local = 3*w
    regularized = g + .20*np.diag(g.diagonal().clip(.00005))
    cross = []
    for side in range(2):
        value = np.linalg.solve(regularized[:local, :local], payoffs[1-side::2].mean(0))
        reward = later[side::2].mean(0) @ value

        def bridge(ids):
            coeff = np.zeros(len(g))
            coeff[ids] = np.linalg.solve(regularized[np.ix_(ids, ids)], reward[ids])
            return coeff

        own = bridge(np.arange(local))
        base = bridge(np.arange(4*w))
        sender = bridge(np.arange(5*w))
        receiver = bridge(np.r_[0:4*w, 5*w:6*w])
        cross.append(np.column_stack((base-own, sender-base, receiver-base)))
    bank = np.mean(cross, 0)
    variance = np.einsum("fi,fg,gi->i", bank, g, bank)
    return bank / np.sqrt(variance.clip(1e-10))


def views(raw, grid, code, bank):
    values = state(raw, grid, code, np.load(OUT / "peers.npy"))
    return values @ bank


if __name__ == "__main__":
    collect()
    rel.main(OUT, ARMS, train, views, rolling_reference=REFERENCE)
