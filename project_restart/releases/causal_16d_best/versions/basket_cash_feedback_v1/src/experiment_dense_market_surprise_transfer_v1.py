"""Distill later five-tick common forecasts and their predictable errors into today's market state."""
import numpy as np

import experiment_dense_projected_future_surprises_v1 as stock

rel, ref = stock.rel, stock.ref
fine = rel.fine
OUT = ref.base.ROOT / "artifacts/dense_market_surprise_transfer_v1"
ARMS = ("market_teacher", "stock_error_prior", "shared_error_prior")


def context(market, grid):
    result = np.empty((len(market), market.shape[1] + 38))
    result[:, :market.shape[1]] = market
    for start, end in rel.fields.parent.sessions(grid):
        book = market[start:end, :38]
        result[start:end, market.shape[1]:] = book - rel.fields.parent.smooth(book, 32)
    return result


def collect():
    OUT.mkdir(exist_ok=True)
    for day in range(4, 16):
        with np.load(ref.clock.live.data.OUT / f"day_{day:02d}.npz") as f:
            m = f["state"]
        y = np.load(ref.base.CACHE / f"day_{day:02d}_y.npy", mmap_mode="r").mean(1, dtype=float)
        grid = np.load(ref.base.CACHE / f"day_{day:02d}_time.npy").astype(int)
        p = stock.future.forecast(day).mean(1, dtype=float)
        later = np.column_stack((p, m))
        now = np.column_stack((p, context(m, grid)))
        centered = p - p.mean(0)
        slope = ((centered * (y - y.mean(0))).sum(0) / (centered**2).sum(0)).clip(.2, 4.)
        unit = ref.base.UNITS[0] / slope[2]
        error = unit * (y[:, 0] - slope[0] * p[:, 0])
        gram = cross = transfer = covariance = reward = 0.
        count = 0
        for a, b in rel.fields.parent.sessions(grid):
            t = np.arange(a, b-60, 2)
            if not len(t):
                continue
            def demean(v):
                return v - v.mean(0)
            x = demean(now[t])
            offsets = t[:, None] + np.arange(0, 60, 5)
            maturities = later[offsets]
            future = demean(maturities.mean(1))
            values = demean(unit * slope[0] * p[offsets, 0].sum(1))
            issued = maturities.reshape(-1, later.shape[1])
            issued = demean(issued)
            gram += x.T @ x
            cross += x.T @ values
            transfer += x.T @ future
            covariance += issued.T @ issued / 12
            reward += issued.T @ demean(error[offsets].ravel()) / 12
            count += len(t)
        np.savez(OUT / f"day_{day:02d}.npz", gram=gram/count, cross=cross/count,
                 transfer=transfer/count, cov=covariance/count, reward=reward/count)


def train(history):
    records = []
    for day in range(4, len(history)):
        with (np.load(OUT / f"day_{day:02d}.npz") as m,
              np.load(stock.OUT / f"day_{day:02d}.npz") as s):
            records.append((m["gram"], m["cross"], m["transfer"], m["cov"], m["reward"],
                            s["bb"].mean(0), s["bs"].mean(0)/4))
    gram, cross, transfer, cov, reward, scov, sreward = map(np.stack, zip(*records))
    # Remove the three already issued predictions before estimating a new book direction.
    proj = np.linalg.solve(gram[:, :3, :3] + .015*np.eye(3)*gram[:, :3, :3].diagonal(axis1=1,axis2=2)[:, None], gram[:, :3, 3:])
    left = proj.transpose(0, 2, 1)
    base = (gram[:, 3:, 3:] - left@gram[:, :3, 3:] - gram[:, 3:, :3]@proj + left@gram[:, :3, :3]@proj).mean(0)
    cross = cross[:, 3:] - (left@cross[:, :3, None])[:, :, 0]
    transfer = transfer[:, 3:] - left@transfer[:, :3]
    units = np.sqrt(base.diagonal()).clip(.003)
    bank = []
    for i in (0, 1):
        src = slice(i, None, 2)
        dst = slice(1-i, None, 2)
        def estimate(g, rhs):
            sd = np.sqrt(g.diagonal()).clip(.003)
            return np.linalg.solve(g/np.outer(sd, sd) + .7*np.eye(len(sd)), rhs/sd), sd
        market_price, sd = estimate(cov[src].mean(0), reward[src].mean(0))
        stock_price, _ = estimate(scov[src].mean(0), sreward[src].mean(0))
        # Both price vectors now apply to unit-standard-deviation movements. Match their risk on training dates.
        market_risk = cov[src].mean(0) / np.outer(sd, sd)
        ratio = np.sqrt((market_price@market_risk@market_price) / max(stock_price@market_risk@stock_price, 1e-12))
        prior = stock_price * ratio
        prices = np.column_stack((market_price, prior, .5*(prior+market_price)))
        gradients = cross[dst].mean(0)[:, None] + 12*transfer[dst].mean(0)@(prices/sd[:, None])
        bank.append(np.linalg.solve(base/np.outer(units, units) + .85*np.eye(len(units)),
                                    gradients/units[:, None])/units[:, None])
    weights = np.mean(bank, 0)
    return dict(bank=weights/np.sqrt(np.einsum("ij,ik,kj->j", weights, gram.mean(0)[3:,3:], weights)).clip(1e-8))


def views(m, grid, params):
    return context(m, grid) @ params["bank"]


if __name__ == "__main__":
    collect()
    fine.memory.runner.run(OUT, ARMS, train, views, targets=(2,),
                           prior_head=(fine.memory, fine.news, fine.path, fine),
                           keep_oof=True, check_leave_one_out=True)
