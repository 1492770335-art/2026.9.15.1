"""Use past book observations to predict future book information unexplained by the current observation."""
import numpy as np
import experiment_dense_intraday_payoff_clock_v1 as runner

ref = runner.ref
OUT = ref.base.ROOT / "artifacts/dense_conditional_lag_memory_v1"
ARMS = ("memory20", "memory40", "multiscale_memory")


def parameters(history):
    stock, market, stock_pay, market_pay, stock_var, market_var = [np.stack(x) for x in zip(*history)]
    sd = np.sqrt(market[:, 0].mean(0).diagonal()).clip(.006)
    ss = np.sqrt(stock[:, 0].mean(0).diagonal()).clip(.03)
    cov = .5 * (market / np.outer(sd, sd) + stock / np.outer(ss, ss))
    pay = .5 * (market_pay[:, :, 2] / sd + stock_pay[:, :, 2] / ss * np.sqrt(market_var[:, 2].mean() / stock_var[:, 2].mean()))
    coeff = []
    for half in (0, 1):
        covs = cov[half::2].mean(0)
        c = covs[0]
        identity = np.eye(len(c))
        response = np.linalg.solve(cov[1-half::2, 0].mean(0) + .22 * identity, pay[1-half::2].mean(0))
        partial = []
        for i in (3, 4):
            # K is E[past.T @ now]. Remove the best linear prediction from both the past and the future.
            k, twice = covs[i], covs[i+1]
            predict_past = np.linalg.solve(c + .06 * identity, k.T)
            predict_future = np.linalg.solve(c + .06 * identity, k)
            residual_cross = twice - k @ predict_future - predict_past.T @ k + predict_past.T @ c @ predict_future
            residual_self = c - k @ predict_past - predict_past.T @ k.T + predict_past.T @ c @ predict_past
            gain = np.linalg.solve(residual_self + .25 * identity, residual_cross @ response)
            partial.append(np.stack((-predict_past @ gain / sd, gain / sd)))
        coeff.append(partial)
    return dict(coeff=np.mean(coeff, axis=0))


def views(m, grid, model):
    output = np.zeros((len(m), 3))
    for a, b in ref.market.parent.sessions(grid):
        for j, lag in enumerate((20, 40)):
            idx = np.arange(a + lag, b)
            now, old = model["coeff"][j]
            output[idx, j] = m[idx] @ now + m[idx-lag] @ old
    output[:, 2] = output[:, 0] + output[:, 1]
    return output


if __name__ == "__main__":
    runner.run(OUT, ARMS, parameters, views, keep_oof=True)
