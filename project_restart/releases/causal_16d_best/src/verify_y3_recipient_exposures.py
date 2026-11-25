"""Recompute the selected leave-one-stock-out update with per-stock scalar regressions."""
import json

import numpy as np
import torch

import experiment_y3_realized_recipient_exposures_v1 as model


def run():
    with np.load(model.OUT / 'model.npz') as archive:
        state = dict(archive)
    assert int(state['pick']) == 7
    weights = state['w'][:, 7]
    columns = [0, 1, 2, 17, 18]
    assert np.all(np.delete(weights, columns) == 0)
    basis = state['basis']
    np.testing.assert_allclose(basis.sum(1), 0, atol=1e-12)
    assert np.max(abs(basis[np.arange(50), np.arange(50)])) == 0
    raw = model.trial.clock.book.rawday(15)
    reference = model.history.references(15, raw)[0]
    forecast = model.correction()
    full = forecast(raw, reference)
    ticks = np.unique(np.r_[0, 1, 7, 8, 79, 80, 81, 959, 960, 961, np.arange(41, len(raw), 53)])
    x, _ = model.model.inputs(raw, ticks, state)
    k = float(state['k'])
    prices = np.log1p(k * (raw[:, :, 0].astype(float) - float(np.float16(.00136)))) / k
    direct = []
    for index, tick in enumerate(ticks):
        right = (tick // 80) * 80
        left = max(0, right - 960)
        cov = state['prior'][0]
        if right > left:
            obs = prices[left:right].reshape(-1, 8, 50).sum(1) / (np.sqrt(8) * state['sc'][50:])
            centered = obs - obs.mean(0)
            cov = (40 * cov + centered.T @ centered) / (40 + len(obs))
        shared, local = [np.einsum('i,ihj->hj', x[index], matrix) for matrix in state['weights']]
        adjusted = (local + shared).T.copy()
        for j, portfolios in enumerate(basis):
            gram = portfolios.T @ cov @ portfolios
            target = cov[:, j] - np.delete(cov, j, axis=1).mean(1)
            penalty = .3 * np.trace(gram) / len(gram) + .025
            coef = np.linalg.solve(gram + penalty * np.eye(len(gram)), portfolios.T @ target)
            adjusted[j] += (shared @ portfolios) @ (coef - state['coef'][0, j])
        features = np.column_stack((reference[tick], adjusted * state['sc'][50:, None]))
        direct.append((features - features.mean(0)) @ weights[columns])
    direct_error = float(np.max(np.abs(full[ticks] - direct)))
    prefixes = (1, 7, 8, 9, 79, 80, 81, 159, 160, 161, 959, 960, 961, 2401, len(raw)-1)
    prefix_error = max(float(np.max(abs(forecast(raw[:n], reference[:n]) - full[:n]))) for n in prefixes)
    assert max(direct_error, prefix_error) < 4e-6
    report = dict(day=16, stock_ticks=len(ticks)*50, direct_error=direct_error, prefixes=len(prefixes),
                  prefix_error=prefix_error)
    (model.OUT / 'inference_verification.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
