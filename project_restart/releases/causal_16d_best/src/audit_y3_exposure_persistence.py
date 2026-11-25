"""Walk-forward test of how strongly to carry recent pair exposures into the next market block."""
from functools import lru_cache
import json

import numpy as np
import torch

import experiment_y3_realized_recipient_exposures_v1 as model


@lru_cache(maxsize=16)
def returns(day):
    raw = model.trial.clock.book.rawday(day)
    return model.observations(raw, model.model.calibration())[0]


def samples(day, state):
    r = returns(day)
    # Evaluate non-overlapping 320-frame target blocks; omit blocks straddling the session boundary.
    for end in range(120, len(r) - 39, 40):
        if end < 300 < end + 40:
            continue
        recent = r[end - 120:end]
        centered = recent - recent.mean(0)
        cov = (40 * state['prior'][0] + centered.T @ centered) / 160
        delta = model.loading(cov, cov, state['basis']) - state['coef'][0]
        later = r[end:end + 40]
        actual = (later - later.mean(1, keepdims=True)) * (50 / 49)
        factors = np.einsum('ts,rsk->trk', later, state['basis'], optimize=True)
        fixed = np.einsum('trk,rk->tr', factors, state['coef'][0])
        adjustment = np.einsum('trk,rk->tr', factors, delta)
        error = actual - fixed
        yield np.stack(((error * adjustment).sum(0), (adjustment ** 2).sum(0),
                        (error ** 2).sum(0), (actual ** 2).sum(0)))


def run():
    records = []
    for cutoff in (8, 10, 12, 14):
        state = model.params(cutoff)
        prior = np.sum([np.sum(list(samples(day, state)), 0) for day in range(cutoff)], 0)
        pooled = prior[0].sum() / prior[1].sum()
        # Partial pooling across stocks prevents low-variance stocks from getting extreme weights.
        strength = np.median(prior[1])
        by_stock = ((prior[0] + pooled * strength) / (prior[1] + strength)).clip(-1, 1)
        for d in (cutoff, cutoff + 1):
            cross, energy, base_error, target_energy = np.sum(list(samples(d, state)), 0)
            def score(weight):
                return float(1 - np.sum(base_error - 2*weight*cross + weight**2*energy) / target_energy.sum())
            records.append(dict(day=d+1, training_days=cutoff, fitted_gain=float(pooled),
                                fixed=score(0), full_update=score(1), pooled_weight=score(pooled),
                                stock_weight=score(by_stock)))
    averages = {key: float(np.mean([row[key] for row in records]))
                for key in ('fixed', 'full_update', 'pooled_weight', 'stock_weight')}
    result = dict(task='Reconstruct next 320 frames of relative returns given other stocks realized in those frames',
                  days=records, mean_daily_r2=averages)
    (model.OUT / 'exposure_persistence.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
