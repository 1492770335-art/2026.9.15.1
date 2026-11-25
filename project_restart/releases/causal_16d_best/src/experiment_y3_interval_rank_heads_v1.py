"""Fit middle and final Y3 intervals with a within-tick ordinal target across the 50 stocks."""
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
import torch

import experiment_y3_pair_interval_heads_v1 as split
from experiment_y3_pair_subinterval_curve_v1 import changes, select
import y3_filter_validation as trial

OUT = trial.clock.ART / 'y3_interval_rank_heads_v1'
ARMS = ['retained specialist', 'middle ordinal signal', 'final ordinal signal', 'both ordinal signals']
state = split.state


def observe(z, y, d):
    earlier = split.observe(z, y, d)
    coeff = trial.curve.settled.historical_geometry()[1]
    ticks = np.arange(d % 5, len(y) - 60, 5)
    moment = np.zeros((z.shape[2], 2))
    for ids in np.array_split(ticks, (len(ticks) + 95) // 96):
        payoffs = np.stack((y[ids + 20, :, 1], y[ids + 40, :, 1]), axis=2) * (coeff[2, 1] / coeff[1, 1])
        ranks = (rankdata(payoffs, axis=1) - .5) / 50
        # Give a large move its cross-sectional scale without letting its magnitude determine its individual label.
        amplitude = np.mean(abs(trial.clock.center(payoffs)), axis=1, keepdims=True).clip(.03, 1.2)
        target = (ndtri(ranks) * amplitude).astype('float32')
        x = z[ids].reshape(-1, z.shape[2]).astype('float32')
        moment += x.T @ target.reshape(-1, 2)
    return earlier, moment / (len(ticks) * 50)


def fit(history):
    records, moments = zip(*history)
    model, _ = split.fit(records)
    w = model['w']
    baseline = .32 * w[:, 0] + 2.4 * (w[:, 1] - model['overlap'][0] * w[:, 0])
    ranked = split.level.source.fit([(r[0], m) for r, m in zip(records, moments)])[0]['w'][:, 2:]
    g = np.mean([r[0] for r in records], axis=0)
    linear = w[:, :2]
    overlap = np.linalg.solve(linear.T @ g @ linear + 1e-10 * np.eye(2), linear.T @ g @ ranked)
    new = ranked - linear @ overlap
    return dict(w=np.column_stack((baseline, new)).astype('float32')), dict(days=len(history))


def correction():
    return trial.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


def load():
    return trial.load(sys.modules[__name__])


if __name__ == '__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
