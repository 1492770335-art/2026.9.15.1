"""Test temporal slopes inside the two twenty-tick future intervals of Y3."""
import sys

import numpy as np
import torch

import experiment_y3_finite_impact_curve_v1 as payouts
import experiment_y3_pair_interval_heads_v1 as split
import y3_filter_validation as trial

OUT = trial.clock.ART / 'y3_pair_subinterval_curve_v1'
ARMS = ['retained specialist', 'middle interval slope', 'last interval slope', 'both slopes']
state = split.state


def observe(z, y, d):
    baseline = split.observe(z, y, d)
    t = np.arange(d % 5, len(y) - 60, 5)
    moment = np.zeros((z.shape[2], 2))
    for idx in np.array_split(t, (len(t) + 95) // 96):
        r = payouts.blocks(y, idx)[:, :, 4:].reshape(len(idx), 50, 4, 2).sum(3)
        contrasts = trial.clock.center(r[:, :, ::2] - r[:, :, 1::2])
        x = z[idx].reshape(-1, z.shape[2]).astype('float32')
        moment += x.T @ contrasts.clip(-3, 3).reshape(-1, 2)
    return baseline, moment / (len(t) * 50)


def fit(history):
    records, draws = zip(*history)
    base, _ = split.fit(records)
    w = base['w']
    old = .32 * w[:, 0] + 2.4 * (w[:, 1] - base['overlap'][0] * w[:, 0])
    g = np.mean([r[0] for r in records], axis=0)
    earned = split.level.source.fit([(r[0], m) for r, m in zip(records, draws)])[0]['w'][:, 2:]
    span = w[:, :2]
    overlap = np.linalg.solve(span.T @ g @ span + 1e-10 * np.eye(2), span.T @ g @ earned)
    new = earned - span @ overlap
    return dict(w=np.column_stack((old, new)).astype('float32')), dict(days=len(history), overlap=overlap.tolist())


def changes(z, model):
    result = np.concatenate([v.astype('float32') @ model['w'] for v in np.array_split(z, (len(z) + 95) // 96)])
    old, middle, end = np.moveaxis(result, 2, 0)
    return [old, old + .6 * middle, old + .6 * end, old + .45 * (middle + end)]


def select(folds, daily):
    f, d = np.asarray(folds), np.asarray(daily)
    chosen = trial.scoring.select(f[:, :, :, 1:] - f[:, :, :, :1], d[:, :, 1:] - d[:, :, :1])
    return None if chosen is None else chosen + 1


def correction():
    return trial.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


def load():
    return trial.load(sys.modules[__name__])


if __name__ == '__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
