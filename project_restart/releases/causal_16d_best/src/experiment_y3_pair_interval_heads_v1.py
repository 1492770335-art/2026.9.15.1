"""Train distinct 21–40 and 41–60 return heads on the retained ordered paired-feature representation."""
import sys

import numpy as np
import torch

import experiment_y3_paired_level_strength_v1 as level
import y3_filter_validation as trial

OUT = trial.clock.ART / 'y3_pair_interval_heads_v1'
ARMS = ['retained full tail', 'separately fitted intervals', 'extra steps 21–40',
        'extra steps 41–60', 'extra combined interval estimate']
state = level.state


def observe(z, y, d):
    grams, base = level.source.observe(z, y, d)
    ratio = trial.curve.settled.historical_geometry()[1]
    scale = ratio[2, 1] / ratio[1, 1]
    ticks = np.arange(d % 5, len(y) - 60, 5)
    moments = np.zeros((z.shape[2], 2))
    for ids in np.array_split(ticks, (len(ticks) + 95) // 96):
        x = z[ids].reshape(-1, z.shape[2]).astype('float32')
        labels = np.stack([trial.clock.center(y[ids + shift, :, 1]) * scale for shift in (20, 40)], axis=2)
        moments += x.T @ labels.clip(-3, 3).reshape(-1, 2)
    return grams, base, moments / (len(ticks) * 50)


def fit(hist):
    g = np.mean([r[0] for r in hist], axis=0)
    old = level.source.fit([(g, b) for g, b, _ in hist])[0]['w'][:, 2]
    separate = level.source.fit([(g, h) for g, _, h in hist])[0]['w'][:, 2:]
    overlap = (old @ g @ separate) / max(1e-12, old @ g @ old)
    return dict(w=np.column_stack((old, separate)).astype('float32'), overlap=overlap.astype('float32')), dict(overlap=overlap.tolist())


def changes(z, model):
    x = np.concatenate([r.astype('float32') @ model['w'] for r in np.array_split(z, (len(z) + 127) // 128)])
    old = .32 * x[:, :, 0]
    parts = x[:, :, 1:]
    unique = parts - x[:, :, :1] * model['overlap']
    return [old, .32 * parts.sum(2), old + .30 * unique[:, :, 0],
            old + .30 * unique[:, :, 1], old + .30 * unique.sum(2)]


def select(folds, daily):
    f, d = np.asarray(folds), np.asarray(daily)
    pick = trial.scoring.select(f[:, :, :, 1:] - f[:, :, :, :1], d[:, :, 1:] - d[:, :, :1])
    return None if pick is None else pick + 1


def correction():
    return trial.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


def load():
    return trial.load(sys.modules[__name__])


if __name__ == '__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
