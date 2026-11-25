"""Fit a cross-sectional Y3 correction with price history that decays faster after reversals."""
import argparse
import json

import numpy as np
from scipy.signal import lfilter
import torch

import experiment_y3_directed_unexpected_prices_v1 as price
import y3_directed_walkforward_reference as reference
from y3_portfolio_walkforward import replay

OUT = price.OUT.parent/'y3_reversal_conditioned_memory_fixed_v1'
H = np.array([6, 24, 100, 240], float)
ARMS = ('incumbent', 'fixed decay', 'shock energy decay', 'reversal decay', 'confirmation decay',
        'unrelated stock controls decay', 'reversal decay with variance normalization',
        'shock energy decay with variance normalization')


def summarize(z, sd):
    n = z.shape[1]
    unit = z/sd
    v = np.stack((z, unit, np.tanh(unit), np.tanh(unit)*np.minimum(abs(unit), 3), unit*unit), axis=2)
    common = v.mean(1, keepdims=True)
    leave = (n*common-v)/(n-1)

    def transform(a):
        vol = np.sqrt(np.maximum(.04, a[:, :, 4]-a[:, :, 1]**2+.04))
        return np.concatenate((a[:, :, :4], (a[:, :, 1]/(.5+vol))[:, :, None]), axis=2)
    full = transform(common)
    change = (n-1)*(full-transform(leave))
    return full[:, 0].astype('float32'), (change-change.mean(1, keepdims=True)).astype('float32')


def row(x, p, y, d):
    t = np.concatenate([np.arange(a+d % 7, b-60, 7) for a, b in price.segments(len(p))])
    ref = p[t]
    pred = ref[:, :, 2].astype(float)
    actual = y[t, :, 2].astype(float)
    scale = np.clip(np.sum((actual-actual.mean())*(pred-pred.mean()))/np.sum((pred-pred.mean())**2), .4, 2.5)
    err = actual/scale-pred
    err -= err.mean(1, keepdims=True)
    control = ref-ref.mean(1, keepdims=True)
    stats = []
    for v in x:
        z = np.concatenate((v[t], control, err[:, :, None]), 2).reshape(-1, 24).astype(float)
        z -= z.mean(0)
        stats.append(z.T@z/len(z))
    return np.stack(stats)


def weights(hist):
    series = np.stack(hist)
    ws = []
    for i, g in enumerate(series.mean(0)):
        ctrl = g[20:23, 20:23]
        proj = np.linalg.solve(ctrl+.12*np.diag(ctrl.diagonal().clip(.00001)), g[20:23, :20])
        b = np.eye(24)[:, :20]
        b[20:23] = -proj
        cov = b.T@g@b
        scores = series[:, i, :, -1]@b
        grad = scores.mean(0)
        grad *= abs(grad)/(abs(grad)+scores.std(0)/np.sqrt(len(scores))+.00001)
        ws.append((b@np.linalg.solve(cov+1.4*np.diag(cov.diagonal().clip(.00001)), grad))[:-1])
    return np.array(ws, dtype=np.float32)


def signals(views, p, w):
    q = p-p.mean(1, keepdims=True)
    return np.stack([v@b[:20]+q@b[20:] for v, b in zip(views, w)], axis=2)


def views(chunks, model):
    w, _, sd = model
    h = H
    base = (1-1/h)
    normalizer = 1/np.sqrt(h)
    buffers = [[] for _ in ARMS[1:]]
    for move, lag in chunks:
        innovations = move-lag@w
        # Use a single trailing price path to define reversal across all comparison arms.
        ref = np.stack([lfilter([n], [1, -b], innovations, axis=0) for n, b in zip(normalizer, base)], axis=2)
        past = np.concatenate((np.zeros_like(ref[:1]), ref[:-1]))
        signed = innovations[:, :, None]*past/(sd[None, :, None]**2)
        tilt = np.tanh(signed)
        energy = np.minimum(12, (innovations/sd)**2)[:, :, None]
        shocks = np.broadcast_to(energy, tilt.shape)
        rates = np.stack((shocks, energy*(1-tilt), energy*(1+tilt), np.roll(energy*(1-tilt), 17, axis=1)), axis=1)
        decay = base**(.30+.70*rates)
        state = np.zeros((4, 50, len(h)))
        variance = np.full_like(state, 1/(h*(1-base*base)))
        output = np.empty((len(move), 6, 50, len(h)), np.float32)
        for i, u in enumerate(innovations):
            b = decay[i]
            state = b*state+u[:, None]*normalizer
            variance = b*b*variance+1/h
            output[i, :4] = state
            output[i, 4:] = state[[1, 0]]/np.sqrt(variance[[1, 0]]+.05)
        for buf, v in zip(buffers, [ref, *output.transpose(1, 0, 2, 3)]):
            buf.append(np.concatenate([summarize(v[:, :, i], sd)[1] for i in range(len(h))], axis=2))
    return [np.concatenate(parts) for parts in buffers]


def train():
    assert (reference.OUT/'results.json').is_file()
    k = price.previous_stage.early_slopes.geometry()[0]
    gh = [price.observe(price.tape(np.load(price.CACHE/f'day_{d:02d}_x.npy', mmap_mode='r'), k)) for d in range(4)]
    hist, cal, days = [], [], []
    for d in range(4, 16):
        raw, _, y = price.read_day(d)
        q = np.load(reference.OUT/f'day_{d}.npy', mmap_mode='r')
        if d % 2 == 0:
            model = price.innovation_model(gh)
        chunks = price.tape(raw, k)
        x = views(chunks, model)
        if d >= 6:
            if d % 2 == 0:
                w = weights(hist)
            z = signals(x, q, w)
            if d >= 8:
                if d % 2 == 0:
                    size = price.sizing(cal)
                days.append(price.rscore(q, y, z*size))
                if d == 8:
                    for n in (991, 2537):
                        short = views(price.tape(raw[:n], k), model)
                        np.testing.assert_allclose(z[:n], signals(short, q[:n], w), atol=4.e-6, rtol=0)
            cal.append(price.rscore(q, y, z))
        hist.append(row(x, q, y, d))
        gh.append(price.observe(chunks))
        print('day', d+1, flush=True)
    halves = np.stack([price.diff(sum(v)) for v in (days[:4], days[4:])])
    daily = np.stack([price.diff(v) for v in days])
    j = 3
    controls = [0, 1, 2, 4, 5]
    keep = bool((halves[:, j, None] > halves[:, controls]).all() and (daily[:, j] > 0).sum() >= 5)
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS, choice=j, selected=j if keep else 0,
                                                    halves=halves.tolist(), daily=daily.tolist(), controls=controls,
                                                    selected_using_both_training_folds=True), indent=2))
    if keep:
        w, pr, sd = price.innovation_model(gh)
        np.savez(OUT/'model.npz', k=k, w=w, pr=pr, sd=sd, read=weights(hist), size=price.sizing(cal)[j-1], pick=j)
    print('choice', j, 'keep', keep, 'Y3 gains', halves, flush=True)


def correction():
    previous = price.correction()
    with np.load(OUT/'model.npz') as st:
        f = dict(st)

    def update(raw, p):
        v = previous(raw, p)
        q = p.copy()
        q[:, :, 2] += v
        return v+adjust(raw, q, f)
    return update


def adjust(raw, p, f):
    """Evaluate just the retained filter. Historical training also evaluates the controls."""
    h = H
    base = 1-1/h
    norm = 1/np.sqrt(h)
    parts = []
    for move, lag in price.tape(raw, float(f['k'])):
        u = move-lag@f['w']
        ref = np.stack([lfilter([n], [1, -b], u, axis=0) for n, b in zip(norm, base)], axis=2)
        past = np.concatenate((np.zeros_like(ref[:1]), ref[:-1]))
        tilt = np.tanh(u[:, :, None]*past/(f['sd'][None, :, None]**2))
        energy = np.minimum(12, (u/f['sd'])**2)[:, :, None]
        decay = base**(.30+.70*(energy*(1-tilt)))
        state = np.zeros((50, len(h)))
        saved = np.empty_like(ref, dtype=np.float32)
        for t, shock in enumerate(u):
            state = decay[t]*state+shock[:, None]*norm
            saved[t] = state
        parts.append(np.concatenate([summarize(saved[:, :, j], f['sd'])[1] for j in range(len(h))], axis=2))
    w = f['read'][int(f['pick'])-1]
    z = np.concatenate(parts)@w[:20]+(p-p.mean(1, keepdims=True))@w[20:]
    return z*float(f['size'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'verify'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'train':
        train()
    else:
        replay(OUT, correction)
