"""Fit the cross-stock response to predictable price changes on chronological training folds."""
import argparse
import json
import numpy as np
from scipy.linalg import eigh, solve
from scipy.signal import lfilter
import torch

import experiment_y3_supported_morning_slopes_v1 as previous_stage
from experiment_causal_cross_horizon_ridge_v1 import ART, CACHE
from y3_portfolio_walkforward import replay

segments = previous_stage.parent.base.segments
OUT = ART / 'y3_directed_unexpected_prices_v1'
ARMS = ('incumbent', 'all price changes', 'unanticipated from past', 'unanticipated from past and other stocks',
        'expected from peer prices and past', 'wrong stock-specific regressions', 'unexpected large changes')


def read_day(d):
    raw = np.load(CACHE/f'day_{d:02d}_x.npy', mmap_mode='r')
    refs = np.load(ART/'y3_settled_walkforward_panels_v1'/f'day_{d+1}.npy', mmap_mode='r')
    p = refs[-1] if d >= 8 else np.mean(refs[:2], axis=0)
    y = np.load(CACHE/f'day_{d:02d}_y.npy', mmap_mode='r')
    return raw, p, y


def design(raw, p, k):
    """Each tick uses only trailing returns. Across-stock centering removes the shared market leg."""
    rows = []
    for a, b in segments(len(raw)):
        move = np.log1p(k*(raw[a:b, :, 0].astype(float)-.00136))/k
        states = [lfilter([1/np.sqrt(span)], [1., -(1-1/span)], move, axis=0) for span in (240, 8, 32)]
        x = np.stack((*states, *p[a:b].transpose(2, 0, 1)), axis=2)
        rows.append((x-x.mean(1, keepdims=True)).astype(np.float32))
    return np.concatenate(rows)


def moments(x, p, y, d):
    times = np.concatenate([np.arange(a+32+d % 5, b-60, 5) for a, b in segments(len(x))])
    yt = y[times, :, 2]
    pt = p[times, :, 2]
    yt = yt-yt.mean(1, keepdims=True)
    pt = pt-pt.mean(1, keepdims=True)
    beta = np.clip(np.sum(pt*yt)/np.sum(pt*pt), .35, 2.5)
    e = (yt/beta-pt).clip(-4, 4)
    z = np.concatenate((x[times], e[:, :, None]), 2).transpose(0, 2, 1).reshape(len(times), -1).astype(float)
    return z.T@z/len(z)


def signals(x, w):
    flat = x.transpose(0, 2, 1).reshape(len(x), -1)
    pred = np.stack([flat@m for m in w], axis=2)
    return pred-pred.mean(1, keepdims=True)


def rscore(p, y, z):
    rows = np.concatenate((np.ones_like(y[:, :, :1]), p[:, :, 2:],
                          y[:, :, 2:], z), axis=2).reshape(-1, 3+z.shape[2]).astype(float)
    return rows.T@rows


def diff(g):
    c = g[1:, 1:]/g[0, 0]-np.outer(g[0, 1:], g[0, 1:])/g[0, 0]**2
    eye = np.eye(len(c))
    outs = np.vstack((eye[0], eye[0]+eye[2:]))
    s = outs@c[:, 1]/np.sqrt(np.einsum('ij,jk,ik->i', outs, c, outs)*c[1, 1])
    return s-s[0]


def directed_response(history):
    n = 50
    g = np.mean(history, 0)
    stock = np.arange(n)
    bias = np.zeros((len(g), n))
    bias[stock, stock] = 1
    controls = np.arange(1, 6)[:, None]*n+stock
    for i in stock:
        idx = controls[:, i]
        h = g[np.ix_(idx, idx)]
        reg = .16*np.diag(h.diagonal().clip(.0001))
        bias[idx, i] = -np.linalg.solve(h+reg, g[idx, i])
    # Normalize the inputs and the recipients in the same regularized portfolio covariance.
    h = bias.T@g@bias
    scale = h.diagonal().clip(.0001)**.5
    ratio = h/np.outer(scale, scale)
    values, axes = eigh(ratio)
    vals = np.sqrt(values.clip(0)+.60)
    root = (axes*vals)@axes.T
    inv = (axes/vals)@axes.T
    response = bias.T@g[:, -n:]/np.outer(scale, scale)
    op = inv@response@inv
    directed = (inv@op@root)*scale[None, :]/scale[:, None]
    return (.30*bias[:-n]@directed).astype(np.float32)


def calibrate(g):
    c = g[1:, 1:]/g[0, 0]-np.outer(g[0, 1:], g[0, 1:])/g[0, 0]**2
    beta = np.clip(c[0, 1]/c[0, 0], .30, 3)
    gradient = c[2:, 1]/beta-c[2:, 0]
    variance = c.diagonal()[2:]-c[2:, 0]**2/c[0, 0]
    return gradient, variance


def sizing(history):
    grads, variances = [np.array(v) for v in zip(*(calibrate(g) for g in history))]
    avg = grads.mean(0)
    rel = abs(avg)/(abs(avg)+grads.std(0)/np.sqrt(len(grads))+.000001)
    return (.55*avg/variances.mean(0).clip(.00001)*rel).clip(-1.25, 1.25)


def tape(raw, k):
    chunks = []
    for a, b in segments(len(raw)):
        move = np.log1p(k*(raw[a:b, :, 0].astype(float)-.00136))/k
        past = np.vstack((np.zeros_like(move[:1]), move[:-1])).clip(-15, 15)
        lags = np.concatenate([past, lfilter([.25], [1, -.75], past, axis=0),
                               lfilter([.05], [1, -.95], past, axis=0)], axis=1)
        chunks.append((move, lags))
    return chunks


def observe(chunks):
    blocks = [np.concatenate((lag, r.clip(-15, 15)), axis=1) for r, lag in chunks]
    n = sum(len(v) for v in blocks)
    return sum(z.T@z for z in blocks)/n


def innovation_model(grams):
    g = np.mean(grams, 0)
    n = 150
    xx = g[:n, :n]
    w = solve(xx+.4*np.diag(xx.diagonal().clip(.001)), g[:n, n:], assume_a='pos')
    mapping = np.vstack((-w, np.eye(50)))
    cov = mapping.T@g@mapping
    reg = cov+.55*np.diag(cov.diagonal().clip(.001))
    precision = solve(reg, np.eye(50), assume_a='pos')
    return w, precision/precision.diagonal(), cov.diagonal().clip(.001)**.5


def designs(raw, p, k, model):
    base = design(raw, p, k)
    chunks = tape(raw, k)
    w, precision, sd = model
    order = np.roll(np.arange(50), 17)
    wrong_order = np.r_[order, order+50, order+100]
    collected = [[] for _ in range(5)]
    for move, lag in chunks:
        surprise = move-lag@w
        own = surprise@precision
        wrong = (move-lag@w[np.ix_(wrong_order, order)])@precision[np.ix_(order, order)]
        large = own*(1-np.exp(-.5*(own/sd)**2))
        for parts, v in zip(collected, (surprise, own, move-own, wrong, large)):
            state = lfilter([1/np.sqrt(240.)], [1, -(1-1/240.)], v, axis=0)
            parts.append((state-state.mean(1, keepdims=True)).astype(np.float32))
    return base, [np.concatenate(rows) for rows in collected], chunks


def generate(base, states, banks):
    results = []
    for i, w in enumerate(banks):
        x = base.copy() if i else base
        if i:
            x[:, :, 0] = states[i-1]
        results.append(signals(x, w[None])[:, :, 0])
    return np.stack(results, axis=2)


def train():
    k = previous_stage.early_slopes.geometry()[0]
    rawstats = [observe(tape(np.load(CACHE/f'day_{d:02d}_x.npy', mmap_mode='r'), k)) for d in range(4)]
    hist = [[] for _ in ARMS[1:]]
    measured, days = [], []
    for d in range(4, 16):
        raw, p, y = read_day(d)
        if d % 2 == 0:
            model = innovation_model(rawstats)
        base, states, chunks = designs(raw, p, k, model)
        if d >= 6:
            if d % 2 == 0:
                banks = [directed_response(g) for g in hist]
            z = generate(base, states, banks)
            if d >= 8:
                if d % 2 == 0:
                    size = sizing(measured)
                if d == 8:
                    for n in (161, 2489):
                        bx, st, _ = designs(raw[:n], p[:n], k, model)
                        np.testing.assert_allclose(z[:n], generate(bx, st, banks), atol=3e-6, rtol=0)
                days.append(rscore(p, y, z*size))
            measured.append(rscore(p, y, z))
        for i, h in enumerate(hist):
            x = base.copy() if i else base
            if i:
                x[:, :, 0] = states[i-1]
            h.append(moments(x, p, y, d))
        rawstats.append(observe(chunks))
        print('date', d+1, flush=True)
    half = np.stack([diff(sum(g)) for g in (days[:4], days[4:])])
    daily = np.stack([diff(g) for g in days])
    ids = (2, 3, 4, 6)
    pick = ids[np.argmax(half[0, list(ids)])]
    keep = (half[:, pick] > half[:, 1]).all() and (half[:, pick] >
                                                   half[:, 5]).all() and (daily[:, pick] > daily[:, 1]).sum() >= 5
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS, choice=int(pick), selected=int(pick) if keep else 0,
                                                    half_delta=half.tolist(), daily_delta=daily.tolist()), indent=2))
    if keep:
        w, prec, sd = innovation_model(rawstats)
        np.savez(OUT/'model.npz', w=w, prec=prec, sd=sd, k=k, pick=pick,
                 direction=directed_response(hist[pick-1])*sizing(measured)[pick-1])
    print('choice', pick, 'retain', keep, 'Y3 changes', half.tolist(), flush=True)


def adjust(raw, q, model, k, pick, direction):
    # The deployed head only needs its selected price stream.
    x = design(raw, q, k)
    history_weights, peer_precision, deviation = model
    for (start, stop), (move, lag) in zip(segments(len(raw)), tape(raw, k)):
        prior_error = move - lag @ history_weights
        stock_error = prior_error @ peer_precision
        if pick == 4:
            filtered = move - stock_error
        elif pick == 3:
            filtered = stock_error
        elif pick == 2:
            filtered = prior_error
        elif pick == 6:
            filtered = stock_error * (1 - np.exp(-.5 * (stock_error/deviation)**2))
        else:
            raise ValueError(f"Unknown price stream {pick}")
        slow = lfilter([1/np.sqrt(240.)], [1, -(1-1/240.)], filtered, axis=0)
        x[start:stop, :, 0] = slow - slow.mean(1, keepdims=True)
    return signals(x, direction[None])[:, :, 0]


def correction():
    previous = previous_stage.correction()
    with np.load(OUT/'model.npz') as f:
        model = (f['w'], f['prec'], f['sd'])
        k, pick = float(f['k']), int(f['pick'])
        direction = f['direction']

    def update(raw, p):
        delta = previous(raw, p)
        q = p.copy()
        q[:, :, 2] += delta
        return delta+adjust(raw, q, model, k, pick, direction)
    return update


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'verify'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'train':
        train()
    else:
        replay(OUT, correction)
