"""Test whether a stock that lags its historical peers catches up after a shared price pulse."""
import json

import numpy as np

import experiment_intraday_pulse_response_v1 as pulse
import experiment_book_transition_rewards_v1 as walk
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

OUT = ART / 'peer_catchup_after_pulse_v1'
PARENT = pulse.post
WIDTH = 64


def geometry():
    moments = []
    for day in range(8):
        r = PARENT.before.book.rawday(day)[:, :, 0].astype(float) - .00136
        blocks = r[:len(r) // 24 * 24].reshape(-1, 24, 50).sum(1)
        moments.append(np.cov(blocks.T))
    cov = np.median(moments, axis=0)
    var = cov.diagonal().clip(.001)
    corr = cov / np.sqrt(np.outer(var, var))
    np.fill_diagonal(corr, -np.inf)
    ids = np.argsort(corr, axis=1)[:, -8:]
    peers = np.zeros_like(cov)
    np.put_along_axis(peers, ids, 1 / 8, axis=1)
    market = (np.ones_like(cov) - np.eye(len(cov))) / (len(cov) - 1)
    def regress(weights):
        beta = (cov * weights).sum(1) / ((weights @ cov) * weights).sum(1).clip(.01)
        return weights * beta.clip(0, 2.5)[:, None]
    return np.stack((regress(market), regress(peers)))


def features(raw, transfer, cut):
    events = list(pulse.detect(raw, cut))
    if len(raw) > PARENT.before.BOUNDARY:
        events.append(PARENT.before.BOUNDARY)
    events.sort()
    r = raw[:, :, 0].astype(float) - .00136
    peer = r @ transfer.T
    def integral(v):
        return np.vstack((np.zeros(50), np.cumsum(v, axis=0)))
    observed, expected = integral(r), integral(peer)
    tick = np.arange(len(r))[:, None]
    last = np.maximum.accumulate(np.where(np.abs(r) > .022, tick, -1), axis=0)
    ids, values = [], []
    for start in events:
        t = np.arange(start, min(len(r), start + WIDTH))
        end = t + 1
        age = (t - start)[:, None]
        vol = np.sqrt(.15 + (r[start-160:start] ** 2).mean(0))
        # Every basket excludes the stock itself; future peer moves are unavailable here.
        peer_move = (expected[end] - expected[start]) / vol
        own_move = (observed[end] - observed[start]) / vol
        gap = np.arcsinh((peer_move - own_move) / np.sqrt(age + 1))
        unpriced = np.arcsinh((expected[end] - expected[np.maximum(last[t]+1, start), np.arange(50)]) / vol)
        breadth = ((r[t] * peer[t]) > 0).mean(1, keepdims=True)
        quiet = (last[t] < start)
        pulse_size = np.tanh(peer[start] / vol)[None, :]
        flow = np.tanh(np.arcsinh(raw[t, :, 41]) - np.arcsinh(raw[t, :, 42]))
        f = np.stack((gap, unpriced, unpriced * quiet, gap * quiet, gap * breadth,
                      gap * flow * np.tanh(peer_move), gap * np.exp(-age / 12),
                      unpriced * np.exp(-age / 12), np.broadcast_to(pulse_size, gap.shape) * quiet), axis=2)
        ids.append(t)
        values.append(f - f.mean(1, keepdims=True))
    if not ids:
        return np.empty(0, int), np.empty((0, 50, 9))
    return np.concatenate(ids), np.concatenate(values)


def reference(day):
    src = PARENT.before.parent_model
    p = np.load(src.block.segment.best.refit.OUT / f'reference_{day}.npy').copy()
    p[:, :, 1] = np.load(src.linear.earlier.OUT / f'y2_day_{day}.npy')
    return p


def moments(f, p, y):
    x = f.reshape(-1, f.shape[-1])
    return x.T @ x / len(x), x.T @ (y-p).reshape(-1, 3) / len(x)


def weights(history):
    gram, rhs = (np.mean(v, axis=0) for v in zip(*history))
    return .35 * np.linalg.solve(gram + np.diag(gram.diagonal() + .015), rhs)


def screen(geo, cut):
    history = [[], []]
    folds, daily = [], []
    for first in range(8, 16, 2):
        w = [weights(h) for h in history] if history[0] else None
        rows = [[], []]
        for day in range(first, first + 2):
            raw = PARENT.before.book.rawday(day)
            p = reference(day)
            y = labels(day).reshape(p.shape)
            for j, matrix in enumerate(geo):
                t, f = features(raw, matrix, cut)
                assert len(t) == len(np.unique(t))
                if day == first:
                    end = t[len(t)//2] + 1
                    st, sf = features(raw[:end], matrix, cut)
                    np.testing.assert_array_equal(st, t[t < end])
                    np.testing.assert_allclose(sf, f[t < end], rtol=0, atol=1e-7)
                if w is not None:
                    q = p.copy()
                    q[t] += f @ w[j]
                    rows[j].append(tuple(a.reshape(-1, 3) for a in (p, q, y)))
                history[j].append(moments(f, p[t], y[t]))
            if w is not None:
                daily.append([delta(g[-1:]) for g in rows])
        if w is not None:
            folds.append([delta(g) for g in rows])
            print(first + 1, folds[-1], flush=True)
    gain = np.asarray(folds)
    keep = (gain[:, 1].min(0) > 0) & ((gain[:, 1]-gain[:, 0]).mean(0) > 0)
    keep &= (np.asarray(daily)[:, 1] > 0).sum(0) >= 4
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds, daily=daily, retained=keep.tolist()), indent=2))
    if keep.any():
        np.savez(OUT/'model.npz', transfer=geo[1], cut=cut, weights=weights(history[1])*keep)
    return keep.any()


def delta(rows):
    s = measure(rows)
    return (np.array(s['candidate']) - s['base']).tolist()


def load():
    parent = PARENT.load()
    with np.load(OUT/'model.npz') as data:
        matrix, cut, w = data['transfer'], float(data['cut']), data['weights']
    def predict(raw, p):
        out = parent(raw, p).copy()
        t, f = features(raw, matrix, cut)
        out[t] += f @ w
        return out
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    if not screen(geometry(), pulse.fit_cut()):
        return
    incumbent = json.loads((PARENT.OUT/'reload_verification.json').read_text())['pooled']['score']
    first = None
    for name in ('raw_replay', 'reload_verification'):
        res = walk.replay(load())
        if first is not None:
            np.testing.assert_allclose(res['pooled']['candidate'], first['pooled']['candidate'], rtol=0, atol=1e-10)
            for d in res['daily']:
                assert abs(res['daily'][d]['score'] - first['daily'][d]['score']) < 1e-10
        (OUT/f'{name}.json').write_text(json.dumps(res, indent=2))
        print(name, res['pooled']['score'], flush=True)
        if res['pooled']['score'] <= incumbent:
            return
        first = res


if __name__ == '__main__':
    main()
