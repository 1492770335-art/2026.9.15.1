"""Estimate stock-specific responses to comparable book states within the same historical time block."""
import json
import numpy as np

import experiment_cutoff_transition_rerun_v1 as best
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

OUT = ART/'within_stock_state_matches_v1'
BOOK = best.walk.book
WIDTH = 320
MIX = .30
R12 = .272/.559
R23 = best.walk.blend.panel.RATIO


def states(raw, t):
    def col(i):
        return np.arcsinh(raw[:, :, i].astype('float32'))
    imbalance = col(21)
    pressure = col(41)-col(42)
    execution = col(54)-col(55)
    price = (raw[:, :, 0].astype(float)-.00136).cumsum(0)
    c = np.concatenate((np.zeros((1, 50, 3)), np.stack((imbalance, pressure, execution), 2).cumsum(0, dtype=float)))
    left = (t+1-16).clip(0)
    trailing = (c[t+1]-c[left])/(t+1-left)[:, None, None]
    now = np.stack((imbalance[t], pressure[t], execution[t]), 2)
    other = np.stack((col(39)[t], col(8)[t], (price[t]-price[(t-24).clip(0)])/np.sqrt(24),
                      col(26)[t]-col(38)[t]), 2)
    return np.concatenate((now, now-trailing, other), 2).astype('float32')


def geometry():
    days = [states(BOOK.rawday(d), np.arange(d, 4650, 19)) for d in range(4)]
    x = np.concatenate(days)
    center = np.median(x, 0)
    scale = .5*(np.quantile(x, .82, axis=0)-np.quantile(x, .18, axis=0))
    scale = np.maximum(scale, .2*np.median(scale, 0)+.04)
    z = np.tanh((x-center)/scale).reshape(-1, x.shape[2])
    rng = np.random.default_rng(70918)
    knots = z[rng.choice(len(z), 40, replace=False)]
    risk = np.median([BOOK.rawday(d)[:, :, 0].astype(float).std(0) for d in range(4)], 0).clip(.2, 4)
    return dict(center=center.astype('float32'), scale=scale.astype('float32'),
                knots=knots.astype('float32'), risk=risk.astype('float32'))


def basis(raw, t, geo):
    z = np.tanh((states(raw, t)-geo['center'])/geo['scale'])
    knots = geo['knots']
    dist = ((z*z).sum(2, keepdims=True)+(knots*knots).sum(1)-2*z@knots.T).clip(0)
    return np.concatenate((z, np.exp(-dist/3)), 2)


def remove_group(x, groups):
    result = np.empty_like(x)
    for g in np.unique(groups):
        ix = groups==g
        result[ix] = x[ix]-x[ix].mean(0)
    return result


def statistics(raw, y, d, geo):
    t = np.arange(d%5, len(raw)-60, 5)
    x = basis(raw, t, geo)
    y = y[t].copy()
    y[:, :, 2] -= R23*y[:, :, 1]
    y[:, :, 1] -= R12*y[:, :, 0]
    y -= y.mean(1, keepdims=True)
    y = (y/geo['risk'][None, :, None]).clip(-4, 4)
    results = []
    for groups in (np.zeros(len(t), int), t//WIDTH):
        a, b = (remove_group(z, groups).transpose(1, 0, 2).astype(float) for z in (x, y))
        results.append((a.transpose(0, 2, 1)@a/len(t), a.transpose(0, 2, 1)@b/len(t)))
    return results


def weights(history, geo, arm):
    g, b = (np.stack(x) for x in zip(*(h[arm] for h in history)))
    # A stock-specific coefficient is a deviation from the fit of its volatility cohort.
    order = np.argsort(geo['risk']).reshape(5, 10)
    reg = .70*np.diag(g.mean((0, 1)).diagonal().clip(.002))
    base = np.empty_like(b[0])
    for group in order:
        base[group] = np.linalg.solve(g[:, group].mean((0, 1))+reg, b[:, group].mean((0, 1)))
    rem = b-g@base
    pooled = np.linalg.solve(g.mean(0)+reg, rem.mean(0))
    per = np.linalg.solve(g+reg, rem)
    return ((.65*pooled+.35*np.median(per, 0))*abs(np.sign(per).mean(0))).astype('float32')


def signal(x, w, geo):
    a = np.einsum('tsf,sfh->tsh', x, w, optimize=True)*geo['risk'][None, :, None]
    a[:, :, 1] += R12*a[:, :, 0]
    a[:, :, 2] += R23*a[:, :, 1]
    return MIX*(a-a.mean(1, keepdims=True))


def load():
    with np.load(OUT/'model.npz') as st:
        w, mask = st['w'], st['mask']
        geo = {k:st[k] for k in ('center','scale','risk','knots')}
    parent = best.load()
    def apply(raw, p):
        q = parent(raw, p)
        out = q.copy()
        for k in range(min(4, len(q))):
            t = np.arange(k, len(q), 4)
            out[t] += mask*signal(basis(raw, t, geo), w, geo)
        return out
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    geo = geometry()
    history, folds, daily = [], [], []
    for end, stop in ((8, 12), (12, 14), (14, 16)):
        for d in range(len(history), end):
            history.append(statistics(BOOK.rawday(d), labels(d).reshape(-1, 50, 3), d, geo))
        ws = [weights(history, geo, k) for k in (0, 1)]
        groups = [[[], []] for _ in range(2)]
        for d in range(end, stop):
            raw = BOOK.rawday(d)
            t = np.arange(d%4, len(raw), 4)
            x = basis(raw, t, geo)
            if d==end:
                select = t<237
                np.testing.assert_allclose(x[select], basis(raw[:237], t[select], geo), rtol=0, atol=4e-6)
            y = labels(d).reshape(-1, 50, 3)[t]
            for i, w in enumerate(ws):
                delta = signal(x, w, geo)
                for j, name in enumerate(('panel','neural')):
                    p = np.load(best.refs.OUT/f'{name}_{d}.npy')[t]
                    row = tuple(z.reshape(-1, 3) for z in (p, p+delta, y))
                    groups[j][i].append(row)
                    if not j and i:
                        s = measure([row])
                        daily.append(np.subtract(s['candidate'], s['base']))
        f = [[measure(v) for v in group] for group in groups]
        folds.append(f)
        print(end+1, stop, [[s['gain'] for s in g] for g in f], flush=True)
    gain = np.array([[np.subtract(b['candidate'], b['base']) for a,b in f] for f in folds])
    edge = np.array([[np.subtract(b['candidate'], a['candidate']) for a,b in f] for f in folds])
    keep = (gain[:,0].min(0)>0)&(gain[:,1].mean(0)>0)&(edge[:,0].mean(0)>0)&((np.stack(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds, daily=np.stack(daily).tolist(), retained=keep.tolist()), indent=2))
    if keep.any():
        for d in range(len(history), 16):
            history.append(statistics(BOOK.rawday(d), labels(d).reshape(-1, 50, 3), d, geo))
        np.savez(OUT/'model.npz', **geo, w=weights(history, geo, 1), mask=keep)
        result = best.walk.replay(load())
        s = json.loads((best.OUT/'raw_replay.json').read_text())['pooled']['score']
        result['gain_over_best'] = result['pooled']['score']-s
        (OUT/'raw_replay.json').write_text(json.dumps(result, indent=2))
        print(result['pooled'], result['gain_over_best'], flush=True)
        if result['gain_over_best']>0:
            rep = best.walk.replay(load())
            np.testing.assert_allclose(rep['pooled']['candidate'], result['pooled']['candidate'], atol=1e-10, rtol=0)
            (OUT/'reload_verification.json').write_text(json.dumps(rep, indent=2))


if __name__=='__main__':
    main()
