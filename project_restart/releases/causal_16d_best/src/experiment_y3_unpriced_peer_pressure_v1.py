"""Forecast delayed relative returns after pressure arrives at a conditionally co-moving peer before a recipient reprices."""
import argparse
import json
import numpy as np
from experiment_pairwise_book_tree_v1 import rawday
from y3_settled_walkforward_panels import panels, parent
from y3_portfolio_walkforward import scores, replay
import experiment_y3_ordered_book_area_v1 as area

OUT = parent.ART/'y3_unpriced_peer_pressure_v1'
ARMS = ('incumbent', 'peer pressure', 'unpriced pressure', 'conditional network', 'replicated links',
        'reassigned links', 'conditional above marginal', 'receiver has repriced')


def graph_stat(raw):
    samples = []
    for lo, hi in parent.base.segments(len(raw)):
        r = (raw[lo:hi, :, 0].astype(float)-.00136).clip(-4, 4)
        cs = np.vstack((np.zeros(50), r.cumsum(0)))
        for h in (5, 20):
            end = np.arange(h, len(r)+1, h)
            x = cs[end]-cs[end-h]
            x -= np.median(x, axis=1, keepdims=True)
            x = np.arcsinh(x/(.15+np.median(abs(x), axis=0)))
            x -= x.mean(0)
            sd = np.sqrt((x*x).mean(0)+.001)
            z = x/sd
            samples.append(z.T@z/len(z))
    return np.mean(samples, 0)


def graph(rows):
    def estimate(vals):
        c = np.mean(vals, 0)
        inv = np.linalg.inv(c+.40*np.eye(50))
        partial = -inv/np.sqrt(np.outer(inv.diagonal(), inv.diagonal()))
        np.fill_diagonal(partial, 0)
        c = c.copy()
        np.fill_diagonal(c, 0)
        return c, partial
    common, partial = estimate(rows)
    a, b = [estimate(rows[i::2])[1] for i in (0, 1)]
    stable = np.where(a*b > 0, np.sign(a)*np.minimum(abs(a), abs(b)), 0.)
    out = []
    for v in (common, partial, stable):
        w = np.sign(v)*np.maximum(abs(v)-np.quantile(abs(v), .80, axis=1, keepdims=True), 0)
        w /= abs(w).sum(1, keepdims=True).clip(.01)
        out.append(w.astype('float32'))
    out.append(np.roll(np.roll(out[1], 17, axis=1), 17, axis=0))
    return out


def fields(raw, links):
    parts = []
    for lo, hi in parent.base.segments(len(raw)):
        r = raw[lo:hi]
        q = np.tanh(.8*np.arcsinh(r[:, :, [26, 38, 41, 42, 54, 55]].astype('float32')))
        flow = q[:, :, ::2]-q[:, :, 1::2]
        ret = np.tanh(r[:, :, 0:1].astype('float32')-.00136)
        values = np.concatenate((flow, ret), 2)
        values -= values.mean(1, keepdims=True)
        cs = np.concatenate((np.zeros_like(values[:1], dtype=float), values.cumsum(0, dtype=float)))
        t = np.arange(len(r))+1

        def avg(h):
            begin = (t-h).clip(0)
            return ((cs[t]-cs[begin])/(t-begin)[:, None, None]).astype(np.float32)
        fast, slow = avg(6), avg(48)
        innovation = fast[:, :, :3]-slow[:, :, :3]
        own_move = fast[:, :, 3]-.5*slow[:, :, 3]
        rows = []
        for w in links:
            source = np.stack([innovation[:, :, k]@w.T for k in range(3)], 2)
            # Where the recipient has already moved along the donor's pressure, the signal may be spent.
            moved = np.tanh(8*own_move[:, :, None]*np.sign(source))
            quiet = source*(1-moved)*.5
            confirmed = source*(1+moved)*.5
            rows.append(np.concatenate((source, quiet, confirmed), 2))
        parts.append(np.stack(rows, axis=2))
    x = np.concatenate(parts)
    return x-x.mean(1, keepdims=True)


def moment(z, p, y, d):
    t = np.concatenate([np.arange(lo+d % 7, hi-60, 7) for lo, hi in parent.base.segments(len(p))])
    pr = parent.exposure.bands(p[t])
    yt = parent.exposure.bands(y[t])
    v = np.concatenate((z[t].reshape(len(t), 50, -1), pr, yt), 2)
    v = (v-v.mean(1, keepdims=True)).reshape(-1, v.shape[2]).astype(float)
    return v.T@v/len(v)


def fit(rows):
    gs = np.stack(rows)
    g = gs.mean(0)

    def read(graph, subset):
        ix = 9*graph+np.array(subset)
        xx = g[np.ix_(ix, ix)]
        units = (np.diag(g[-4:-2, -2:])/g.diagonal()[-4:-2]).clip(.45, 2.5)
        grads = gs[:, ix, -2:]/units-gs[:, ix, -4:-2]
        mu = grads.mean(0)
        mu *= abs(mu)/(abs(mu)+grads.std(0)/np.sqrt(len(rows))+.00003)
        avg = mu@np.array([.45, .55])
        w = np.zeros(36)
        w[ix] = np.linalg.solve(xx+1.3*np.diag(xx.diagonal().clip(.00004)), avg)
        return w
    base, quiet, spent = np.arange(3), np.arange(3, 6), np.arange(6, 9)
    marginal, part = read(0, quiet), read(1, quiet)
    return (.35*np.column_stack((read(0, base), marginal, part, read(2, quiet), read(3, quiet), part-marginal, read(1, spent)))).astype('float32')


def shifts(z, w):
    return z.reshape(len(z), 50, -1)@w


def train():
    gd = [graph_stat(rawday(d)) for d in range(4)]
    hist, prior, days = [], [], []
    for d, raw, ps, y in panels():
        if d % 2 == 0:
            links = graph(gd)
        z = fields(raw, links)
        qs = ps
        if d >= 8:
            if d % 2 == 0:
                w = fit(hist)
                old = area.fit(prior)
            qs = ps.copy()
            qs[:, :, :, 2] += area.signals(raw, old)[None, :, :, 5]
            delta = shifts(z, w)
            if d == 8:
                np.testing.assert_allclose(delta[:2451], shifts(
                    fields(raw[:2451], links), w), atol=2e-6, rtol=0)
            days.append([parent.base.records(p, y, delta) for p in qs])
        ref = qs[-1] if d >= 8 else np.mean(qs[:2], 0)
        hist.append(moment(z, ref, y, d))
        gd.append(graph_stat(raw))
        prior.append(area.gram(raw, ps, y, d))
        print('through', d+1, flush=True)
    h = np.stack([scores(v, len(ARMS)) for v in (days[:4], days[4:])])
    daily = np.stack([scores([v], len(ARMS)) for v in days])
    j = max((3, 4, 6), key=lambda i: h[0, :, 0, i].min())
    keep = (h[:, -1, 0, j].min() > 0 and (daily[:, -1, 0, j] > 0).sum()
            >= 5 and (h[:, -1, 0, j]-h[:, -1, 0, 5]).mean() > 0)
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS, choice=j,
                                                    selected=j if keep else 0, halves=h.tolist(), daily=daily.tolist()), indent=2))
    if keep:
        np.savez(OUT/'model.npz', links=graph(gd), w=fit(hist), pick=j)
    print('choice', ARMS[j], 'selected', j if keep else 0, 'strict Y3', h[:, -1, 0, j], flush=True)


def correction():
    prior = area.correction()
    with np.load(OUT/'model.npz') as f:
        links, w, j = f['links'], f['w'], int(f['pick'])
    return lambda raw, p: prior(raw, p)+shifts(fields(raw, links), w)[:, :, j-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'screen', 'verify'))
    parser.add_argument('--upstream-dir')
    args = parser.parse_args()
    if args.mode == 'screen':
        if not args.upstream_dir:
            parser.error('screen requires --upstream-dir')
        replay(OUT, correction, args.upstream_dir)
    elif args.upstream_dir:
        parser.error('--upstream-dir is only supported by screen')
    elif args.mode == 'train':
        train()
    else:
        replay(OUT, correction)
