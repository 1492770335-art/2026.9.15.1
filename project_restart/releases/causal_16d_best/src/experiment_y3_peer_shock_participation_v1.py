"""Revise replicated peer-pressure signals according to the effective number and sign agreement of their active donors."""
import argparse
import json
import numpy as np
import experiment_y3_unpriced_peer_pressure_v1 as pressure
from y3_settled_walkforward_panels import panels, parent
from y3_portfolio_walkforward import scores, replay

area = pressure.area
OUT = parent.ART/'y3_peer_shock_participation_v1'
ARMS = ('pressure incumbent', 'participation', 'signed consensus', 'consensus among broad shocks',
        'concentrated shocks', 'stock-specific activity above its history', 'reassigned breadth')


def design(raw, links):
    fields = []
    for a, b in parent.base.segments(len(raw)):
        r = raw[a:b]
        q = np.tanh(.8*np.arcsinh(r[:, :, [26, 38, 41, 42, 54, 55]].astype('float32')))
        own = q[:, :, ::2]-q[:, :, 1::2]
        ret = np.tanh(r[:, :, 0].astype('float32')-.00136)
        end = np.arange(len(r))+1

        def avg(v, h):
            cs = np.concatenate((np.zeros_like(v[:1], dtype=float), v.cumsum(0, dtype=float)))
            start = (end-h).clip(0)
            return ((cs[end]-cs[start])/(end-start).reshape((-1,)+(1,)*(v.ndim-1))).astype(np.float32)
        shock = avg(own, 6)-avg(own, 48)
        move = avg(ret, 6)-.5*avg(ret, 48)
        w = links[2]
        aw = abs(w)
        z = []
        for k in range(3):
            x = shock[:, :, k]
            x = x-x.mean(1, keepdims=True)
            signed = x@w.T
            gross = abs(x)@aw.T
            energy = (x*x)@(w*w).T
            count = gross*gross/(energy+.00002)
            max_count = 1/(w*w).sum(1).clip(.00001)
            breadth = np.minimum(count/max_count, 1.5)
            accord = abs(signed)/(gross+.009)
            trust = 1-np.tanh(8*move*np.sign(signed))
            base = .5*trust*signed
            z.append(np.stack((base, base*breadth, base*accord, base*breadth*accord, base*(1-breadth),
                              base*(breadth-avg(breadth, 160)), base*np.roll(breadth, 17, 1)), 2))
        fields.append(np.stack(z, axis=3))
    v = np.concatenate(fields)
    return v-v.mean(1, keepdims=True)


def collect(z, p, y, d):
    t = np.concatenate([np.arange(a+d % 7, b-60, 7) for a, b in parent.base.segments(len(p))])
    xx = z[t].reshape(len(t), 50, -1)
    pred = parent.exposure.bands(p[t])
    truth = parent.exposure.bands(y[t])
    v = np.concatenate((xx, pred, truth), 2)
    v = (v-v.mean(1, keepdims=True)).reshape(-1, v.shape[2]).astype(float)
    return v.T@v/len(v)


def fit(hist):
    rows = np.stack(hist)
    g = rows.mean(0)
    units = (np.diag(g[-4:-2, -2:])/g.diagonal()[-4:-2]).clip(.45, 2.5)
    ctrl = g[:3, :3]
    out = []
    for j in range(1, 7):
        idx = np.arange(3*j, 3*j+3)
        proj = np.linalg.solve(ctrl+.04*np.diag(ctrl.diagonal().clip(.00001)), g[:3, idx])
        select = np.zeros((21, 3))
        select[:3] = -proj
        select[idx] = np.eye(3)
        gram = select.T@g[:21, :21]@select
        grad = ((rows[:, :21, -2:]/units-rows[:, :21, -4:-2])@np.array([.6, .4]))@select
        mu = grad.mean(0)
        mu *= abs(mu)/(abs(mu)+grad.std(0)/np.sqrt(len(rows))+.00002)
        out.append(select@np.linalg.solve(gram+1.6*np.diag(gram.diagonal().clip(.00003)), mu))
    return (.35*np.column_stack(out)).astype('float32')


def changes(z, w):
    return z.reshape(len(z), 50, -1)@w


def train():
    graphs = [pressure.graph_stat(pressure.rawday(d)) for d in range(4)]
    phist, hist, prior, days = [], [], [], []
    for d, raw, ps, y in panels():
        if d % 2 == 0:
            links = pressure.graph(graphs)
        z = design(raw, links)
        source = pressure.fields(raw, links)
        qs = ps
        if d >= 8:
            if d % 2 == 0:
                w = fit(hist)
                old = area.fit(prior)
                pw = pressure.fit(phist)
            qs = ps.copy()
            qs[:, :, :, 2] += area.signals(raw, old)[None, :, :, 5]
            before = qs[-1].copy()
            qs[:, :, :, 2] += pressure.shifts(source, pw)[None, :, :, 3]
            delta = changes(z, w)
            if d == 8:
                np.testing.assert_allclose(delta[:2451], changes(
                    design(raw[:2451], links), w), atol=2e-6, rtol=0)
            days.append([parent.base.records(p, y, delta) for p in qs])
        else:
            before = np.mean(qs[:2], 0)
        ref = qs[-1] if d >= 8 else before
        hist.append(collect(z, ref, y, d))
        phist.append(pressure.moment(source, before, y, d))
        graphs.append(pressure.graph_stat(raw))
        prior.append(area.gram(raw, ps, y, d))
        print('through', d+1, flush=True)
    h = np.stack([scores(v, len(ARMS)) for v in (days[:4], days[4:])])
    daily = np.stack([scores([v], len(ARMS)) for v in days])
    j = max(range(1, 6), key=lambda i: h[0, :, 0, i].min())
    keep = (h[:, -1, 0, j].min() > 0 and (daily[:, -1, 0, j] > 0).sum()
            >= 5 and (h[:, -1, 0, j]-h[:, -1, 0, 6]).mean() > 0)
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS, choice=j,
                                                    selected=j if keep else 0, halves=h.tolist(), daily=daily.tolist()), indent=2))
    if keep:
        np.savez(OUT/'model.npz', links=pressure.graph(graphs), w=fit(hist), pick=j)
    print('selected', j if keep else 0, 'choice', ARMS[j], 'strict Y3', h[:, -1, 0, j], flush=True)


def correction():
    prior = pressure.correction()
    with np.load(OUT/'model.npz') as f:
        links, w, j = f['links'], f['w'], int(f['pick'])
    return lambda raw, p: prior(raw, p)+changes(design(raw, links), w)[:, :, j-1]


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
