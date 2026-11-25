"""Average causal, historically fitted graph recurrences for the Y3 forecast."""
import argparse
import json
import numpy as np

import experiment_y3_peer_shock_participation_v1 as incumbent
from y3_settled_walkforward_panels import panels, parent
from y3_portfolio_walkforward import replay, scores

area, peer = incumbent.area, incumbent.pressure
DIM = 28
OUT = parent.ART / 'y3_echo_vintage_ensemble_v1'
ARMS = ('incumbent', 'most recent replicated readout', 'mean of complete earlier models',
        'common sign across vintages', 'shared market changes', 'reassigned graph contrast')


def reference_days():
    graphs = [peer.graph_stat(peer.rawday(d)) for d in range(4)]
    old_records, pressure_records, participation_records = [], [], []
    for d, raw, ps, y in panels():
        if d % 2 == 0:
            links = peer.graph(graphs)
        src = peer.fields(raw, links)
        pat = incumbent.design(raw, links)
        qs = ps
        if d >= 8:
            if d % 2 == 0:
                ow = area.fit(old_records)
                pw = peer.fit(pressure_records)
                cw = incumbent.fit(participation_records)
            qs = ps.copy()
            qs[:, :, :, 2] += area.signals(raw, ow)[None, :, :, 5]
        ref = qs[-1] if d >= 8 else np.mean(qs[:2], 0)
        pressure_records.append(peer.moment(src, ref, y, d))
        if d >= 8:
            qs[:, :, :, 2] += peer.shifts(src, pw)[None, :, :, 3]
        participation_records.append(incumbent.collect(pat, ref, y, d))
        if d >= 8:
            qs[:, :, :, 2] += incumbent.changes(pat, cw)[None, :, :, 3]
        old_records.append(area.gram(raw, ps, y, d))
        graphs.append(peer.graph_stat(raw))
        yield d, raw, qs, y


def projection():
    rng = np.random.default_rng(81371)
    input_map = rng.standard_normal((100, DIM)).astype('float32')*.095
    special = rng.standard_normal((4, DIM)).astype('float32')*.4
    q = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype('float32')
    return input_map, special, q


def fit_part(rows):
    """Fit readouts, including the shared directions from two disjoint date sets."""
    mats = np.stack(rows)
    n = 4*DIM
    a, b, c, wrong = np.arange(n).reshape(4, DIM)

    def fit_on(gs):
        g = gs.mean(0)
        ref = g[n:n+3, n:n+3]
        def reg(s): return np.diag(np.diag(s).clip(.00001))
        slope = np.linalg.solve(ref+.025*reg(ref), g[n:n+3, -2:])
        unit = np.clip(g[n+2, -2]/max(g[n+2, n+2], 1e-6), .45, 2.5)
        gr = (gs[:, :n, -2:]-gs[:, :n, n:n+3]@slope)@np.array([.55, .45])/unit
        # Project each pair of dynamic features off the matching static state.

        def read(indices, controls):
            v = np.eye(n)[:, indices]
            if len(controls):
                cc = g[np.ix_(controls, controls)]
                v[controls] = -np.linalg.solve(cc+.045*reg(cc), g[np.ix_(controls, indices)])
            cov = v.T@g[:n, :n]@v
            day_gr = gr@v
            mean = day_gr.mean(0)
            mean *= abs(mean)/(abs(mean)+day_gr.std(0)/np.sqrt(len(gs))+.00002)
            return v@np.linalg.solve(cov+2.*reg(cov), mean)
        ctrl = np.r_[a, b]
        return np.column_stack((read(a, []), read(b, []), read(c, []), read(c, ctrl),
                                read(c[DIM//2:], ctrl), read(wrong, ctrl)))
    weights = fit_on(mats)
    x, y = [fit_on(mats[j::2])[:, 3] for j in (0, 1)]
    agree = np.where(x*y > 0, np.sign(x)*np.minimum(abs(x), abs(y)), 0.)
    return (.28*np.column_stack((weights[:, :5], agree, weights[:, -1]))).astype('float32')


def design(raw, graph):
    """Run all four graphs, resetting hidden states at the start of each session."""
    imap, special, rot = projection()
    n = DIM
    rate = np.repeat(np.array([.20, .032], np.float32), n//2)
    gs = [np.roll(np.roll(graph, s, 0), s, 1) for s in (0, 9, 27, 39)]
    features = []
    for lo, hi in parent.base.segments(len(raw)):
        state = np.zeros((4, 50, n), dtype='float32')
        local = np.zeros_like(state[0])
        z = np.empty((hi-lo, 50, 4, n), np.float32)
        for start in range(lo, hi, 128):
            v = raw[start:min(start+128, hi)].astype('float32')
            v[:, :, :83] = np.tanh(.75*np.arcsinh(v[:, :, :83]))
            v[:, :, 83:] -= 1
            q = v[:, :, [26, 38, 41, 42, 54, 55]]
            u = v@imap+np.concatenate((q[:, :, ::2]-q[:, :, 1::2], v[:, :, :1]), 2)@special
            for j, drive in enumerate(u):
                messages = gs@state
                state += (np.tanh(drive[None]+(.68*state+.32*messages)@rot)-state)*rate
                local += (np.tanh(drive+local@rot)-local)*rate
                null = state[1:].mean(0)
                z[start+j-lo] = np.stack((local, null, state[0]-null,
                                         state[1]-.5*(state[2]+state[3])), 1)
        features.append(z)
    return np.concatenate(features)


def rolling_mean(x, raw, h):
    result = []
    for lo, hi in parent.base.segments(len(raw)):
        v = x[lo:hi]
        cs = np.concatenate((np.zeros_like(v[:1], dtype=float), v.cumsum(0, dtype=float)))
        end = np.arange(len(v))+1
        start = (end-h).clip(0)
        denom = (end-start).reshape((-1,)+(1,)*(v.ndim-1))
        result.append(((cs[end]-cs[start])/denom).astype('float32'))
    return np.concatenate(result)


def split(z, raw):
    x = z.reshape(len(z), 50, -1)
    m = x.mean(1, keepdims=True)
    return x-m, m-rolling_mean(m, raw, 180)


def record(z, raw, p, y, d):
    x, m = split(z, raw)
    t = np.concatenate([np.arange(a+d % 8, b-60, 8) for a, b in parent.base.segments(len(z))])
    goals = np.concatenate((p[t], parent.exposure.bands(y[t])), 2)
    mean = goals.mean(1, keepdims=True)

    def moment(v):
        a = v.reshape(-1, v.shape[-1]).astype(float)
        a -= a.mean(0)
        return a.T@a/len(a)
    return moment(np.concatenate((x[t], goals-mean), 2)), moment(np.concatenate((m[t], mean), 2))


def fit(rows):
    return tuple(fit_part(part) for part in zip(*rows))


def changes(raw, bank):
    """Evaluate each readout on its own graph before combining its forecast."""
    aligned = []
    parts = []
    wrong = []
    for graph, w in bank:
        z = design(raw, graph)
        x, m = split(z, raw)
        direct = (x@w[0][:, 5:6])[:, :, 0]
        market = (m@w[1][:, 5:6])[:, :, 0]
        # Both negative control and true graphs use the same four-state evaluation.
        placebo = (x@w[0][:, 6:7]+m@w[1][:, 6:7])[:, :, 0]
        aligned.append(direct+market)
        parts.append(market)
        wrong.append(placebo)
    stack = np.stack(aligned)
    low, high = stack.min(0), stack.max(0)
    shared = np.where(low*high > 0, np.sign(low)*np.minimum(abs(low), abs(high)), 0.)
    market = np.broadcast_to(np.mean(parts, 0), shared.shape)
    return np.stack((stack[-1], stack.mean(0), shared, market, np.mean(wrong, 0)), 2)


def train():
    graphs = [peer.graph_stat(peer.rawday(d)) for d in range(4)]
    hist, bank, days = [], [], []
    for d, raw, ps, y in reference_days():
        if d % 2 == 0:
            graph = peer.graph(graphs)[2]
            if d >= 8:
                bank.append((graph, fit(hist)))
        if d >= 8:
            delta = changes(raw, bank)
            if d in (8, 10):
                cut = 2451
                np.testing.assert_allclose(delta[:cut], changes(raw[:cut], bank), atol=3e-6, rtol=0)
            days.append([parent.base.records(p, y, delta) for p in ps])
        z = design(raw, graph)
        p = ps[-1] if d >= 8 else np.mean(ps[:2], 0)
        hist.append(record(z, raw, p, y, d))
        graphs.append(peer.graph_stat(raw))
        print('day', d+1, flush=True)
    h = np.stack([scores(v, len(ARMS)) for v in (days[:4], days[4:])])
    ds = np.stack([scores([v], len(ARMS)) for v in days])
    j = 2
    keep = h[:, -1, 0, j].min() > 0 and (ds[:, -1, 0, j] >
                                         0).sum() >= 5 and (h[:, -1, 0, j]-h[:, -1, 0, 5]).mean() > 0
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,
                                                    selected=j if keep else 0, halves=h.tolist(), daily=ds.tolist()), indent=2))
    print('complete vintages selected', j if keep else 0, 'gain',
          h[:, -1, :, j], 'positive days', int((ds[:, -1, 0, j] > 0).sum()), flush=True)
    if keep:
        bank.append((peer.graph(graphs)[2], fit(hist)))
        np.savez(OUT/'model.npz', graphs=np.stack([g for g,
                 w in bank]), weights=np.stack([w for g, w in bank]))


def correction():
    prior = incumbent.correction()
    with np.load(OUT/'model.npz') as f:
        bank = list(zip(f['graphs'], f['weights']))
    return lambda raw, p: prior(raw, p)+changes(raw, bank)[:, :, 1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'verify'))
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        replay(OUT, correction)
