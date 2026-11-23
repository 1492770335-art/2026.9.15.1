"""Compare direct delayed returns with a discrete order-book transition and reward model."""
import hashlib
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure
import experiment_pairwise_book_tree_v1 as book
import experiment_tail_portfolio_filter_v1 as panel
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_seasonal_pressure_surprises_v1 as parent

OUT = ART / 'book_transition_rewards_v1'
FIELDS = (0, 2, 3, 5, 7, 8, 10, 12, 19, 21, 23, 28, 29, 30)
H = (5, 20, 40)


def features(raw, t):
    state, vol = book.state(raw)
    now = state[t][:, :, FIELDS]
    changes = state[t, :, 31:][:, :, ::3]
    x = np.concatenate((now, changes, np.log(vol[t, :, None])), axis=2)
    mean = x.mean(1, keepdims=True)
    strength = now.std(1)
    market = np.column_stack((mean[:, 0], strength, np.exp(-t/400)))
    local = np.concatenate((x-mean, np.broadcast_to(mean[:, :, :4], (len(t), 50, 4))), axis=2)
    return market.astype('float32'), local.reshape(-1, local.shape[-1]).astype('float32')


def codebooks():
    probes = [[], []]
    rng = np.random.default_rng(38321)
    for d in range(8):
        raw = book.rawday(d)
        t = np.arange(d%5, len(raw), 5)
        for i, x in enumerate(features(raw, t)):
            probes[i].append(x if i == 0 else x[rng.choice(len(x), 4200, replace=False)])
    geometry = []
    for rows, n in zip(probes, (48, 112)):
        x = np.concatenate(rows)
        center = np.median(x, axis=0)
        scale = np.maximum(np.quantile(abs(x-center), .75, axis=0), .07).astype('float32')
        z = ((x-center)/scale).clip(-4, 4).astype('float32')
        model = MiniBatchKMeans(n_clusters=n, random_state=92, batch_size=1024, n_init=3).fit(z)
        geometry.append((center, scale, model.cluster_centers_))
    return geometry


def assign(x, geo):
    center, scale, centers = geo
    bias = np.sum(centers*centers, axis=1)
    ids = []
    for start in range(0, len(x), 4096):
        z = ((x[start:start+4096]-center)/scale).clip(-4, 4)
        ids.append(np.argmin(bias[None] - 2*(z@centers.T), axis=1))
    return np.concatenate(ids)


def states(raw, ticks, geo):
    return [assign(x, g).reshape(len(ticks), -1) for x, g in zip(features(raw, ticks), geo)]


def statistics(raw, d, geo):
    t = np.arange(d%5, len(raw)-61, 5)
    all_t = np.unique(np.concatenate([t]+[t+h for h in H]))
    series = states(raw, all_t, geo)
    prices = (raw[:, :, 0].astype(float)-.00136).cumsum(0)
    targets = [prices[t+h]-prices[t] for h in H]
    targets.append(prices[t+60]-prices[t+20])
    vol = np.sqrt(.3 + np.mean((prices[t+5]-prices[t])**2, axis=0)/5).clip(.5, 4)
    stats = []
    for i, (s, g) in enumerate(zip(series, geo)):
        n = len(g[2])
        ids = s[np.searchsorted(all_t, t)].ravel()
        counts = np.bincount(ids, minlength=n)
        sums = []
        for actual in targets:
            avg = actual.mean(1, keepdims=True)
            value = avg if i == 0 else (actual-avg)/vol
            sums.append(np.bincount(ids, weights=value.clip(-40, 40).ravel(), minlength=n))
        pairs = []
        for h in H:
            later = s[np.searchsorted(all_t, t+h)].ravel()
            pairs.append(np.bincount(ids*n+later, minlength=n*n).reshape(n, n))
        stats.append((counts, np.stack(sums), np.stack(pairs)))
    return stats


def fit_rewards(history):
    out = []
    for i, days in enumerate(zip(*history)):
        counts, sums, pairs = (np.sum(block, axis=0) for block in zip(*days))
        shrink = (8 if i == 0 else 45)*len(days)
        freq = (counts+1)/(counts.sum()+len(counts))
        reward = (sums+shrink*(sums.sum(1)/counts.sum())[:, None])/(counts+shrink)
        prob = (pairs+shrink*freq[None, None, :])/(counts+shrink)[None, :, None]
        five, twenty, forty = prob
        # Future 40-step payout after reaching the state at t+20.
        bridge = twenty @ reward[2]
        # Two 20-step payouts and eight 5-step payouts provide alternative transition lengths.
        split = twenty @ reward[1] + forty @ reward[1]
        short = reward[0].copy()
        for _ in range(7):
            short = reward[0] + five @ short
        boot = .5*bridge + .35*split + .15*(twenty @ short)
        out.append(np.column_stack((reward[3], boot)).astype('float32')*.161)
    return out


def estimates(raw, ticks, geo, rewards):
    m, r = states(raw, ticks, geo)
    a = rewards[0][m]
    b = rewards[1][r]
    ret = raw[:, :, 0].astype(float)-.00136
    cs = np.vstack((np.zeros(50), np.cumsum(ret*ret, axis=0)))
    end = ticks+1
    start = (end-120).clip(0)
    local_vol = np.sqrt((10+cs[end]-cs[start])/(10+end-start)[:, None]).clip(.5, 4)
    b *= local_vol[:, :, None]
    return a, b-b.mean(1, keepdims=True)


def weights(history):
    if not history:
        return np.zeros((2, 2))
    num, den = map(np.stack, zip(*history))
    reg = np.maximum(den.mean(0), .00002)*1.2
    daily = num/(den+reg)
    return (.6*(.5*num.mean(0)/(den.mean(0)+reg)+.5*np.median(daily, axis=0))*abs(np.sign(daily).mean(0))).clip(-.45, .45)


def moments(p, y, z):
    error = panel.target_residual(p, y)
    common = error.mean(1, keepdims=True)
    nums, dens = [], []
    for x, e in zip(z, (common, error-common)):
        nums.append((x*e[:, :, None]).mean((0, 1)))
        dens.append((x*x).mean((0, 1)))
    return np.stack(nums), np.stack(dens)


def replay(predict):
    import replay_session_experts_16d as original
    import polars as pl
    rows = []
    daily = {}
    ref = json.loads((parent.OUT/'raw_replay.json').read_text())
    for d in range(16, 20):
        raw = original.features(original.DATA, d)
        with np.load(parent.OUT/'raw_trace'/f'day_{d}.npz') as trace:
            if trace['raw_sha256'].item() != hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest():
                raise ValueError('raw inputs differ from replay')
            p = trace['full']
            n = int(trace['prefix_ticks'])
            q = predict(raw, p)
            short = predict(raw[:n], trace['head'])
        err = float(np.max(abs(q[:n]-short)))
        if not np.isfinite(q).all() or err >= .0001:
            raise ValueError('invalid prefix or grid')
        y = pl.scan_parquet(original.DATA).filter(pl.col('date') == d).sort('time', 'code').select('y0','y1','y2').collect().to_numpy()
        row = p.reshape(-1,3), q.reshape(-1,3), y
        rows.append(row)
        s = measure([row])
        if abs(s['baseline_score']-ref['daily'][str(d+1)]['score']) > 2e-6:
            raise ValueError('parent day mismatch')
        daily[d+1] = dict(score=s['score'], gain=s['gain'], prefix_error=err)
    pooled = measure(rows)
    if pooled['rows'] != 936250:
        raise ValueError('incomplete development coverage')
    return dict(pooled=pooled, daily=daily)


def main():
    OUT.mkdir(exist_ok=True)
    geo = codebooks()
    history = [statistics(book.rawday(d), d, geo) for d in range(8)]
    stat, blend_stat = [], []
    report = {'folds': []}
    daily = []
    for end in (8, 10, 12, 14):
        w = weights(stat)
        c = blend.fixed_weight(blend_stat)
        rewards = fit_rewards(history)
        groups = []
        for d in range(end, end+2):
            raw = book.rawday(d)
            p, y, fs, b = blend.training_day(d, blend_stat)
            blend_stat.append(b)
            t = np.arange(d%3, len(p), 3)
            p = blend.parent.fixed_forecast(p, fs, c)[t]
            y = y[t]
            m, r = estimates(raw, t, geo, rewards)
            if d == end:
                mask = t < 211
                small = estimates(raw[:211], t[mask], geo, rewards)
                for s, v in zip(small, (m, r)):
                    np.testing.assert_allclose(s, v[mask], atol=1e-6, rtol=0)
            z = w[0]*m + w[1]*r
            if end >= 10:
                groups.append((p, y, z))
                daily.append(panel.scores(p, y, z[:, :, 1])-panel.scores(p, y, 0))
            stat.append(moments(p, y, (m, r)))
            history.append(statistics(raw, d, geo))
        if groups:
            p, y, z = map(np.concatenate, zip(*groups))
            entry = dict(days=[end+1,end+2], baseline=panel.scores(p,y,0).tolist(),
                         direct=panel.scores(p,y,z[:,:,0]).tolist(), transition=panel.scores(p,y,z[:,:,1]).tolist())
            report['folds'].append(entry)
            print(entry, flush=True)
    gain = np.stack([np.subtract(f['transition'], f['baseline']) for f in report['folds']])
    comp = np.stack([np.subtract(f['transition'], f['direct']) for f in report['folds']])
    gate = bool((gain.mean(0)>0).all() and (gain[-1]>0).all() and ((gain>0).sum(0)>=2).all()
                and (comp.mean(0)>0).all() and ((np.stack(daily)>0).sum(0)>=4).all())
    report['retained'] = gate
    (OUT/'results.json').write_text(json.dumps(report, indent=2))
    if gate:
        rewards = fit_rewards(history)
        w = weights(stat)[:,1]
        arrays = {f'geo_{i}_{j}': val for i, g in enumerate(geo) for j, val in enumerate(g)}
        np.savez(OUT/'model.npz', **arrays, market=rewards[0], stock=rewards[1], w=w)
        def predict(raw, p):
            m, r = estimates(raw, np.arange(len(p)), geo, rewards)
            q = p.copy()
            q[:, :, 2] += w[0]*m[:, :, 1] + w[1]*r[:, :, 1]
            return q
        result = replay(predict)
        (OUT/'raw_replay.json').write_text(json.dumps(result, indent=2))
        print(result, flush=True)


if __name__ == '__main__':
    main()
