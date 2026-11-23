"""Compare buy-side and sell-side quote refresh histories in matched stock portfolios."""
import json
import numpy as np

import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_refresh_cohort_transfer_v1 as cohort
import experiment_netted_execution_cost_v1 as costs
import experiment_pairwise_book_tree_v1 as book
import replay_refresh_y3_specialist_v1 as specialist
from experiment_book_transition_rewards_v1 import replay
from experiment_causal_cross_horizon_ridge_v1 import ART, measure
from experiment_eight_date_expert_contrasts_v1 import fit, moments

OUT = ART / 'asymmetric_refresh_transfer_v1'


def clocks(raw, floor):
    return np.stack((raw[:, :, 77] == floor[3], raw[:, :, 78] == floor[4]), axis=-1)


def geometry():
    _, peers, pairs = cohort.geometry()
    floor = costs.decode_geometry()['floor']
    rates = np.median([clocks(book.rawday(d), floor).mean(0) for d in range(8)], axis=0)
    # Swap exactly one stock in each historical matched pair for the control.
    flip = np.arange(len(pairs)) < pairs
    return dict(peers=peers, rates=rates.clip(.01,.95).astype('float32'), floor=floor, flip=flip)


def features(raw, p, geo, placebo):
    tick = np.arange(len(p))[:, None, None]
    event = clocks(raw, geo['floor'])
    last = np.maximum.accumulate(np.where(event, tick, -1), axis=0)
    fresh = np.where(last < 0, 0., np.exp(-.75 * (tick-last) * geo['rates'])).astype('float32')
    if placebo:
        fresh[:,geo['flip']] = fresh[:,geo['flip'],::-1]
    bid, ask = np.moveaxis(fresh, 2, 0)
    band = p.copy()
    band[:,:,2] -= blend.panel.RATIO*p[:,:,1]
    band[:,:,1] -= (.272/.559)*p[:,:,0]
    revision = band-band[(tick[:,0,0]-8).clip(0)]
    stream = np.concatenate((band, np.abs(band), revision, np.abs(revision)), axis=2)
    def portfolio(conf):
        weight = (.1+conf)[:,:,None]
        return (geo['peers']@(stream*weight))/(geo['peers']@weight)
    delta = portfolio(bid)-portfolio(ask)
    stale = (1-.5*(bid+ask))[:,:,None]
    own = (bid-ask)[:,:,None]
    tilt = (geo['peers']@(bid-ask)[:,:,None])
    f = np.concatenate((delta, stale*delta, own*np.abs(band), own*np.abs(revision),
                        tilt*np.abs(band), own*tilt*band), axis=2)
    m = f.mean(1)
    return m, f-m[:,None]


def apply(p, mr, weights):
    m,r = mr
    return (p+(m@weights[0])[:,None]+r@weights[1]).astype('float32')


def run_validation(out, geo, make_features, control, candidate):
    out.mkdir(exist_ok=True)
    parent_geometry = cohort.geometry()
    prior, parent_stats, daily, folds = [], [], [], []
    stats = {control: [], candidate: []}
    for end in (8,10,12,14):
        blend_w = blend.fixed_weight(prior)
        parent_w = np.stack(fit(parent_stats)) if parent_stats else None
        if parent_w is not None:
            parent_w[:,:,:2] = 0.
        weights = {k:fit(v) for k,v in stats.items() if v}
        rows = {k:[] for k in stats}
        for day in (end, end+1):
            raw = book.rawday(day)
            p,y,expert,s = blend.training_day(day, prior)
            prior.append(s)
            p = blend.parent.fixed_forecast(p, expert, blend_w)
            z = cohort.features(raw, p, parent_geometry, False)
            parent_stats.append(moments(p[::4], y[::4], tuple(v[::4] for v in z)))
            if parent_w is not None:
                p = apply(p, z, parent_w)
            for k in stats:
                placebo = k == control
                z = make_features(raw, p, geo, placebo)
                stats[k].append(moments(p[::4], y[::4], tuple(v[::4] for v in z)))
                if k in weights:
                    q = apply(p,z,weights[k])
                    if day == end:
                        head = make_features(raw[:211],p[:211],geo,placebo)
                        np.testing.assert_allclose(q[:211],apply(p[:211],head,weights[k]),rtol=0,atol=2e-6)
                    row = tuple(v.reshape(-1,3) for v in (p,q,y))
                    rows[k].append(row)
                    if not placebo:
                        s = measure([row])
                        daily.append(np.subtract(s['candidate'],s['base']))
        if weights:
            folds.append({k:measure(v) for k,v in rows.items()})
            print(end+1,{k:v['gain'] for k,v in folds[-1].items()},flush=True)
    gain = np.stack([np.subtract(f[candidate]['candidate'],f[candidate]['base']) for f in folds])
    edge = np.stack([np.subtract(f[candidate]['candidate'],f[control]['candidate']) for f in folds])
    gate = (gain.min(0)>0) & (edge.mean(0)>0) & ((np.stack(daily)>0).sum(0)>=4)
    (out/'results.json').write_text(json.dumps(dict(folds=folds,daily=np.stack(daily).tolist(),retained=gate.tolist()),indent=2))
    if gate.any():
        w = np.stack(fit(stats[candidate]))*gate
        np.savez(out/'model.npz',**geo,weights=w)
        base = specialist.load()
        def predict(raw,p):
            q = base(raw,p)
            return apply(q,make_features(raw,q,geo,False),w)
        result = replay(predict)
        incumbent = json.loads((specialist.OUT/'raw_replay.json').read_text())['pooled']['score']
        result['gain_over_specialist'] = result['pooled']['score']-incumbent
        (out/'raw_replay.json').write_text(json.dumps(result,indent=2))
        print(result['pooled'],result['gain_over_specialist'],flush=True)

if __name__ == '__main__':
    run_validation(OUT, geometry(), features, 'mixed_sides', 'true_sides')
