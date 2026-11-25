"""Rebuild the current three-target reference using only preceding dates for each training-period fold."""
import numpy as np

import experiment_joint_sort_horizon_transfer_v1 as joint

rotation, portfolios, clock, parent = joint.rotation, joint.portfolios, joint.clock, joint.parent


def historical_panels():
    geometry = rotation.geometry()
    exposure, market, sorts, medium, settled = [], [], [], [], []
    for day in range(4, 16):
        raw, refs, y = parent.history.training_day(day)
        refs = np.array(refs, copy=True)
        exp_records = [parent.exposure.observations(p, y) for p in refs]
        ps = list(refs)
        if day >= 8:
            if day % 2 == 0:
                ew = parent.exposure.fit(exposure)[6]
                cw, pw = clock.params(market), portfolios.fit(sorts)
                mw, rw = portfolios.fit(medium), rotation.fit(settled)
            refs[:, :, :, 2] += parent.apply_market(*parent.historical_heads(day), ew)
            strict = np.load(parent.ART/'cutoff_reference_panel_v1'/f'panel_{day}.npy')
            ps = []
            for p in [*refs, strict]:
                q = p.copy()
                q[:, :, 2] += (clock.signals(raw, p, cw)[:, :, clock.CHOSEN-1]
                               + portfolios.corrections(raw, p, pw)[:, :, 2]
                               + rotation.signals(raw, p, rw, geometry)[:, :, 1])
                q[:, :, 1] += joint.deltas(raw, p, mw, pw)[:, :, 2]
                ps.append(q)
        exposure.append(tuple(np.mean(v, 0) for v in zip(*exp_records)))
        for dest, fn in ((market, clock.record), (sorts, lambda r,p,y:portfolios.record(r,p,y,day)),
                         (medium, lambda r,p,y:portfolios.record(r,joint.medium(p),joint.medium(y),day)),
                         (settled, lambda r,p,y:rotation.record(r,p,y,geometry,day))):
            rows = [fn(raw,p,y) for p in refs]
            dest.append([tuple(np.mean(v, 0) for v in zip(*s)) for s in zip(*rows)])
        yield day, raw, ps, y


def scores(rows, arms):
    return np.stack([parent.base.score(m,np.ones(arms-1)) for m in np.sum(rows,0)])


def replay(out, load_update, cached_upstream=None, load_final_y2=None, load_final_y1=None, load_post_y2=None):
    """Score a final Y3 head, with fresh upstream and reload for full verification."""
    import json
    from pathlib import Path
    from experiment_causal_cross_horizon_ridge_v1 import labels, measure
    import replay_session_experts_16d as session
    cache = Path(cached_upstream) if cached_upstream else None
    reference = 'packaged_inference.json' if cache else 'reload_verification.json'
    expected = json.loads((joint.OUT / reference).read_text())
    def run(stage):
        medium_update, long_update = joint.correction(), rotation.correction()
        apply = load_update()
        finish_medium = load_final_y2() if load_final_y2 is not None else None
        finish_short = load_final_y1() if load_final_y1 is not None else None
        post_y2 = load_post_y2() if load_post_y2 is not None else None
        if cache is None:
            verifier, upstream = parent.replay_with()
        rows, daily = [], {}
        for d, n in session.PREFIX.items():
            print(f"[{stage}] day {d+1} started", flush=True)
            raw = session.features(session.DATA,d)
            if cache:
                head = np.load(cache/f'prefix_{d}.npy')
                whole = np.load(cache/f'full_{d}.npy')
            else:
                whole,head,cut = upstream(d,raw)
                assert cut == n
            def forecast(r,initial):
                p = initial.copy()
                p[:,:,1] += medium_update(r,initial)
                p[:,:,2] += long_update(r,initial)
                q = p.copy()
                q[:,:,2] += apply(r,p)
                assert np.array_equal(p[:,:,:2],q[:,:,:2]) and np.isfinite(q).all()
                issued = q.copy() if post_y2 is not None else None
                if finish_medium is not None:
                    q[:, :, 1] += finish_medium(r, q)
                    assert np.isfinite(q[:, :, 1]).all()
                if finish_short is not None:
                    q[:, :, 0] += finish_short(r, q)
                    assert np.isfinite(q[:, :, 0]).all()
                if post_y2 is not None:
                    q[:, :, 1] += post_y2(issued, q)
                    assert np.isfinite(q[:, :, 1]).all()
                return p,q
            p,q = forecast(raw,whole)
            _,short = forecast(raw[:n],head)
            err = float(abs(short-q[:n]).max())
            assert err < 1e-4
            row=(p.reshape(-1,3),q.reshape(-1,3),labels(d))
            m=measure([row])
            assert abs(m['baseline_score']-expected['daily'][str(d+1)]['score']) < 1e-9
            daily[d+1]=dict(score=m['score'],gain=m['gain'],corr=m['candidate'],prefix_error=err)
            rows.append(row)
            print(f"[{stage}] day {d+1} score {m['score']:.8f}, prefix error {err:.2g}", flush=True)
        result=measure(rows)
        assert result['rows']==936250
        return dict(parent=joint.OUT.name,pooled=result,daily=daily)
    name='cached_validation' if cache else 'raw_replay'
    result=run(name)
    (out/(name+'.json')).write_text(json.dumps(result,indent=2))
    if not cache:
        assert result==run('reload_verification')
        (out/'reload_verification.json').write_text(json.dumps(result,indent=2))
    print(name,result['pooled'],flush=True)
