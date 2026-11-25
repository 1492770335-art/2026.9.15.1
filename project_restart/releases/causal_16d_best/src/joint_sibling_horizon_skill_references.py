"""Reproduce both target-specific readouts independently and reconstruct their chronological forecasts."""
import json
import sys
import numpy as np
import torch
import experiment_joint_sibling_horizon_skill_v1 as incumbent
from y3_touch_clock_references import digest,read_stage
from y3_expanding_path_references import read_verified

previous = incumbent.history
trial,stack,study = previous.trial,previous.stack,previous.study
inc,skill = incumbent.increments,incumbent.skill
CACHE = incumbent.OUT/'historical_references'
FULL = incumbent.OUT/'full_chain_verification'


def read(d,raw):
    return read_stage(FULL,d,raw)


def references(d,raw):
    return read_verified(CACHE,d,raw)


def replay_with():
    return stack.verifier,read


def rebuild():
    CACHE.mkdir(exist_ok=True)
    sel = json.loads((incumbent.OUT/'results.json').read_text())['selected']
    srep = json.loads((inc.OUT/'results.json').read_text())
    lrep = json.loads((skill.OUT/'results.json').read_text())
    hist,stats,errors = [],[],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        preds = previous.references(d,raw)
        new_stats = tuple(np.mean(v,axis=0) for v in zip(*[skill.record(p,y) for p in preds]))
        if d>=10:
            if d%2==0:
                w = np.stack([inc.study.weights(inc,inc.target_rows(hist,k))[sel[k]] for k in range(2)])
                st,group = skill.fit(stats),[]
            day = []
            for i,p in enumerate(preds):
                q = incumbent.update(raw,p,w,st,sel[2])
                v = np.concatenate((np.ones_like(p[:,:,:1]),p,q,y),axis=2).reshape(-1,10).astype(float)
                day.append(v.T@v)
                preds[i] = q
            group.append(day)
            if d%2:
                obs = []
                for m in np.sum(group,axis=0):
                    c = stack.covariance(m)
                    sd = np.sqrt(c.diagonal())
                    cor = c/sd[:,None]/sd[None,:]
                    obs.append(np.diag(cor[3:6,6:])-np.diag(cor[:3,6:]))
                fold = (d-11)//2
                expected = np.column_stack([np.array(srep['targets'][k]['folds'])[fold,:,sel[k]] for k in range(2)]+[
                                           np.array(lrep['folds'])[fold,:,sel[2]]])
                errors.append(float(np.max(abs(expected-obs))))
                assert errors[-1]<1e-8
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=preds,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(preds))
        hist.append(inc.record(d))
        stats.append(new_stats)
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def independent(p,st):
    assert int(st['pick'])==4
    w = st['w']
    assert not w[1].any() and set(np.flatnonzero(w[0,1:])).issubset({3,4,5})
    z = trial.clock.center(p.astype(float))
    scale = inc.observed.historical_geometry()[1][:,1]
    ratios = scale[2]/scale
    cash = z*ratios
    legs = np.concatenate((cash[:,:,:1],np.diff(cash,axis=2)),axis=2)/ratios
    first = w[0,0]*p[:,:,0].astype(float)
    third = np.zeros_like(first)
    pairs = np.triu_indices(3,1)
    for side in (0,1):
        peer = legs[:,side::2]
        v = (peer*peer).mean(1)
        slow = inc.study.ema(v,96)
        change = np.sqrt((slow+.002)/(.75*v+.25*slow+.002))-1
        first[:,1-side::2] += (legs[:,1-side::2]*change[:,None,:])@w[0,4:7]
        peer = z[:,side::2]
        gram = np.einsum('nsi,nsj->nij',peer,peer)/25
        slow = inc.study.ema(gram,128)
        fast = .8*inc.study.ema(gram,6)+.2*slow
        a,b = [np.sqrt(m.diagonal(axis1=1,axis2=2)+.003) for m in (fast,slow)]
        corr = fast/a[:,None,:]/a[:,:,None]-slow/b[:,None,:]/b[:,:,None]
        state = np.column_stack((np.ones(len(p)),np.log(a/b),corr[:,pairs[0],pairs[1]]))
        risk = st['risk']
        coef = np.linalg.solve(risk+.8*np.diag(risk.diagonal().clip(.001)),(state@st['all']).T).T
        amp = np.sqrt((coef*(coef@risk)).sum(1)).clip(1e-8)
        coef *= (.55*np.minimum(1.,.18*np.sqrt(risk[2,2])/amp))[:,None]
        third[:,1-side::2] = (z[:,1-side::2]*coef[:,None,:]).sum(2)
    q = p.copy()
    q[:,:,0] += first.astype('float32')
    q[:,:,2] += third.astype('float32')
    return q


def verify():
    FULL.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    trace = stack.verifier.trace
    apply = incumbent.adjustment()
    with np.load(incumbent.OUT/'model.npz') as f:
        st = dict(f)
    rows,daily = [],{}
    for d in range(16,20):
        raw = trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n = previous.read(d,raw)
        q,part = apply(raw,p),apply(raw[:n],h)
        for orig,dest in ((p,q),(h,part)):
            np.testing.assert_allclose(dest,independent(orig,st),rtol=0,atol=1e-6)
        err = float(np.max(abs(q[:n]-part)))
        assert err<1.e-4
        row = (p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        rows.append(row)
        s = trace.older.measure([row])
        assert abs(s['score']-report['daily'][str(d+1)]['score'])<1.e-10
        daily[d+1] = dict(score=s['score'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=part,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(part))
    pooled = trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-report['pooled']['score'])<1e-10
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))
    print(pooled,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_early_book_horizon_gain_v1 as horizon
        import audit_y3_release_gain_v1 as gain
        gain.run(incumbent,read,previous.previous.incumbent,previous.previous.read)
        gain.run(incumbent,read)
        horizon.run(sys.modules[__name__],previous.previous)
    else:
        rebuild()
