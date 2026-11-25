"""Verify the directional update and regenerate its strictly preceding training-window references."""
import json
import sys

import numpy as np

import experiment_y3_angular_forecast_history_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage

previous = incumbent.history
base,trial,study,stack = previous.base,previous.trial,previous.study,previous.stack
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def references(day,raw):
    return read_verified(CACHE,day,raw)


def read(day,raw):
    return read_stage(FULL,day,raw)


def independently_rotate(p,w):
    result = np.zeros(p.shape[:2],np.float32)
    def norm(x):
        return np.sqrt(x@x)
    def log(u,v):
        c = np.clip(u@v,-.999999,1.)
        a = np.arccos(c)
        return (v-c*u)*np.clip(a/max(np.sin(a),1e-6),.99,4.)
    for lo,hi in base.segments(len(p)):
        directions = []
        for t in range(lo,hi):
            near,full = p[t,:,1:].astype(float).T
            near -= near.mean()
            near /= max(norm(near),1e-8)
            full -= full.mean()
            b = full-near*(near@full)
            radius = norm(b)
            u = b/max(radius,1e-8)
            directions.append(u)
            total = np.zeros(50)
            for j,lag in enumerate(incumbent.LAGS):
                if t-lo<2*lag:
                    continue
                v,s = directions[-1-lag],directions[-1-2*lag]
                vel = -log(u,v)
                old = -log(v,s)
                old -= ((old@u)/max(1+v@u,.1))*(v+u)
                total += w[j]*vel+w[j+8]*(vel-old)
            total -= total.mean()
            total -= near*(total@near)
            total -= u*(total@u)
            total *= radius
            new = b+min(.32,.16*radius/max(norm(total),1e-9))*total
            result[t] = radius*new/max(norm(new),1e-8)-b
    return result


def verify():
    FULL.mkdir(exist_ok=True)
    expected = json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']
    with np.load(incumbent.OUT/'model.npz') as f:
        w = f['w']
    assert not w[4:8].any() and not w[12:].any()
    loader = incumbent.replay_with()[1]
    rows,daily = [],{}
    for d in range(16,20):
        raw = trial.clock.book.rawday(d)
        p,_,_ = previous.read(d,raw)
        q,head,n = loader(d,raw)
        delta = independently_rotate(p,w)
        err = float(abs((p[:,:,2]+delta)-q[:,:,2]).max())
        assert err<2e-6 and np.array_equal(p[:,:,:2],q[:,:,:2]) and abs(head-q[:n]).max()<1e-4
        np.testing.assert_allclose(q[:,:,2].mean(1),p[:,:,2].mean(1),atol=4e-7,rtol=0)
        np.testing.assert_allclose(q[:,:,2].std(1),p[:,:,2].std(1),atol=4e-7,rtol=0)
        row = p.reshape(-1,3),q.reshape(-1,3),stack.verifier.trace.replay.labels(d)
        rows.append(row)
        daily[d+1] = dict(independent_error=err,score=stack.verifier.trace.older.measure([row])['score'])
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    pooled = stack.verifier.trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-expected['score'])<1e-12
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = report['selected']-1
    training,checks = [],{}
    for d in range(4,16):
        if d in (8,12):
            w = incumbent.fit(training)[:,pick:pick+1]
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        ps = incumbent.training_references(d,raw)
        qs = ps.copy()
        t = np.concatenate([np.arange(lo+d%5,hi-60,5) for lo,hi in base.segments(len(raw))])
        moments,gains = [],[]
        for p,q in zip(ps,qs):
            x,far,length = incumbent.design(p)
            moments.append(incumbent.observe(x[t],y[t]))
            if d>=8:
                q[:,:,2] += incumbent.apply(x,far,length,w)[:,:,0]
                gains.append(np.corrcoef(q[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-
                             np.corrcoef(p[:,:,2].ravel(),y[:,:,2].ravel())[0,1])
        training.append(tuple(np.mean(col,0) for col in zip(*moments)))
        if d>=8:
            np.testing.assert_allclose(gains,np.asarray(report['daily'])[d-8,:,pick+1],atol=1e-8,rtol=0)
            checks[d+1] = gains
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=qs,cutoff=d-d%2,calibration_cutoff=d-d%4,
                     raw_hash=digest(raw),forecast_hash=digest(qs))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(incumbent.fit(training)[:,pick],f['w'],atol=1e-10,rtol=0)
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=checks),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as release
        release.run(incumbent,read,previous.incumbent,previous.read)
        release.run(incumbent,read)
