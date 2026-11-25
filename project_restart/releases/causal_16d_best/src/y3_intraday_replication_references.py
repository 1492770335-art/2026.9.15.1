"""Independently replay the accepted tracking adjustment from past returns, and reconstruct its training references."""
import json
import sys

import numpy as np

import experiment_y3_intraday_replication_v1 as incumbent
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


def check_weights(w):
    np.testing.assert_allclose(w.sum(1),1.,atol=1e-9,rtol=0)
    assert np.all(w>=0) and not np.any(w.diagonal())


def independent(raw,p,prior,weights):
    q = np.zeros(p.shape[:2],np.float32)
    check_weights(prior)
    for lo,hi in base.segments(len(raw)):
        r = np.clip(raw[lo:hi,:,0].astype(float)-.00136,-15,15)
        for t in range(360,len(r),60):
            sl = slice(lo+t,min(lo+t+60,hi))
            past,check = r[max(0,t-864):t-144],r[t-144:t-48]
            def blocks(v,h):
                count = len(v)//h
                return np.array([row.sum(0) for row in np.split(v[-count*h:],count)])/np.sqrt(h)
            fit = [blocks(past,h) for h in (8,24)]
            gram = sum(z.T@z/len(z) for z in fit)/2
            w = incumbent.solve(gram,prior,1.8*np.mean(gram.diagonal()).clip(.02))
            check_weights(w)
            gains = []
            for h in (8,24):
                z = blocks(check,h)
                r0 = np.square(z-z@prior.T).mean(0)
                r1 = np.square(z-z@w.T).mean(0)
                gains.append(((r0-r1)/(r0+.02)).clip(-.5,.5))
            confidence = np.sqrt(np.maximum(np.mean(gains,0),0))
            for local in range(t,sl.stop-lo):
                ix = lo+local
                bank = [r[max(0,local-width+1):local+1].sum(0)/np.sqrt(width) for width in (8,40,100)]
                vals = np.column_stack((*bank,p[ix,:,2],p[ix,:,2]-base.RATIO*p[ix,:,1]))
                direct = (prior-w)@vals
                calibrated = (direct-direct.mean(0))*confidence[:,None]
                q[ix] = (calibrated-calibrated.mean(0))@weights
    return q


def verify():
    FULL.mkdir(exist_ok=True)
    expected = json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']
    with np.load(incumbent.OUT/'model.npz') as f:
        assert incumbent.ARMS[int(f['pick'])+1]=='validated price and forecast change'
        w,prior = f['w'],f['initial']
    loader = incumbent.replay_with()[1]
    rows,daily = [],{}
    for d in range(16,20):
        raw = trial.clock.book.rawday(d)
        p,_,_ = previous.read(d,raw)
        q,h,n = loader(d,raw)
        recon = p[:,:,2]+independent(raw,p,prior,w)
        err = float(abs(q[:,:,2]-recon).max())
        assert err<2e-6 and abs(q[:n]-h).max()<1e-4 and np.array_equal(p[:,:,:2],q[:,:,:2])
        row = p.reshape(-1,3),q.reshape(-1,3),stack.verifier.trace.replay.labels(d)
        rows.append(row)
        daily[d+1] = dict(independent_error=err,score=stack.verifier.trace.older.measure([row])['score'])
        np.savez(FULL/f'day_{d}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled = stack.verifier.trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-expected['score'])<1e-12
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))


def rebuild():
    CACHE.mkdir(exist_ok=True)
    rows,checks = [],{}
    initial = incumbent.prior()
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = report['selected']-1
    for d in range(4,16):
        if d in (8,12):
            w = incumbent.fit(rows)[pick]
        raw = trial.clock.book.rawday(d)
        ps = previous.references(d,raw) if d>=8 else incumbent.parent.parent.training_references(d,raw)
        qs = ps.copy()
        y = trial.clock.labels(d).reshape(-1,50,3)
        banks = incumbent.stages(raw,initial)
        moments,gains = [],[]
        for p,q in zip(ps,qs):
            x = incumbent.design(p,banks,initial)
            ix = slice(d%5,None,5)
            moments.append(incumbent.observe(x[ix],p[ix],y[ix]))
            if d>=8:
                q[:,:,2] += (x[:,:,pick]@w).astype('float32')
                gains.append(np.corrcoef(q[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-
                             np.corrcoef(p[:,:,2].ravel(),y[:,:,2].ravel())[0,1])
        rows.append(tuple(np.mean(v,0) for v in zip(*moments)))
        if d>=8:
            np.testing.assert_allclose(gains,np.asarray(report['daily'])[d-8,:,pick+1],atol=1e-8,rtol=0)
            checks[d+1] = gains
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=qs,cutoff=d-d%2,calibration_cutoff=d-d%4,
                     raw_hash=digest(raw),forecast_hash=digest(qs))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_array_equal(initial,f['initial'])
        np.testing.assert_allclose(incumbent.fit(rows)[pick],f['w'],atol=1e-9,rtol=0)
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=checks),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as release
        release.run(incumbent,read,previous.incumbent,previous.read)
        release.run(incumbent,read)
