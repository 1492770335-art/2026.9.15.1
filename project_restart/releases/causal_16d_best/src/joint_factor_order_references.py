"""Rebuild the chronological three-target factor-order forecasts and verify serialized readouts."""
import json
import sys

import numpy as np
import torch

import experiment_joint_factor_order_calibration_v1 as incumbent
from y3_expanding_path_references import read_verified

previous = incumbent.history
stack, trial, study = previous.stack, previous.trial, previous.study
CACHE = incumbent.OUT/'historical_references'
FULL = incumbent.OUT/'full_chain_verification'
from y3_touch_clock_references import digest, read_stage


def references(d,raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    return read_stage(FULL,d,raw)


def replay_with():
    return stack.verifier,read


def outer(module):
    stack.verifier.outer(module,reference=incumbent,load_reference=read)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    picks = report['selected']
    stats, errors = [], []
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        q = previous.references(d,raw)
        if d>=10:
            if d%2==0:
                weights = np.column_stack([incumbent.fitted(stats,k)[picks[k]] for k in range(3)])
                folds = []
            old = previous.previous.references(d,raw)
            x = incumbent.signals(raw, incumbent.parent.params(d-d%2))
            y = trial.clock.labels(d).reshape(-1,50,3)
            grams = []
            for p,current in zip(old,q):
                before = current.copy()
                current += (incumbent.basis(p,current,x)@weights).astype('float32')
                a = np.concatenate((np.ones_like(y[:,:,:1]),before,current,y),2).reshape(-1,10).astype(float)
                grams.append(a.T@a)
            folds.append(grams)
            if d%2:
                gain = []
                for g in np.sum(folds,0):
                    c = stack.covariance(g)
                    norm = np.sqrt(c.diagonal())
                    corr = c/(norm[None,:]*norm[:,None])
                    gain.append(np.diag(corr[3:6,6:])-np.diag(corr[:3,6:]))
                want = np.column_stack([np.array(r['folds'])[(d-11)//2,:,j] for r,j in zip(report['targets'],picks)])
                errors.append(float(np.max(abs(gain-want))))
                assert errors[-1]<1e-8
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(q))
        stats.append(incumbent.record(d))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    FULL.mkdir(exist_ok=True)
    trace = stack.verifier.trace
    expect = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as archive:
        w = archive['w']
    with np.load(incumbent.parent.OUT/'model.npz') as archive:
        st = dict(archive)
    run = incumbent.adjustment()
    rows, daily = [], {}
    def independent(raw,old,p):
        x = incumbent.signals(raw,st)
        z = incumbent.basis(old,p,x)
        out = p+np.einsum('tsi,ij->tsj',z.astype(float),w).astype('float32')
        q = run(raw,old,p)
        np.testing.assert_allclose(q,out,rtol=0,atol=1e-6)
        return q
    for d in range(16,20):
        raw = trace.replay.original.features(trace.replay.original.DATA,d)
        old,part,_ = previous.previous.read(d,raw)
        p,h,n = previous.read(d,raw)
        q,head = independent(raw,old,p),independent(raw[:n],part,h)
        err = float(abs(q[:n]-head).max())
        assert err<1e-4 and np.isfinite(q).all()
        row = (p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        rows.append(row)
        s = trace.older.measure([row])
        assert abs(s['score']-expect['daily'][str(d+1)]['score'])<1e-10
        daily[str(d+1)] = dict(score=s['score'],gain=s['gain'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    s = trace.older.measure(rows)
    assert s['rows']==936250
    np.testing.assert_allclose(s['candidate'],expect['pooled']['candidate'],rtol=0,atol=1e-10)
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=s,daily=daily),indent=2))
    print('verified',s['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
