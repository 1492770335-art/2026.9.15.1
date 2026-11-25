"""Rebuild the two horizon amplitudes without refitting on a verification date."""
import json
import sys

import numpy as np
import torch

import experiment_joint_interval_covariance_scale_v1 as incumbent
from y3_touch_clock_references import digest, read_stage
from y3_expanding_path_references import read_verified

previous = incumbent.history
trial,stack,study = previous.trial,previous.stack,previous.study
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
    report = json.loads((incumbent.OUT/'results.json').read_text())
    picks = report['selected']
    stats,errors = [],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        prev = previous.previous.references(d,raw)
        q = previous.references(d,raw)
        if d>=10:
            if d%2==0:
                w = np.column_stack([incumbent.fitted(stats,k)[picks[k]] for k in range(3)])
                folds = []
            y = trial.clock.labels(d).reshape(-1,50,3)
            grams = []
            for p,a in zip(q,prev):
                old = p.copy()
                p += (incumbent.basis(a,p)@w).astype('float32')
                v = np.concatenate((np.ones_like(p[:,:,:1]),old,p,y),axis=2).reshape(-1,10).astype(float)
                grams.append(v.T@v)
            folds.append(grams)
            if d%2:
                obs = []
                for m in np.sum(folds,axis=0):
                    c = stack.covariance(m)
                    n = np.sqrt(c.diagonal())
                    corr = c/n[:,None]/n[None,:]
                    obs.append(np.diag(corr[3:6,6:])-np.diag(corr[:3,6:]))
                expected = np.column_stack([np.array(r['folds'])[(d-11)//2,:,i] for r,i in zip(report['targets'],picks)])
                errors.append(float(np.max(abs(obs-expected))))
                assert errors[-1]<1.e-8
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(q))
        stats.append(incumbent.record(d))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    report = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    FULL.mkdir(exist_ok=True)
    trace = stack.verifier.trace
    rows,daily = [],{}
    with np.load(incumbent.OUT/'model.npz') as f:
        w = f['w']
    apply = incumbent.adjustment()
    def independent(a,p):
        delta = p[:,:,2].astype(float)-a[:,:,2].astype(float)
        m = delta.mean(1,keepdims=True)
        expected = p+(m[:,:,None]*w[0]+(delta-m)[:,:,None]*w[1]).astype('float32')
        q = apply(a,p)
        np.testing.assert_allclose(q,expected,rtol=0,atol=1.e-6)
        return q
    for d in range(16,20):
        raw = trace.replay.original.features(trace.replay.original.DATA,d)
        a,b,n = previous.previous.read(d,raw)
        p,h,_ = previous.read(d,raw)
        q,head = independent(a,p),independent(b,h)
        err = float(abs(q[:n]-head).max())
        assert err<1e-4
        row = (p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        rows.append(row)
        s = trace.older.measure([row])
        assert abs(s['score']-report['daily'][str(d+1)]['score'])<1.e-10
        daily[str(d+1)] = dict(score=s['score'],gain=s['gain'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    s = trace.older.measure(rows)
    assert s['rows']==936250
    np.testing.assert_allclose(s['candidate'],report['pooled']['candidate'],atol=1.e-10,rtol=0)
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=s,daily=daily),indent=2))
    print('verified',s['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
