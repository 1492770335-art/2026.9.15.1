"""Verify the selected opposite-stock slope change and reconstruct expanding training references."""
import json
import sys

import numpy as np
import torch

import experiment_y3_dynamic_horizon_geometry_v1 as incumbent
from y3_touch_clock_references import digest,read_stage,verify_stage
from y3_expanding_path_references import read_verified

previous = incumbent.history
trial,stack,study = previous.trial,previous.stack,previous.study
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def read(d,raw):
    return read_stage(FULL,d,raw)


def references(d,raw):
    return read_verified(CACHE,d,raw)


def replay_with():
    return stack.verifier,read


def verify():
    with np.load(incumbent.OUT/'model.npz') as f:
        w = f['w']
    assert np.array_equal(np.flatnonzero(w[1:]),[6,7])
    def independent(raw,p):
        rel = p.astype(float)-p.astype(float).mean(1,keepdims=True)
        change = w[0]*p[:,:,2].astype(float)
        rho = np.exp(-1/96)
        for side in (0,1):
            u = rel[:,side::2]
            g = (u*u[:,:,2:]).mean(1)
            avg = np.empty_like(g)
            for t in range(len(g)):
                avg[t] = g[t] if t in (0,2401) else rho*avg[t-1]+(1-rho)*g[t]
            instant = .75*g+.25*avg
            diff = avg[:,:2]/(avg[:,2:]+.003)-instant[:,:2]/(instant[:,2:]+.003)
            change[:,1-side::2] += rel[:,1-side::2,2]*(diff@w[7:9])[:,None]
        q = p.copy()
        q[:,:,2] += change
        return q
    verify_stage(incumbent,previous.read,independent)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = report['selected']
    rows,errors = [],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        refs = previous.references(d,raw)
        if d>=10:
            if d%2==0:
                w = incumbent.weights(rows)[pick]
                group = []
            y = trial.clock.labels(d).reshape(-1,50,3)[:,:,2]
            mats = []
            for p in refs:
                old = p[:,:,2].copy()
                p[:,:,2] += w[0]*old+incumbent.design(p)@w[1:]
                a = np.column_stack((np.ones(y.size),old.ravel(),p[:,:,2].ravel(),y.ravel()))
                mats.append(a.T@a)
            group.append(mats)
            if d%2:
                measured = []
                for m in np.sum(group,axis=0):
                    c = stack.covariance(m)
                    ic = c[:2,2]/np.sqrt(c.diagonal()[:2]*c[2,2])
                    measured.append(ic[1]-ic[0])
                expected = np.array(report['folds'])[(d-11)//2,:,pick]
                err = float(np.max(np.abs(expected-measured)))
                assert err<1e-8
                errors.append(err)
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=refs,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(refs))
        rows.append(incumbent.records(d))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(incumbent.weights(rows)[pick],f['w'],rtol=0,atol=1.e-8)
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_early_book_horizon_gain_v1 as horizons
        import audit_y3_release_gain_v1 as gains
        gains.run(incumbent,read,previous.incumbent,previous.read)
        gains.run(incumbent,read)
        horizons.run(sys.modules[__name__],previous)
    else:
        rebuild()
