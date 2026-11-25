"""Verify the selected midpoint covariance readouts and reconstruct chronological forecasts."""
import json
import sys

import numpy as np
import torch

import experiment_y3_interval_covariance_bridge_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage,verify_stage

previous = incumbent.history
trial,stack,study = previous.trial,previous.stack,previous.study
CACHE = incumbent.OUT/'historical_references'
FULL = incumbent.OUT/'full_chain_verification'


def references(d,raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    return read_stage(FULL,d,raw)


def replay_with():
    return stack.verifier,read


def verify():
    with np.load(incumbent.OUT/'model.npz') as f:
        readout,w = f['read'],f['w']
        assert int(f['pick'])==5 and np.array_equal(np.flatnonzero(w[3:]),[4,5])
    def independent(raw,p):
        q = p.copy()
        for t in np.array_split(np.arange(len(p)),max(1,(len(p)+699)//700)):
            x = incumbent.book.observations(raw,t)[0].astype(float)
            z = trial.clock.center(x@readout[:,1:3].astype(float))
            q[t,:,2] += p[t]@w[:3]+z@w[7:9]
        return q
    verify_stage(incumbent,previous.read,independent)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = report['selected']
    c = incumbent.book.model.observed.historical_geometry()[1][:,1]
    rows,errors = [],[]
    for day in range(4,16):
        raw = trial.clock.book.rawday(day)
        y = trial.clock.labels(day).reshape(-1,50,3)
        refs = previous.references(day,raw) if day>=8 else [study.screen.history.market.prior.path.early.load(day,raw)]
        if day%2==0:
            st = incumbent.params(day)
            if day>=8:
                w = incumbent.book.model.weights(rows,incumbent.GROUPS)[pick]
                group = []
        t = np.arange(day%7,len(y)-60,7)
        t = t[(t+60<2401)|(t>=2401)]
        at = np.arange(len(y)) if day>=8 else t
        allz = np.concatenate([incumbent.advice(raw,chunk,st) for chunk in np.array_split(at,max(1,(len(at)+499)//500))])
        z = allz[t] if day>=8 else allz
        rows.append([incumbent.book.model.stats(p,y,t,z) for p in refs])
        if day<8:
            continue
        legs = np.stack([y[t+h,:,1]*c[2]/c[1] for h in (0,20,40)],axis=2)
        gs = []
        for p in refs:
            old = p[t,:,2].copy()
            p[:,:,2] += p@w[:3]+allz@w[3:]
            a = np.concatenate((np.ones_like(old[:,:,None]),old[:,:,None],p[t,:,2:3],y[t,:,2:3],legs),axis=2)
            a = a.reshape(-1,7).astype(float)
            gs.append(a.T@a)
        group.append(gs)
        if day%2:
            observed = []
            for g in np.sum(group,axis=0):
                cov = stack.covariance(g)
                corr = cov[:2,2:]/np.sqrt((cov.diagonal()[:2]*cov[2,2])[:,None])
                observed.append(corr[1]-corr[0])
            expected = np.array(report['folds'])[(day-9)//2,:,pick]
            errors.append(float(np.max(abs(observed-expected))))
            assert errors[-1]<1.e-8
        np.savez(CACHE/f'day_{day:02d}.npz',forecasts=refs,cutoff=day-day%2,raw_hash=digest(raw),forecast_hash=digest(refs))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(incumbent.book.model.weights(rows,incumbent.GROUPS)[pick],f['w'],atol=1e-7,rtol=0)
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
