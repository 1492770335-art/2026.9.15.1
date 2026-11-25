"""Rebuild time-ordered forecasts of the three-head readout for subsequent experiments."""
import json
import os
import numpy as np
import torch

import experiment_joint_horizon_stage_transfer_v1 as incumbent
from y3_expanding_path_references import read_verified

stack,trial,previous=incumbent.stack,incumbent.trial,incumbent.history
study=previous.study
CACHE=incumbent.OUT/'historical_references'
digest=stack.verifier.trace.replay.digest


def references(day,raw):
    return read_verified(CACHE,day,raw)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report=json.loads((incumbent.OUT/'results.json').read_text())
    picks=report['selected']
    history=[]
    fold=[]
    errors=[]
    for day in range(8,16):
        raw=trial.clock.book.rawday(day)
        p,z=stack.panels(day,raw)
        if day>=10 and day%2==0:
            weights=np.array([incumbent.fit(history,k)[picks[k]] for k in range(3)])
            fold=[]
        if day>=10:
            forecasts=np.stack([incumbent.apply_readout(v,a,weights) for v,a in zip(p,z)])
            target=trial.clock.labels(day).reshape(-1,50,3)
            scores=[]
            for old,new in zip(p,forecasts):
                a=np.column_stack((np.ones(target.size//3),old.reshape(-1,3),new.reshape(-1,3),target.reshape(-1,3))).astype(float)
                scores.append(a.T@a)
            fold.append(scores)
            if day%2:
                observed=[]
                for g in np.sum(fold,0):
                    c=stack.covariance(g)
                    denom=np.sqrt(c.diagonal().clip(1e-9))
                    corr=c/(denom[:,None]*denom[None,:])
                    observed.append(np.diag(corr[3:6,6:])-np.diag(corr[:3,6:]))
                expected=np.column_stack([np.array(v['folds'])[(day-11)//2,:,j] for v,j in zip(report['targets'],picks)])
                err=float(abs(expected-np.array(observed)).max())
                errors.append(err)
                assert err<1e-8,err
        else:
            # No earlier chronological examples of the full stage library exist for these two dates.
            forecasts=p
        path=CACHE/f'day_{day:02d}.npz'
        tmp=path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(tmp,forecasts=forecasts,cutoff=day-day%2,raw_hash=digest(raw),forecast_hash=digest(forecasts))
        tmp.replace(path)
        history.append(incumbent.moments(day))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors,comparison_days=list(range(11,17)),
                                                       unadjusted_warmup_days=[9,10]),indent=2))
    print('reconstructed chronological scores',errors,flush=True)


def replay_with(extra=()):
    trace=stack.verifier.trace
    folder=incumbent.OUT/'full_chain_verification'
    full_predict=None
    updates=[m.correction() for m in extra]
    def load(day,raw):
        nonlocal full_predict
        path=folder/f'day_{day}.npz'
        if (folder/'composed_replay.json').is_file() and path.is_file():
            with np.load(path) as f:
                q,h=f['full'],f['head']
                assert str(f['raw_hash'])==digest(raw) and str(f['pred_hash'])==digest(q)
        else:
            if full_predict is None:
                full_predict=incumbent.layer()
            with np.load(trace.replay.CACHE/f'day_{day}.npz') as f:
                assert str(f['raw_sha256'])==digest(raw)
                n=int(f['prefix_ticks'])
                q=full_predict(raw,f['full'])
                h=full_predict(raw[:n],f['head'])
        n=len(h)
        for update in updates:
            q[:,:,2]+=update(raw,q)
            h[:,:,2]+=update(raw[:n],h)
        return q,h,n
    return stack.verifier,load


def outer(module):
    evaluator,loader=replay_with()
    evaluator.outer(module,reference=incumbent,load_reference=loader)


if __name__=='__main__':
    torch.set_num_threads(2)
    rebuild()
