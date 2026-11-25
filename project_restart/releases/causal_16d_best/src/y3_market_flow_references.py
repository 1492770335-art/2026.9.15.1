"""Reconstruct the frozen market-flow update on earlier dates for training later Y3 models."""
import json
import numpy as np
import torch

import experiment_y3_market_reversal_flow_v1 as incumbent
import y3_unfulfilled_references as previous

trial=incumbent.trial
CACHE=incumbent.OUT/'historical_references'


def prepare():
    saved=json.loads((incumbent.OUT/'results.json').read_text())
    pick=saved['selected']
    CACHE.mkdir(exist_ok=True)
    history=[];errors=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        z=incumbent.pools(raw,incumbent.prior.teacher())
        y=trial.clock.labels(d).reshape(-1,50,3)
        ref=[incumbent.prior.path.early.load(d,raw)] if d<8 else previous.references(d,raw)
        if d>=8 and d%2==0:
            frozen=incumbent.fit(history)
            folds=[[],[]]
        records=[];signals=[]
        for j,p in enumerate(ref):
            records.append(incumbent.record(z,p,y))
            if d>=8:
                folds[j].append(incumbent.stats(z,p,y,frozen))
                s=incumbent.changes(z,p,frozen)[pick]
                signals.append(s[:,0].copy())
                p[:,:,2]+=s
        history.append([np.mean(a,axis=0) for a in zip(*records)])
        if d>=8:
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=ref,signals=signals,
                cutoff=d-d%2,raw_hash=previous.rolling.cached.digest(raw),
                forecast_hash=previous.rolling.cached.digest(ref))
            if d%2:
                observed=np.array([[trial.scoring.audit.pooled(v) for v in np.sum(f,axis=0)] for f in folds])
                expected=np.array(saved['folds'][(d-9)//2])
                errors.append(float(abs(observed-expected).max()))
                assert errors[-1]<1e-9
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def references(d,raw):
    return previous.rolling.read_verified(CACHE,d,raw)


def outer(module):
    older=previous.previous
    stages=(previous.rolling.incumbent,older.parent.previous.incumbent,older.parent.incumbent,
            older.incumbent,previous.incumbent,incumbent)
    updates=[s.correction() for s in stages]
    def load(day,raw):
        p,h,n=previous.rolling.cached.load(day,raw)
        for update in updates:
            p[:,:,2]+=update(raw,p)
            h[:,:,2]+=update(raw[:n],h)
        return p,h,n
    previous.rolling.cached.scoring.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
