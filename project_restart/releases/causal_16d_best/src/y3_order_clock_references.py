"""Chronological and held-out inputs for calibrating the ordered pressure model."""
import json
import os
import numpy as np
import torch
import experiment_y3_independent_order_clocks_v1 as incumbent
from y3_expanding_path_references import read_verified

previous,study,trial=incumbent.history,incumbent.study,incumbent.trial
stack=previous.stack
CACHE=incumbent.OUT/'historical_references'
digest=previous.previous.digest


def references(d,raw):
    return read_verified(CACHE,d,raw)


def rebuild():
    report=json.loads((incumbent.OUT/'results.json').read_text())
    pick=report['selected']
    assert pick is not None
    past=[]
    errors=[]
    CACHE.mkdir(exist_ok=True)
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        t=np.arange(90+d%21,len(raw)-60,21)
        ev=np.arange(d%4,len(raw)-60,4)
        refs=previous.references(d,raw) if d>=8 else [study.screen.history.market.prior.path.early.load(d,raw)]
        design=incumbent.historical_design(d)
        if d>=8 and not d%2:
            st=incumbent.fit(past)
            fold=[[],[]]
        records=[]
        for i,p in enumerate(refs):
            x=design(raw,p,np.arange(len(raw)) if d>=8 else t)
            records.append((x[t] if d>=8 else x,study.error(p,y,t)))
            if d>=8:
                delta=incumbent.changes(x,st)[pick]
                fold[i].append(stack.verifier.audit.blockstats(p[ev],y[ev],[delta[ev]]).sum(0))
                p[:,:,2]+=delta
        past.append(tuple(np.concatenate(v) for v in zip(*records)))
        if d>=8:
            path=CACHE/f'day_{d:02d}.npz'
            tmp=path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,forecasts=refs,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(refs))
            tmp.replace(path)
            if d%2:
                scores=[stack.verifier.audit.pooled(sum(v))[0] for v in fold]
                error=float(abs(np.array(scores)-np.array(report['folds'][(d-9)//2])[:,0,pick]).max())
                assert error<1e-8,error
                errors.append(error)
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors,days=list(range(9,17))),indent=2))
    print('verified chronological references',errors,flush=True)


def replay_with():
    ver,load=previous.replay_with()
    update=incumbent.correction()
    def read(day,raw):
        p,h,n=load(day,raw)
        p[:,:,2]+=update(raw,p)
        h[:,:,2]+=update(raw[:n],h)
        return p,h,n
    return ver,read


def outer(module):
    verifier,load=replay_with()
    verifier.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    rebuild()
