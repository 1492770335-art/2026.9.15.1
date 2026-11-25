"""Rebuild chronological inputs for experiments following the early book readout."""
import json
import os
import numpy as np
import torch
import experiment_joint_early_book_transfer_v1 as incumbent
import joint_horizon_stage_references as previous
from y3_expanding_path_references import read_verified

trial,study,stack=previous.trial,previous.study,previous.stack
CACHE=incumbent.OUT/'historical_references'


def references(day, raw):
    return read_verified(CACHE,day,raw)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report=json.loads((incumbent.OUT/'results.json').read_text())
    picks=report['selected']
    early=[]
    training=[]
    errors=[]
    for d in range(16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        sig=incumbent.pressure.signature.observations(raw)
        if d>=8:
            if d%2==0:
                st=incumbent.estimates(early)
                if d>=10:
                    w=np.array([incumbent.fit(training,j)[picks[j],3:] for j in range(3)]).T
                    fold=[]
            x=incumbent.signals(raw,sig,st)
            p=previous.references(d,raw)
            if d>=10:
                q=p+(x@w).astype('float32')
                grams=[]
                for a,b in zip(p,q):
                    v=np.concatenate((np.ones_like(a[:,:,:1]),a,b,y),axis=2).reshape(-1,10).astype(float)
                    grams.append(v.T@v)
                fold.append(grams)
                if d%2:
                    comparison=[]
                    for g in np.sum(fold,axis=0):
                        c=stack.covariance(g)
                        std=np.sqrt(c.diagonal())
                        corr=c/(std[:,None]*std[None,:])
                        comparison.append(np.diag(corr[3:6,6:])-np.diag(corr[:3,6:]))
                    desired=np.column_stack([np.array(r['folds'])[(d-11)//2,:,pick] for r,pick in zip(report['targets'],picks)])
                    err=float(abs(np.asarray(comparison)-desired).max())
                    assert err<1e-8,err
                    errors.append(err)
            else:
                q=p
            path=CACHE/f'day_{d:02d}.npz'
            tmp=path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,forecasts=q,cutoff=d-d%2,raw_hash=previous.digest(raw),forecast_hash=previous.digest(q))
            tmp.replace(path)
            training.append(incumbent.observations(d,x,raw))
        t=np.arange(d%5,len(y)-60,5)
        early.append((incumbent.transition.incumbent.stats(raw,y,d),incumbent.pressure.record(sig,t,y)))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors,dates=list(range(9,17)),warmup=[9,10]),indent=2))
    print('historical fold errors',errors,flush=True)


def replay_with():
    from verify_joint_early_book_v1 import read,OUT
    old,loader=previous.replay_with()
    if (OUT/'composed_replay.json').is_file():
        return old,read
    apply=incumbent.correction()
    def run(day,raw):
        q,h,n=loader(day,raw)
        return apply(raw,q),apply(raw[:n],h),n
    return old,run


def outer(module):
    evaluator,load=replay_with()
    evaluator.outer(module,reference=incumbent,load_reference=load)


if __name__=='__main__':
    torch.set_num_threads(2)
    rebuild()
