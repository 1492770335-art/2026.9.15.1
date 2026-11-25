"""Reconstruct the chronological group-specific stack and independently replay the serialized readouts."""
import json
import sys

import numpy as np
import torch

import experiment_joint_factor_stage_stack_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest, read_stage

previous=incumbent.history
stack,study,trial=previous.stack,previous.study,previous.trial
CACHE=incumbent.OUT/'historical_references'
FULL=incumbent.OUT/'full_chain_verification'


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
    report=json.loads((incumbent.OUT/'results.json').read_text())
    picks=report['selected']
    records,errors=[],[]
    u=incumbent.risk.geometry()[0]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d)
        panels=[h.references(d,raw) for h in incumbent.ANCHORS]
        q=panels[-1].copy()
        if d>=10:
            if d%2==0:
                w=np.column_stack([incumbent.fit(records,k)[picks[k]] for k in range(3)])
                folds=[]
            y=trial.clock.labels(d).reshape(-1,50,3)
            grams=[]
            for j,anchors in enumerate(zip(*panels)):
                before=anchors[-1]
                q[j]+=(incumbent.regressors(anchors,u)[:,:,3:]@w).astype('float32')
                a=np.concatenate((np.ones_like(y[:,:,:1]),before,q[j],y),2).reshape(-1,10).astype(float)
                grams.append(a.T@a)
            folds.append(grams)
            if d%2:
                gains=[]
                for g in np.sum(folds,0):
                    c=stack.covariance(g);sd=np.sqrt(c.diagonal())
                    cor=c/sd[:,None]/sd[None,:]
                    gains.append(np.diag(cor[3:6,6:])-np.diag(cor[:3,6:]))
                expected=np.column_stack([np.array(r['folds'])[(d-11)//2,:,p] for r,p in zip(report['targets'],picks)])
                errors.append(float(np.max(abs(gains-expected))))
                assert errors[-1]<1e-8
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(q))
        records.append(incumbent.record(d))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    FULL.mkdir(exist_ok=True)
    with np.load(incumbent.OUT/'model.npz') as f:
        w=f['w'].reshape(4,9,3);u=f['basis'].astype(float)
    expected=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    trace=stack.verifier.trace
    apply=incumbent.adjustment()
    def recompute(p):
        v=np.concatenate([b-a for a,b in zip(p,p[1:])],2).astype(float)
        q=apply(p)
        heads=sum((u[:,a:b]@u[:,a:b].T)@(v@w[i]) for i,(a,b) in enumerate(incumbent.GROUPS))
        np.testing.assert_allclose(q,p[-1]+heads,rtol=0,atol=1e-6)
        return q
    rows,daily=[],{}
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,sizes=zip(*(stage.read(d,raw) for stage in incumbent.ANCHORS))
        assert len(set(sizes))==1
        q,head=recompute(p),recompute(h)
        err=float(abs(head-q[:sizes[0]]).max())
        assert err<1e-4 and np.isfinite(q).all()
        row=p[-1].reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
        rows.append(row)
        s=trace.older.measure([row])
        assert abs(s['score']-expected['daily'][str(d+1)]['score'])<1e-10
        daily[str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    s=trace.older.measure(rows)
    assert s['rows']==936250
    np.testing.assert_allclose(s['candidate'],expected['pooled']['candidate'],rtol=0,atol=1e-10)
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=s,daily=daily),indent=2))
    print('verified',s['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
