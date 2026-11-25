"""Independently reconstruct market and relative transfer forecasts in stock coordinates."""
import json
import sys

import numpy as np
import torch

import experiment_joint_separated_peer_transfer_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest, read_stage

previous = incumbent.history
stack, trial, study = previous.stack, previous.trial, previous.study
CACHE = incumbent.OUT/'historical_references'
FULL = incumbent.OUT/'full_chain_verification'


def references(day,raw):
    return read_verified(CACHE,day,raw)


def read(day,raw):
    return read_stage(FULL,day,raw)


def replay_with():
    return stack.verifier,read


def outer(module):
    stack.verifier.outer(module,reference=incumbent,load_reference=read)


def stock_weights(w,basis):
    return np.einsum('si,ihjk,tj->shtk',basis,w.reshape(50,3,50,3),basis,optimize=True).reshape(150,150)


def predict(p,weights):
    return p+(p.reshape(len(p),150).astype(float)@weights).reshape(p.shape).astype('float32')


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report=json.loads((incumbent.OUT/'results.json').read_text())
    picks=report['selected']
    records,errors=[],[]
    basis=incumbent.peer.U()[0].astype(float)
    for day in range(8,16):
        raw=trial.clock.book.rawday(day)
        q=previous.references(day,raw)
        y=trial.clock.labels(day).reshape(-1,50,3)
        moments=[incumbent.peer.moments(p,y,basis) for p in q]
        if day>=10:
            if day%2==0:
                w=np.stack([incumbent.readouts(records,k)[j] for k,j in enumerate(picks)],axis=2)
                weights=stock_weights(w,basis)
                group=[]
            grams=[]
            for p in q:
                updated=predict(p,weights)
                a=np.concatenate((np.ones_like(y[:,:,:1]),p,updated,y),axis=2).reshape(-1,10).astype(float)
                grams.append(a.T@a)
                p[:]=updated
            group.append(grams)
            if day%2:
                measured=[]
                for g in np.sum(group,axis=0):
                    c=stack.covariance(g)
                    s=np.sqrt(c.diagonal())
                    cor=c/s[:,None]/s[None,:]
                    measured.append(np.diag(cor[3:6,6:])-np.diag(cor[:3,6:]))
                want=np.column_stack([np.array(v['folds'])[(day-11)//2,:,j] for v,j in zip(report['targets'],picks)])
                err=float(np.max(abs(measured-want)))
                assert err<1e-7,err
                errors.append(err)
        records.append(moments)
        np.savez(CACHE/f'day_{day:02d}.npz',forecasts=q,cutoff=day-day%2,raw_hash=digest(raw),forecast_hash=digest(q))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    FULL.mkdir(exist_ok=True)
    with np.load(incumbent.OUT/'model.npz') as f:
        weights=stock_weights(f['w'],f['basis'].astype(float))
    apply=incumbent.adjustment()
    report=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    trace=stack.verifier.trace
    rows=[]
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=previous.read(d,raw)
        q,part=apply(p),apply(h)
        for x,y in ((q,predict(p,weights)),(part,predict(h,weights)),(q[:n],part)):
            np.testing.assert_allclose(x,y,rtol=0,atol=2e-6)
        row=(p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        rows.append(row)
        metric=trace.older.measure([row])
        assert abs(metric['score']-report['daily'][str(d+1)]['score'])<1e-10
        np.savez(FULL/f'day_{d}.npz',full=q,head=part,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(part))
    s=trace.older.measure(rows)
    assert s['rows']==936250 and abs(s['score']-report['pooled']['score'])<1e-10
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=s)))
    print('verified',s['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
