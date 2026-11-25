"""Verify short-horizon peer impulses and reconstruct chronological Y3 forecasts."""
import json
import sys

import numpy as np
import torch

import experiment_y3_peer_carry_impulse_v1 as incumbent
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
    pick=report['selected']
    stats,errors=[],[]
    context=incumbent.context
    warm=context.parent.readouts(context.carry.warmup(),1)[6]
    basis=context.carry.peer.U()[0].astype(float)
    for d in range(8,16):
        raw=trial.clock.book.rawday(d)
        q=previous.references(d,raw)
        old=previous.previous.references(d,raw)
        y=trial.clock.labels(d).reshape(-1,50,3)
        if d>=10 and d%2==0:
            w=context.weights(stats)[pick]
            group=[]
        rows,grams=[],[]
        for a,p in zip(old,q):
            z=p[:,:,1]-a[:,:,1] if d>=10 else context.carry.legs(a,warm,basis).sum(2)
            rows.append(context.record(raw,p,z,y,incumbent))
            if d>=10:
                before=p[:,:,2].copy()
                p[:,:,2]+=(incumbent.features(raw,p,z)[0]@w).astype('float32')
                terms=np.stack((np.ones_like(before),before,p[:,:,2],y[:,:,2]),axis=2).reshape(-1,4).astype(float)
                grams.append(terms.T@terms)
        stats.append(rows)
        if d>=10:
            group.append(grams)
            if d%2:
                found=[]
                for m in np.sum(group,axis=0):
                    c=stack.covariance(m)
                    r=c[:2,2]/np.sqrt(c.diagonal()[:2]*c[2,2])
                    found.append(r[1]-r[0])
                want=np.asarray(report['folds'])[(d-11)//2,:,pick,1]
                errors.append(float(np.max(abs(found-want))))
                assert errors[-1]<1e-8
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(q))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    FULL.mkdir(exist_ok=True)
    trace = stack.verifier.trace
    expect = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as archive:
        w=archive['coef']
        assert int(archive['pick'])==2 and (w[2:]==0).all()
    run=incumbent.context.adjustment(incumbent)
    rows,daily=[],{}
    def independent(raw,old,p):
        z=p[:,:,1]-old[:,:,1]
        mean=np.empty_like(z,dtype=float)
        for start,end in ((0,min(len(z),2401)),(2401,len(z))):
            for t in range(start,end):
                mean[t]=z[max(start,t-4):t+1].mean(0,dtype=float)
        out=p.copy()
        out[:,:,2]+=(w[0]*z.astype(float)+w[1]*(z-mean)).astype('float32')
        q=run(raw,old,p)
        np.testing.assert_allclose(q,out,rtol=0,atol=1.e-6)
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
