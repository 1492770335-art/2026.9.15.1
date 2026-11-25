"""Reconstruct frozen three-target predictions for chronological research and verify composed replay."""
import json
import os
import sys
import numpy as np
import torch
import experiment_joint_order_clock_calibration_v1 as incumbent
import verify_order_clock_readouts as verifier
from y3_expanding_path_references import read_verified

previous=incumbent.history
trial,study,stack=previous.trial,previous.study,previous.stack
CACHE=incumbent.OUT/'historical_references'
FULL=incumbent.OUT/'full_chain_verification'


def references(d,raw):
    return read_verified(CACHE,d,raw)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    past=[]
    errors=[]
    report=json.loads((incumbent.OUT/'results.json').read_text())
    picks=report['selected']
    for d in range(8,16):
        raw=trial.clock.book.rawday(d)
        p=previous.references(d,raw)
        if d>=10:
            if not d%2:
                w=np.column_stack([incumbent.fitted(past,k)[picks[k]] for k in range(3)])
                grams=[]
            original=previous.previous.references(d,raw)
            w0=incumbent.clocks.parameters(d-d%2)[:,-3:]
            bands=incumbent.clocks.design_bank(raw,np.arange(len(raw)))@w0
            y=trial.clock.labels(d).reshape(-1,50,3)
            day=[]
            q=p.copy()
            for j,(old,current) in enumerate(zip(original,p)):
                z=incumbent.basis(current[:,:,2]-old[:,:,2],bands)
                q[j]+=(z@w).astype('float32')
                v=np.concatenate((np.ones_like(current[:,:,:1]),current,q[j],y),2).reshape(-1,10).astype(float)
                day.append(v.T@v)
            grams.append(day)
            if d%2:
                actual=[]
                for g in np.sum(grams,0):
                    c=stack.covariance(g)
                    std=np.sqrt(c.diagonal())
                    corr=c/(std[:,None]*std[None,:])
                    actual.append(np.diag(corr[3:6,6:])-np.diag(corr[:3,6:]))
                expected=np.column_stack([np.array(result['folds'])[(d-11)//2,:,p] for result,p in zip(report['targets'],picks)])
                err=float(abs(actual-expected).max())
                assert err<1e-8,err
                errors.append(err)
        else:
            q=p
        path=CACHE/f'day_{d:02d}.npz'
        tmp=path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(tmp,forecasts=q,cutoff=d-d%2,raw_hash=previous.digest(raw),forecast_hash=previous.digest(q))
        tmp.replace(path)
        past.append(incumbent.record(d))
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors,dates=list(range(9,17)),warmup_days=[9,10]),indent=2))
    print('verified historical folds',errors,flush=True)


def read(d,raw):
    with np.load(FULL/f'day_{d}.npz') as f:
        q,h=f['full'],f['head']
        assert [str(f[k]) for k in ('raw_hash','full_hash','head_hash')]==[previous.digest(a) for a in (raw,q,h)]
    return q,h,len(h)


def verify():
    trace=verifier.trace
    assert (verifier.OUT/'composed_replay.json').exists()
    ref=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    _,older=previous.previous.replay_with()
    production=incumbent.readout()
    with np.load(incumbent.OUT/'model.npz') as f:
        w=f['w']
    with np.load(incumbent.clocks.OUT/'model.npz') as f:
        directions=f['clocks'][:,-3:]
    rows=[]
    daily={}
    FULL.mkdir(exist_ok=True)
    def reconstruct(raw,old,p):
        # Apply the joint readout independently to the outputs of a freshly recomputed parent chain.
        design=incumbent.clocks.design_bank(raw,np.arange(len(raw)))
        bands=design@directions
        z=incumbent.basis(p[:,:,2]-old[:,:,2],bands)
        delta=sum(z[:,:,j,None].astype(float)*w[j] for j in range(4))
        q=p+delta.astype('float32')
        parent,expected=production(raw,old)
        np.testing.assert_array_equal(parent,p)
        np.testing.assert_allclose(q,expected,rtol=0,atol=1e-6)
        return expected
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        old,short,n=older(d,raw)
        p,h,_=verifier.read(d,raw)
        q=reconstruct(raw,old,p)
        head=reconstruct(raw[:n],short,h)
        err=float(abs(head-q[:n]).max())
        assert np.isfinite(q).all() and err<1e-4
        row=p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
        rows.append(row)
        s=trace.older.measure([row])
        assert abs(s['score']-ref['daily'][str(d+1)]['score'])<1e-10
        daily[str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=previous.digest(raw),full_hash=previous.digest(q),head_hash=previous.digest(head))
    s=trace.older.measure(rows)
    assert s['rows']==ref['pooled']['rows']==936250
    np.testing.assert_allclose(s['candidate'],ref['pooled']['candidate'],rtol=0,atol=1e-10)
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=s,daily=daily),indent=2))
    print('verified full chain and score',s['score'],flush=True)


def replay_with():
    return (stack.verifier,read) if (FULL/'composed_replay.json').is_file() else incumbent.replay_with()


def outer(module):
    evaluation,loader=replay_with()
    evaluation.outer(module,reference=incumbent,load_reference=loader)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    else:
        rebuild()
