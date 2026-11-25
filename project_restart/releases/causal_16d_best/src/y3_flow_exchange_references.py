"""Recreate the historical fold predictions and verify the selected cross-stock comparison one donor at a time."""
import json
import sys

import numpy as np

import experiment_y3_conditional_flow_exchange_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import read_stage,digest

previous=incumbent.history
base,trial,study,stack=incumbent.base,previous.trial,previous.study,previous.stack
FULL=incumbent.OUT/'full_chain_verification'
CACHE=incumbent.OUT/'historical_references'


def read(d,raw):
    return read_stage(FULL,d,raw)


def references(d,raw):
    return read_verified(CACHE,d,raw)


def independent(raw,st):
    # Accumulate donor terms in stock order. Each receiver keeps its observed imbalance.
    amp,b=incumbent.flow(raw,st)
    own=amp*np.tanh(2*b*amp)
    total=np.zeros_like(b)
    weighted=np.zeros_like(b)
    weight=np.zeros_like(b)
    for s in range(50):
        val=amp[:,s,None]*np.tanh(2*b*amp[:,s,None])
        dist=np.exp(-4*abs(b-b[:,s,None]))
        val[:,s]=0
        dist[:,s]=0
        total+=val
        weight+=dist
        weighted+=dist*val
    other=(weighted+2*total/49)/(weight+2)
    v=own-other
    return ((v-v.mean(1,keepdims=True))*float(st['coef'])).astype('float32')


def verify():
    FULL.mkdir(exist_ok=True)
    trace=stack.verifier.trace
    expected=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as f:st=dict(f)
    assert incumbent.ARMS[int(st['pick'])+1]=='own minus similar signed imbalance'
    step=incumbent.correction()
    rows,daily=[],{}
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=previous.read(d,raw)
        q=p.copy();head=h.copy()
        delta=step(raw);q[:,:,2]+=delta;head[:,:,2]+=step(raw[:n])
        err=float(abs(delta-independent(raw,st)).max())
        assert err<5e-7 and abs(q[:n]-head).max()<1e-4 and np.array_equal(q[:,:,:2],p[:,:,:2])
        row=p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)
        rows.append(row)
        score=trace.older.measure([row])['score']
        assert abs(score-expected['daily'][str(d+1)]['score'])<1e-10
        daily[d+1]=dict(score=score,independent_error=err,prefix_error=float(abs(q[:n]-head).max()))
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    report=trace.older.measure(rows)
    assert report['rows']==936250 and abs(report['score']-expected['pooled']['score'])<1e-10
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=report,daily=daily),indent=2))
    print('full check',report['score'],daily,flush=True)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    with np.load(incumbent.OUT/'model.npz') as f:st=dict(f)
    for k,v in incumbent.source_state().items(): np.testing.assert_array_equal(v,st[k])
    pick=int(st['pick'])
    original=json.loads((incumbent.OUT/'results.json').read_text())
    assert original['selected']==pick+1
    rows,errors=[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        z=incumbent.design(raw,st)
        ps=previous.references(d,raw) if d>=8 else incumbent.parent.warm.read(d,raw)
        y=trial.clock.labels(d).reshape(-1,50,3)
        if d in (8,12):
            w=base.calibrate(rows);fold=[]
        block=[base.records(p,y,z) for p in ps]
        rows.append(block)
        if d>=8:
            fold.append(block)
            q=ps.copy();q[:,:,:,2]+=(z[:,:,pick]*w[pick]).astype('float32')
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,
                     calibration_cutoff=d-d%4,raw_hash=digest(raw),forecast_hash=digest(q))
            if d in (11,15):
                v=np.asarray([base.score(m,w) for m in np.sum(fold,axis=0)])
                error=float(abs(v-np.asarray(original['folds'])[(d-11)//4]).max())
                assert error<1e-10
                errors.append(error)
    np.testing.assert_array_equal(st['coef'],base.calibrate(rows)[pick])
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def replay_with():
    return stack.verifier,read


if __name__=='__main__':
    if sys.argv[1:]==['verify']:verify()
    elif sys.argv[1:]==['audit']:
        import audit_y3_release_gain_v1 as audit
        audit.run(incumbent,read,previous.incumbent,previous.read)
        audit.run(incumbent,read)
    else:rebuild()
