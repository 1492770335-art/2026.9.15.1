"""Rebuild chronological forecasts and independently evaluate the selected execution-pressure correction."""
import json
import sys

import numpy as np

import experiment_y3_external_panel_message_origins_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest, read_stage

previous=incumbent.history
base=incumbent.base
trial,study,stack=previous.trial,previous.study,previous.stack
FULL=incumbent.OUT/'full_chain_verification'
CACHE=incumbent.OUT/'historical_references'


def references(d,raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    return read_stage(FULL,d,raw)


def independently(raw,state):
    # Only the late execution readout and its individual-stock control enter the selected correction.
    w=state['w'][0,:,7].astype(float)-state['w'][1,:,7].astype(float)
    lower,upper=state['lim']
    imbalance=(2*(raw[:,:,21].astype(float)-lower)/(upper-lower)-1).clip(-1,1)
    out=np.zeros(raw.shape[:2])
    for lo,hi in base.segments(len(raw)):
        a,_=incumbent.inputs(base.raw_returns(raw[lo:hi]),imbalance[lo:hi])
        for order in state['orders']:
            inv=np.argsort(order)
            for begin in range(0,len(a),97):
                end=min(len(a),begin+97)
                z=incumbent.features(a[begin:end,order]).astype(float)@w
                out[lo+begin:lo+end]+=z[:,inv]/len(state['orders'])
    value=out*np.tanh(2*out*imbalance)
    return ((value-value.mean(1,keepdims=True))*float(state['coef'])).astype('float32')


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report=json.loads((incumbent.OUT/'results.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as f: st=dict(f)
    assert report['arms'][report['selected']]=='peer execution with confirming quote'
    assert int(st['pick'])==report['selected']-1
    learned,source=incumbent.learn()
    np.testing.assert_array_equal(learned,st['w'])
    assert source==report['source']
    np.testing.assert_array_equal(incumbent.imbalance_bounds(),st['lim'])
    rows,errors=[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        ps=previous.references(d,raw) if d>=8 else incumbent.warm.read(d,raw)
        y=trial.clock.labels(d).reshape(-1,50,3)
        if d in (8,12):
            coef=base.calibrate(rows)
            fold=[]
        z=incumbent.design(raw,st['w'],st['orders'],st['lim'])
        r=[base.records(p,y,z) for p in ps]
        rows.append(r)
        if d>=8:
            fold.append(r)
            q=ps.copy()
            q[:,:,:,2]+=(z[:,:,int(st['pick'])]*coef[int(st['pick'])]).astype('float32')
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,
                     calibration_cutoff=d-d%4,raw_hash=digest(raw),forecast_hash=digest(q))
            if d in (11,15):
                actual=[base.score(m,coef) for m in np.sum(fold,axis=0)]
                errors.append(float(abs(np.asarray(actual)-np.array(report['folds'])[(d-11)//4]).max()))
                assert errors[-1]<1e-12
    np.testing.assert_array_equal(st['coef'],base.calibrate(rows)[int(st['pick'])])
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors),indent=2))


def verify():
    FULL.mkdir(exist_ok=True)
    trace=stack.verifier.trace
    report=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    apply=incumbent.correction()
    with np.load(incumbent.OUT/'model.npz') as f: st=dict(f)
    rows,daily=[],{}
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=previous.read(d,raw)
        q=p.copy()
        head=h.copy()
        change=apply(raw)
        q[:,:,2]+=change
        head[:,:,2]+=apply(raw[:n])
        independent=independently(raw,st)
        error=float(abs(change-independent).max())
        assert error<1e-6 and np.array_equal(q[:,:,:2],p[:,:,:2])
        assert abs(head-q[:n]).max()<1e-4
        row=(p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        score=trace.older.measure([row])['score']
        assert abs(score-report['daily'][str(d+1)]['score'])<1e-11
        rows.append(row)
        daily[d+1]=dict(score=score,independent_error=error,prefix_error=float(abs(head-q[:n]).max()))
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    stats=trace.older.measure(rows)
    assert stats['rows']==936250 and abs(stats['score']-report['pooled']['score'])<1e-11
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=stats,daily=daily),indent=2))
    print('verified',stats['score'],daily,flush=True)


def replay_with():
    return stack.verifier,read


if __name__=='__main__':
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_y3_release_gain_v1 as audit
        audit.run(incumbent,read,previous.incumbent,previous.read)
        audit.run(incumbent,read)
    else:
        rebuild()
