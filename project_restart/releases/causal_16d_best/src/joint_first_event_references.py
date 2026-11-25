"""Verify the selected event responses sequentially and reproduce their matched four-day reference folds."""
import json
import sys

import numpy as np
import torch

import experiment_joint_first_event_transfer_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import read_stage,digest

previous=incumbent.history
trial,study,stack=previous.trial,previous.study,previous.stack
FULL=incumbent.OUT/'full_chain_verification'
CACHE=incumbent.OUT/'historical_references'


def read(day,raw):
    return read_stage(FULL,day,raw)


def references(day,raw):
    return read_verified(CACHE,day,raw)


def sequential(raw,st):
    w,stocks,indices,coeff=st['w'],st['stocks'],st['pick'],st['coef']
    names=np.arange(50)
    out=np.zeros((*raw.shape[:2],3),np.float32)
    for lo,hi in incumbent.parent.segments(len(raw)):
        last=np.zeros(50,int)
        pattern=np.full(50,15,int)
        age=np.zeros(50,int)
        drift=np.zeros(50)
        count=np.zeros(50,int)
        for t in range(lo,hi):
            r=raw[t,:,0].astype(float)-.00136
            move=abs(r)>.022
            sign=np.sign(r).astype(int)
            same=(sign==last)|(count==0)
            pattern[move]=(2*pattern[move]+same[move])%16
            age=np.where(move,0,np.minimum(age+1,31))
            drift[move]=.925*drift[move]+.075*sign[move]
            last[move]=sign[move]
            count+=move
            orient=np.digitize(drift*last,[-.13,.13])
            for k,j in enumerate(indices):
                out[t,names,k]=w[k,j,stocks,pattern,orient,age]*last*(count>=5)
    return ((out-out.mean(1,keepdims=True))*coeff).astype('float32')


def verify():
    FULL.mkdir(exist_ok=True)
    trace=stack.verifier.trace
    report=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as saved:
        state=dict(saved)
    with np.load(incumbent.parent.OUT/'model.npz') as f:
        expected=f['w'][3].sum(0)-f['w'][0].sum(0)
        np.testing.assert_allclose(state['w'][2,2],expected,rtol=1e-5,atol=5e-7)
    delta=incumbent.correction()
    rows=[];daily={}
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=previous.read(d,raw)
        q=p+delta(raw)
        head=h+delta(raw[:n])
        independent=p+sequential(raw,state)
        err=float(abs(independent-q).max())
        np.testing.assert_allclose(independent,q,rtol=0,atol=1.e-6)
        assert abs(head-q[:n]).max()<1e-4
        assert np.array_equal(q[:,:,0],p[:,:,0])
        rows.append((p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)))
        s=trace.older.measure(rows[-1:])
        assert abs(s['score']-report['daily'][str(d+1)]['score'])<1e-10
        daily[d+1]=dict(score=s['score'],independent_error=err,prefix_error=float(abs(q[:n]-head).max()))
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    result=trace.older.measure(rows)
    assert result['rows']==936250 and abs(result['score']-report['pooled']['score'])<1.e-10
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=result,daily=daily),indent=2))
    print('verified',result['score'],daily,flush=True)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report=json.loads((incumbent.OUT/'results.json').read_text())
    ids=np.maximum(0,np.array(report['selected'])-1)
    active=np.array(report['selected'])>0
    stocks,_=incumbent.parent.pace()
    stats=[];rows=[];errors=[]
    for d in range(16):
        raw=trial.clock.book.rawday(d)
        if d in (4,8,12):
            w=incumbent.table(stats)
        if d in (8,12):
            scale=incumbent.scales(rows)
            fold=[]
        if d>=4:
            z=incumbent.signals(raw,w,stocks)
            ps=incumbent.warm.read(d,raw)
            y=trial.clock.labels(d).reshape(-1,50,3)
            ms=[incumbent.record(p,y,z) for p in ps]
            rows.append(ms)
            if d>=8:
                fold.append(ms)
                coef=scale[np.arange(3),ids]*active
                delta=(z[:,:,np.arange(3),ids]*coef).astype('float32')
                q=ps+delta
                np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,event_cutoff=d-d%4,
                         forecast_hash=digest(q),raw_hash=digest(raw))
                if d in (11,15):
                    obs=[incumbent.scores(m,scale) for m in np.sum(fold,0)]
                    error=float(abs(np.asarray(obs)-np.array(report['folds'])[(d-11)//4]).max())
                    assert error<1.e-10
                    errors.append(error)
        stats.append(incumbent.parent.moment(raw))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_array_equal(incumbent.table(stats),f['w'])
        np.testing.assert_array_equal(f['coef'],incumbent.scales(rows)[np.arange(3),ids]*active)
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def replay_with():
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_y3_release_gain_v1 as gain
        gain.run(incumbent,read,previous.incumbent,previous.read)
        gain.run(incumbent,read)
    else:
        rebuild()
