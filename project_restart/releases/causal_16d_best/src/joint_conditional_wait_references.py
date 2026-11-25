"""Reproduce event-timing folds and check that the selected five-tick change prices only the first arrival."""
import json
import sys

import numpy as np
import torch

import experiment_joint_conditional_first_wait_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage

previous=incumbent.history
trial,study,stack=previous.trial,previous.study,previous.stack
FULL=incumbent.OUT/'full_chain_verification'
CACHE=incumbent.OUT/'historical_references'


def references(d,raw):
    return read_verified(CACHE,d,raw)


def read(d,raw):
    return read_stage(FULL,d,raw)


def first_arrival_correction(detail):
    base=incumbent.base
    rate=base.pace()[1]
    group_raw=detail.sum(axis=-2)
    opp,yes,no,vp,vm=np.moveaxis(group_raw,-1,0)
    seen,up,down,up_size,down_size=np.moveaxis(detail,-1,0)
    count=yes+no
    pooled=count.sum((1,2),keepdims=True)
    exposure=opp.sum((1,2),keepdims=True)
    hazard=(count+60*(pooled+6*rate[:,None,None,None])/(exposure+6))/(opp+60)
    lift=((up+down).sum((1,2),keepdims=True)+12)/(seen.sum((1,2),keepdims=True)+60)
    lift/=((pooled+36)/(exposure+180))[:,:,:,:,None]
    conditional=(up+down+70*(hazard[:,:,:,:,None]*lift.clip(.4,2.5)).clip(.002,.998))/(seen+70)
    def magnitude(m,n,vc,nc):
        group=(m.sum((1,2),keepdims=True)+1)/(n.sum((1,2),keepdims=True)+1)
        avg=(m+25*group)/(n+25)
        return (vc-nc*avg[:,:,:,:,None])/(nc+40)
    plus=magnitude(vp,yes,up_size,up)
    minus=magnitude(vm,no,down_size,down)
    bins=np.stack([base.delay(np.arange(32),r) for r in rate])
    counts=np.zeros((16,3,5,2))
    for g in range(4):
        for k in range(5):
            for u,v in enumerate((no,yes)):
                counts[:,:,k,u]+=v[g][:,:,bins[g]==k].sum(2)
    def probability(m):
        soft=(m[:,:,:,1].sum(2)+4)/(m.sum((2,3))+8)
        return (m[:,:,:,1]+30*soft[:,:,None])/(m.sum(3)+30)
    fi=.6*probability(counts)+.4*probability(base.external()[0])
    direction=np.stack([fi[:,:,b] for b in bins])[:,:,:,:,None]
    reward=direction*plus-(1-direction)*minus
    value=np.zeros_like(conditional)
    alive=np.ones_like(conditional)
    for t in range(5):
        idx=np.minimum(np.arange(32)+t,31)
        h=conditional[:,:,:,idx]
        value+=alive*h*reward[:,:,:,idx]
        alive*=1-h
    return np.moveaxis(value,-1,0)


def sequential(raw,table,stocks,coef):
    z=np.zeros(raw.shape[:2],np.float32)
    for lo,hi in incumbent.base.segments(len(raw)):
        age=np.zeros(50,int);seen=np.zeros(50,int);motif=np.full(50,15,int)
        last=np.zeros(50,int);drift=np.zeros(50)
        qr=np.zeros(50);qs=np.zeros(50)
        for t in range(lo,hi):
            r=raw[t,:,0].astype(float)-.00136
            move=abs(r)>.022;sign=np.sign(r).astype(int)
            changed=np.zeros(50) if t==lo else np.any(raw[t][:,[21,39,41,42]]!=raw[t-1][:,[21,39,41,42]],axis=1)
            if t==lo:
                mr=move.astype(float);ms=mr.copy()
            def ema(old,new,tau):
                k=1-np.exp(-1/tau)
                return k*new+(1-k)*old
            qr,qs=ema(qr,changed,10),ema(qs,changed,120)
            mr,ms=ema(mr,move,12),ema(ms,move,160)
            speed=np.digitize(.6*(qr-qs)+.4*(mr-ms),[-.065,.065])
            same=(seen==0)|(last==sign)
            motif[move]=(2*motif[move]+same[move])%16
            age=np.where(move,0,np.minimum(age+1,31))
            drift[move]=.925*drift[move]+.075*sign[move]
            seen+=move;last[move]=sign[move]
            b=np.digitize(drift*last,[-.13,.13])
            z[t]=table[speed,stocks,motif,b,age]*last*(seen>=5)
    return ((z-z.mean(1,keepdims=True))*coef).astype('float32')


def verify():
    FULL.mkdir(exist_ok=True)
    trace=stack.verifier.trace
    report=json.loads((incumbent.OUT/'reload_verification.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as f:
        state=dict(f)
    assert int(state['idx'][0])==4 and np.all(state['coef'][1:]==0)
    count=sum(incumbent.stats(trial.clock.book.rawday(d)) for d in range(16))
    direct=first_arrival_correction(count)
    np.testing.assert_allclose(direct,state['w'][:,0,4],atol=3e-7,rtol=1e-5)
    apply=incumbent.correction()
    rows=[];daily={}
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=previous.read(d,raw)
        q=p+apply(raw);head=h+apply(raw[:n])
        pred=p.copy()
        pred[:,:,0]+=sequential(raw,direct,state['stocks'],float(state['coef'][0]))
        error=float(abs(q-pred).max())
        assert error<1e-6 and np.array_equal(p[:,:,1:],q[:,:,1:]) and abs(head-q[:n]).max()<1e-4
        rows.append((p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d)))
        score=trace.older.measure(rows[-1:])['score']
        assert abs(score-report['daily'][str(d+1)]['score'])<1e-10
        daily[d+1]=dict(score=score,independent_error=error,prefix_error=float(abs(head-q[:n]).max()))
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    q=trace.older.measure(rows)
    assert q['rows']==936250 and abs(q['score']-report['pooled']['score'])<1e-10
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=q,daily=daily),indent=2))
    print('verified',q['score'],daily,flush=True)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    rep=json.loads((incumbent.OUT/'results.json').read_text())
    picks=np.array(rep['selected'])
    idx=np.maximum(0,picks-1)
    stocks=incumbent.base.pace()[0]
    stats=[];rows=[];errors=[]
    for d in range(16):
        raw=trial.clock.book.rawday(d)
        if d in (4,8,12):
            table=incumbent.table(stats)
        if d in (8,12):
            coef=incumbent.parent.scales(rows);fold=[]
        if d>=4:
            p=previous.references(d,raw) if d>=8 else incumbent.parent.warm.read(d,raw)
            y=trial.clock.labels(d).reshape(-1,50,3)
            v=incumbent.signals(raw,table,stocks)
            ms=[incumbent.parent.record(ref,y,v) for ref in p]
            rows.append(ms)
            if d>=8:
                w=coef[np.arange(3),idx]*(picks>0)
                q=p+(v[:,:,np.arange(3),idx]*w).astype('float32')
                np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,event_cutoff=d-d%4,raw_hash=digest(raw),forecast_hash=digest(q))
                fold.append(ms)
                if d in (11,15):
                    obs=np.array([incumbent.parent.scores(m,coef) for m in np.sum(fold,0)])
                    error=float(abs(obs-np.array(rep['folds'])[(d-11)//4]).max())
                    assert error<1e-10
                    errors.append(error)
        stats.append(incumbent.stats(raw))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_array_equal(f['w'],incumbent.table(stats))
        np.testing.assert_array_equal(f['coef'],incumbent.parent.scales(rows)[np.arange(3),idx]*(picks>0))
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
