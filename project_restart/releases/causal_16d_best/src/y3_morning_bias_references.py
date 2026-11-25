"""Rebuild training forecasts chronologically and verify the afternoon adjustment by direct compounded returns."""
import json
import sys

import numpy as np

import experiment_y3_morning_bias_transfer_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage

previous=incumbent.history
base,trial,study,stack=previous.base,previous.trial,previous.study,previous.stack
CACHE=incumbent.OUT/'historical_references'
FULL=incumbent.OUT/'full_chain_verification'


def references(day,raw):
    return read_verified(CACHE,day,raw)


def read(day,raw):
    return read_stage(FULL,day,raw)


def direct(raw,p,k,coef,row,weight):
    result=np.zeros(p.shape[:2],np.float32)
    if len(raw)<=2401:
        return result
    origin=np.array([i+j for i in range(0,2340,180) for j in range(0,120,4)])
    x=raw[:2401]
    r=x[:,:,0].astype(float)-.00136
    ok=~(x==row).all(2)
    mask=np.stack([ok[t:t+61].all(0) for t in origin])
    def center(v):
        return (v-(v*mask).sum(1,keepdims=True)/mask.sum(1,keepdims=True).clip(1))*mask
    def ret(h,j):
        z=np.stack([np.expm1(np.log1p(k*r[t+1:t+1+h]).sum(0))/k for t in origin])
        return coef[j,0]+coef[j,1]*z
    near,end=ret(20,1),ret(60,2)
    errs=[]
    for a,b in ((end,p[origin,:,2]),(end-base.RATIO*near,p[origin,:,2]-base.RATIO*p[origin,:,1])):
        label,pred=center(a),center(b)
        slope=np.sum(label*pred)/np.sum(pred*pred)
        error=(label/slope.clip(.35,2.8)-pred).sum(0)/(12+mask.sum(0))
        errs.append(error-error.mean())
    a,b=errs
    s=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    result[2401:]=(s-s.mean()).astype(np.float32)*weight
    return result


def verify():
    FULL.mkdir(exist_ok=True)
    with np.load(incumbent.OUT/'model.npz') as f:
        assert incumbent.ARMS[int(f['pick'])+1]=='consistent full and distant errors'
        params=float(f['k']),f['coef'],f['template'],float(f['w'])
    rows,daily=[],{}
    update=incumbent.correction()
    for d in range(16,20):
        raw=trial.clock.book.rawday(d)
        p,_,_=previous.read(d,raw)
        q,h,n=incumbent.replay_with()[1](d,raw)
        approx=direct(raw,p,*params)
        err=float(abs((p[:,:,2]+approx)-q[:,:,2]).max())
        assert err<2e-6 and np.array_equal(q[:2401],p[:2401]) and np.array_equal(q[:,:,:2],p[:,:,:2])
        for cutoff in (2401,2402,3553):
            np.testing.assert_array_equal(update(raw[:cutoff],p[:cutoff]),update(raw,p)[:cutoff])
        rows.append((p.reshape(-1,3),q.reshape(-1,3),stack.verifier.trace.replay.labels(d)))
        daily[d+1]=dict(afternoon_prefixes=[2402,3553],independent_max_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled=stack.verifier.trace.older.measure(rows)
    expected=json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']
    assert pooled==expected
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report=json.loads((incumbent.OUT/'results.json').read_text())
    pick=report['selected']-1
    rows,past,daily=[],[],{}
    geo=trial.curve.settled.historical_geometry()
    for d in range(4,16):
        if d in (8,12):
            weight=incumbent.fit(rows)[pick]
            fixed=incumbent.historical(past)
        raw,y,ps=incumbent.records(d)
        prior=fixed if d>=8 else incumbent.historical(past)
        noon=[incumbent.signals_at_noon(raw,p,geo) for p in ps]
        q=ps.copy()
        moments,gains=[],[]
        for p,v,m in zip(ps,q,noon):
            x=incumbent.design(m,prior,len(p))
            moments.append(base.records(p,y,x))
            if d>=8:
                v[:,:,2]+=x[:,:,pick]*weight
                gains.append(np.corrcoef(v[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-
                             np.corrcoef(p[:,:,2].ravel(),y[:,:,2].ravel())[0,1])
        rows.append(moments)
        past.append(np.mean(noon,0))
        if d>=8:
            np.testing.assert_allclose(gains,np.array(report['daily'])[d-8,:,pick+1],atol=1e-8,rtol=0)
            daily[d+1]=gains
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,raw_hash=digest(raw),forecast_hash=digest(q),
                     cutoff=d-d%2,calibration_cutoff=d-d%4)
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(incumbent.fit(rows)[pick],f['w'],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(incumbent.historical(past),f['prior'])
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=daily),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as audit
        audit.run(incumbent,read,previous.incumbent,previous.read)
        audit.run(incumbent,read)
