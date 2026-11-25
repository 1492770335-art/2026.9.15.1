"""Learn a separate response for each target to ordered shocks in three orthogonal stock groups."""
import json
import sys

import numpy as np
import torch

import experiment_joint_order_clock_calibration_v1 as prior
import y3_factor_order_references as history

parent,trial,stack=history.incumbent,history.trial,history.stack
OUT=trial.clock.ART/'joint_factor_order_calibration_v1'
ARMS=('unchanged','half increment','one increment','two increments','median amplitude','distinct factor groups',
      'factor groups plus last adjustment','group slopes with date agreement')
fitted,gain=prior.fitted,prior.gain


def signals(raw,st):
    x=parent.rotate(parent.features(raw,np.arange(len(raw))),st['basis'][0])
    parts=[]
    for (a,b),w in zip(parent.GROUPS,st['returns'][0]):
        z=x[:,a:b]@w[:,:,-1].sum(1)
        parts.append(z@st['basis'][0][:,a:b].T)
    return trial.clock.center(np.stack(parts,2))


def basis(old,p,x):
    return np.concatenate(((p[:,:,2]-old[:,:,2])[:,:,None],x),2)


def record(d):
    raw=trial.clock.book.rawday(d)
    q=history.references(d,raw)
    old=history.previous.references(d,raw)
    x=signals(raw,parent.params(d-d%2))
    y=trial.clock.labels(d).reshape(-1,50,3)
    data=[]
    for a,p in zip(old,q):
        v=np.concatenate((np.ones_like(p[:,:,:1]),p,basis(a,p,x),y),2).reshape(-1,11).astype(float)
        data.append(v.T@v)
    return data


def train(module=None):
    module = module or sys.modules[__name__]
    module.OUT.mkdir(exist_ok=True)
    stats=[module.record(d) for d in range(8,16)]
    picks,heads,reports=[],[],[]
    for k in range(3):
        folds,daily=[],[]
        for n in (2,4,6):
            w=fitted(stats[:n],k)
            folds.append([gain(g,w,k) for g in np.sum(stats[n:n+2],0)])
            daily.extend([[gain(g,w,k) for g in v] for v in stats[n:n+2]])
        f,d=np.asarray(folds),np.asarray(daily)
        avg=f.mean(0)
        valid=[i for i in range(1,len(module.ARMS)) if avg[:,i].min()>0 and f[-1,:,i].min()>0 and
               (f[:,:,i]>0).sum(0).min()>=2 and (d[:,:,i]>0).sum(0).min()>=4]
        pick=max(valid,key=lambda i:avg[:,i].min()) if valid else 0
        picks.append(pick)
        heads.append(fitted(stats,k)[pick])
        reports.append(dict(folds=f.tolist(),daily=d.tolist(),mean=avg.tolist()))
        print('target',k+1,module.ARMS[pick],avg[:,pick],flush=True)
    (module.OUT/'results.json').write_text(json.dumps(dict(selected=picks,arms=module.ARMS,targets=reports),indent=2))
    if any(picks):
        np.savez(module.OUT/'model.npz',w=np.column_stack(heads),selected=picks)
    return any(picks)


def adjustment():
    with np.load(OUT/'model.npz') as f:
        w=f['w']
    with np.load(parent.OUT/'model.npz') as f:
        st=dict(f)
    def apply(raw,old,p):
        return p+(basis(old,p,signals(raw,st))@w).astype('float32')
    return apply


def layer():
    before=history.previous.incumbent.layer()
    step=parent.correction()
    adapt=adjustment()
    def apply(raw,base):
        old=before(raw,base)
        p=old.copy()
        p[:,:,2]+=step(raw,old)
        return adapt(raw,old,p)
    return apply


def replay_with():
    adapt=adjustment()
    def read(d,raw):
        old,short,_=history.previous.read(d,raw)
        p,h,n=history.read(d,raw)
        return adapt(raw,old,p),adapt(raw[:n],short,h),n
    return stack.verifier,read


def replay(module=None):
    module = module or sys.modules[__name__]
    expected=json.loads((module.parent.OUT/'reload_verification.json').read_text())
    first=None
    for name in ('raw_replay','reload_verification'):
        ver,new=module.replay_with()
        rows,daily=[],{}
        for d in range(16,20):
            raw=ver.trace.replay.original.features(ver.trace.replay.original.DATA,d)
            p,_,_=module.history.read(d,raw)
            q,h,n=new(d,raw)
            err=float(abs(q[:n]-h).max())
            assert err<1e-4 and np.isfinite(q).all()
            row=(p.reshape(-1,3),q.reshape(-1,3),ver.trace.replay.labels(d))
            rows.append(row)
            s=ver.trace.older.measure([row])
            assert abs(s['baseline_score']-expected['daily'][str(d+1)]['score'])<1e-10
            daily[str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=err,corr=s['candidate'])
        s=ver.trace.older.measure(rows)
        assert s['rows']==936250
        np.testing.assert_allclose(s['base'],expected['pooled']['candidate'],atol=1e-10,rtol=0)
        report=dict(parent=module.parent.OUT.name,pooled=s,daily=daily)
        if first:
            assert first==report
        (module.OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,s,flush=True)
        first=report
        if s['gain']<=0:
            break


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay()
