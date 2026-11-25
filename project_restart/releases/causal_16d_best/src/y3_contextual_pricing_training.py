"""Train return-error readouts on expanding dates and reconstruct their frozen rolling predictions."""
import json

import numpy as np
import experiment_y3_conditional_forecast_skill_v1 as metrics
from y3_touch_clock_references import digest


def compare(mod,raw,p,y,st):
    q = p[:,:,2:]+mod.signals(raw,p,st)
    v = np.concatenate((np.ones_like(q[:,:,:1]),q,y[:,:,2:]),axis=2)
    v = v.reshape(-1,v.shape[-1]).astype(float)
    return v.T@v


def train(mod):
    mod.OUT.mkdir(exist_ok=True)
    hist,folds,daily = [],[],[]
    for day in range(8,16):
        raw = mod.trial.clock.book.rawday(day)
        y = mod.trial.clock.labels(day).reshape(-1,50,3)
        refs = mod.history.references(day,raw)
        if day>=10:
            if day%2==0:
                st,group = mod.fit(hist),[]
            mats = [compare(mod,raw,p,y,st) for p in refs]
            group.append(mats)
            daily.append([metrics.gains(m) for m in mats])
            if day%2:
                folds.append([metrics.gains(m) for m in np.sum(group,axis=0)])
        hist.append(tuple(np.mean(v,axis=0) for v in zip(*[mod.record(raw,p,y) for p in refs])))
    f,d = np.asarray(folds),np.asarray(daily)
    avg = f.mean(0)
    keep = [i for i in mod.CANDIDATES if (avg[:,i]>0).all() and (f[-1,:,i]>0).all() and
            (f[:,:,i]>0).sum(0).min()>=2 and (d[:,:,i]>0).sum(0).min()>=4 and
            all((avg[:,i]>avg[:,j]).all() for j in mod.CONTROLS)]
    pick = max(keep,key=lambda i:avg[:,i].min()) if keep else 0
    (mod.OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=mod.ARMS,mean=avg.tolist(),folds=f.tolist(),daily=d.tolist())))
    print(mod.ARMS[pick],avg.round(6),flush=True)
    if pick:
        np.savez(mod.OUT/'model.npz',pick=pick,**mod.fit(hist))
    return bool(pick)


def correction(mod):
    with np.load(mod.OUT/'model.npz') as f:
        st = dict(f)
    return lambda raw,p:mod.signals(raw,p,st)[:,:,int(st['pick'])]


def replay_with(mod):
    change = correction(mod)
    def read(day,raw):
        p,h,n = mod.history.read(day,raw)
        q,part = p.copy(),h.copy()
        q[:,:,2] += change(raw,p)
        part[:,:,2] += change(raw[:n],h)
        return q,part,n
    return mod.history.stack.verifier,read


def rebuild(mod,path):
    path.mkdir(exist_ok=True)
    report = json.loads((mod.OUT/'results.json').read_text())
    pick = report['selected']
    hist,errors = [],[]
    for day in range(8,16):
        raw = mod.trial.clock.book.rawday(day)
        y = mod.trial.clock.labels(day).reshape(-1,50,3)
        preds = mod.history.references(day,raw)
        record = tuple(np.mean(v,axis=0) for v in zip(*[mod.record(raw,p,y) for p in preds]))
        if day>=10:
            if day%2==0:
                st,group = mod.fit(hist),[]
            mats = []
            for p in preds:
                old = p[:,:,2].copy()
                p[:,:,2] += mod.signals(raw,p,st)[:,:,pick]
                v = np.column_stack((np.ones(old.size),old.ravel(),p[:,:,2].ravel(),y[:,:,2].ravel()))
                mats.append(v.T@v)
            group.append(mats)
            if day%2:
                got = [metrics.gains(m)[1] for m in np.sum(group,axis=0)]
                err = float(np.max(abs(np.asarray(report['folds'])[(day-11)//2,:,pick]-got)))
                assert err<1e-8
                errors.append(err)
        np.savez(path/f'day_{day:02d}.npz',forecasts=preds,cutoff=day-day%2,raw_hash=digest(raw),forecast_hash=digest(preds))
        hist.append(record)
    with np.load(mod.OUT/'model.npz') as f:
        for k,v in mod.fit(hist).items():
            np.testing.assert_allclose(v,f[k],rtol=0,atol=1.e-8)
    (path/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))
