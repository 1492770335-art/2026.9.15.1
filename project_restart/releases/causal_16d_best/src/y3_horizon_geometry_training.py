"""Compare Y3 corrections to both historical forecasts using expanding two-day validation folds."""
import json

import numpy as np
from scipy.signal import lfilter

import y3_dynamic_horizon_geometry_references as history

trial,stack = history.trial,history.stack


def ema(x,span):
    rho = np.exp(-1/span)
    return np.concatenate([lfilter([1-rho],[1,-rho],v,axis=0,zi=rho*v[:1])[0]
                           for v in (x[:2401],x[2401:]) if len(v)])


def records(mod,d):
    raw = trial.clock.book.rawday(d)
    y = trial.clock.labels(d).reshape(-1,50,3)
    out = []
    for p in mod.history.references(d,raw):
        z = mod.design(p,raw)
        a = np.concatenate((np.ones_like(p[:,:,:1]),p[:,:,2:],z,y[:,:,2:]),axis=2)
        a = a.reshape(-1,a.shape[-1]).astype(float)
        out.append(a.T@a)
    return out


def weights(mod,rows):
    days = np.array([np.mean([stack.covariance(m)/m[0,0] for m in d],axis=0) for d in rows])
    g = days.mean(0)
    n = len(g)-2
    v = np.vstack((-g[0,1:-1]/g[0,0],np.eye(n)))
    c = v.T@g[:-1,:-1]@v
    grad = np.array([v.T@(d[:-1,-1]/np.clip(d[0,-1]/d[0,0],.3,3)-d[:-1,0]) for d in days])
    mu = grad.mean(0)
    mu *= abs(mu)/(abs(mu)+grad.std(0)/np.sqrt(len(days))+1e-10)
    bank = []
    for ids in mod.GROUPS:
        w = np.zeros(n)
        if ids:
            i = list(ids)
            h = c[np.ix_(i,i)]
            w[i] = .5*np.linalg.solve(h+.6*np.diag(h.diagonal().clip(1e-8)),mu[i])
            w *= min(1.,.10*np.sqrt(g[0,0]/max(1e-16,w@c@w)))
        bank.append(v@w)
    return np.array(bank)


def train(mod):
    mod.OUT.mkdir(exist_ok=True)
    rows = [records(mod,d) for d in range(8,16)]
    f,ds = [],[]
    for n in (2,4,6):
        w = weights(mod,rows[:n])
        gain = history.incumbent.gain
        f.append([gain(m,w) for m in np.sum(rows[n:n+2],axis=0)])
        ds.extend([[gain(m,w) for m in day] for day in rows[n:n+2]])
    f,ds = np.array(f),np.array(ds)
    avg = f.mean(0)
    good = [i for i in mod.CANDIDATES if (avg[:,i]>0).all() and (f[-1,:,i]>0).all() and
            (f[:,:,i]>0).sum(0).min()>=2 and (ds[:,:,i]>0).sum(0).min()>=4 and
            all((avg[:,i]>avg[:,j]).all() for j in mod.CONTROLS)]
    pick = max(good,key=lambda i:avg[:,i].min()) if good else 0
    (mod.OUT/'results.json').write_text(json.dumps(dict(arms=mod.ARMS,selected=pick,mean=avg.tolist(),folds=f.tolist(),daily=ds.tolist())))
    print(mod.ARMS[pick],avg.round(6),flush=True)
    if pick:
        np.savez(mod.OUT/'model.npz',w=weights(mod,rows)[pick])
    return bool(pick)


def correction(mod):
    with np.load(mod.OUT/'model.npz') as archive:
        w = archive['w']
    return lambda raw,p:w[0]*p[:,:,2]+mod.design(p,raw)@w[1:]


def replay_with(mod):
    delta = correction(mod)
    def read(d,raw):
        p,h,n = mod.history.read(d,raw)
        q,head = p.copy(),h.copy()
        q[:,:,2] += delta(raw,p)
        head[:,:,2] += delta(raw[:n],h)
        return q,head,n
    return stack.verifier,read


def rebuild(mod,cache):
    from y3_touch_clock_references import digest
    cache.mkdir(exist_ok=True)
    report = json.loads((mod.OUT/'results.json').read_text())
    pick = report['selected']
    hist,errors = [],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        preds = mod.history.references(d,raw)
        if d>=10:
            if d%2==0:
                w = weights(mod,hist)[pick]
                group = []
            y = trial.clock.labels(d).reshape(-1,50,3)[:,:,2]
            moments = []
            for p in preds:
                a = p[:,:,2].copy()
                p[:,:,2] += w[0]*a+mod.design(p,raw)@w[1:]
                v = np.column_stack((np.ones(y.size),a.ravel(),p[:,:,2].ravel(),y.ravel()))
                moments.append(v.T@v)
            group.append(moments)
            if d%2:
                observed = []
                for m in np.sum(group,axis=0):
                    c = stack.covariance(m)
                    ic = c[:2,2]/np.sqrt(c.diagonal()[:2]*c[2,2])
                    observed.append(ic[1]-ic[0])
                err = float(np.max(abs(np.array(report['folds'])[(d-11)//2,:,pick]-observed)))
                assert err<1e-8
                errors.append(err)
        np.savez(cache/f'day_{d:02d}.npz',forecasts=preds,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(preds))
        hist.append(records(mod,d))
    with np.load(mod.OUT/'model.npz') as f:
        np.testing.assert_allclose(weights(mod,hist)[pick],f['w'],rtol=0,atol=1.e-8)
    (cache/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))
