"""Estimate the amplitude of the bridge correction in units of forecast volatility on past dates."""
import json
import sys

import numpy as np
import torch

import experiment_joint_factor_order_calibration_v1 as calibration
import y3_interval_covariance_bridge_references as history

parent,trial,stack = history.incumbent,history.trial,history.stack
OUT = trial.clock.ART/'joint_interval_covariance_scale_v1'
ARMS = ('unchanged','one additional increment','date-median amplitude','uncertainty-shrunk pooled amplitude',
        'conservative agreement between references','separate market and relative amplitudes',
        'stock-relative amplitude','amplitude supported in both date halves')


def basis(old,p):
    z = (p[:,:,2]-old[:,:,2]).astype(float)
    market = np.broadcast_to(z.mean(1,keepdims=True),z.shape)
    return np.stack((market,z-market),axis=2)


def record(d):
    raw = trial.clock.book.rawday(d)
    refs = history.references(d,raw)
    before = history.previous.references(d,raw)
    y = trial.clock.labels(d).reshape(-1,50,3)
    rows = []
    for a,p in zip(before,refs):
        v = np.concatenate((np.ones_like(p[:,:,:1]),p,basis(a,p),y),axis=2).reshape(-1,9).astype(float)
        rows.append(v.T@v)
    return rows


def fitted(rows,k):
    cuts = np.array([[stack.covariance(s)/s[0,0] for s in day] for day in rows])
    def stats(g,direction):
        v = g[3:5,3:5]
        var = direction@v@direction
        old = g[k,k]
        cov = g[k,3:5]@direction
        beta = np.clip(g[k,5+k]/old,.3,3)
        # Partial covariance and variance after removing the current score direction.
        gain = (direction@g[3:5,5+k]-cov*g[k,5+k]/old)/beta
        risk = max(1e-20,var-cov*cov/old)
        return np.array([gain,risk,np.sqrt(old/max(var,1e-20))])
    def coefficients(direction):
        sample = np.stack([[stats(s,direction) for s in d] for d in cuts])
        r,g,ratio = sample.mean(1).T
        # Limit movement in prediction units equally for all directions and label scales.
        limit = .05*np.median(ratio)
        smooth = .5*r/g
        mean = r.mean()
        robust = mean*abs(mean)/(abs(mean)+1.5*r.std()/np.sqrt(len(r))+1e-16)
        pooled = .5*robust/g.mean()
        per_ref = sample.mean(0)
        sides = .5*per_ref[:,0]/per_ref[:,1]
        consensus = np.sign(sides[0])*min(abs(sides)) if np.prod(sides)>0 else 0.
        blocks = np.array_split(r,2)
        consistent = float(all(np.mean(part)*mean>0 for part in blocks))
        return np.clip((np.median(smooth),pooled,consensus,consistent*pooled),-limit,limit)
    ones = np.ones(2)
    whole = coefficients(ones)
    separate = np.array([coefficients(v)[1] for v in np.eye(2)])
    return np.stack((np.zeros(2),ones,whole[0]*ones,whole[1]*ones,whole[2]*ones,separate,
                     np.array([0,separate[1]]),whole[3]*ones))


def gain(m,w,k):
    c = stack.covariance(m)
    v = np.zeros((len(w),5))
    v[:,k] = 1
    v[:,3:5] = w
    corr = (v@c[:5,5+k])/np.sqrt(((v@c[:5,:5]*v).sum(1))*c[5+k,5+k])
    return corr-corr[0]


def train():
    OUT.mkdir(exist_ok=True)
    rows = [record(d) for d in range(8,16)]
    weights,picks,reports = [],[],[]
    for k in range(3):
        folds,daily = [],[]
        for n in (2,4,6):
            w = fitted(rows[:n],k)
            folds.append([gain(m,w,k) for m in np.sum(rows[n:n+2],0)])
            daily.extend([[gain(m,w,k) for m in d] for d in rows[n:n+2]])
        f,d = np.array(folds),np.array(daily)
        avg = f.mean(0)
        good = [i for i in range(1,len(ARMS)) if avg[:,i].min()>0 and f[-1,:,i].min()>0 and
                (d[:,:,i]>0).sum(0).min()>=4 and (f[:,:,i]>0).sum(0).min()>=2]
        pick = max(good,key=lambda i:avg[:,i].min()) if good else 0
        picks.append(pick)
        weights.append(fitted(rows,k)[pick])
        reports.append(dict(mean=avg.tolist(),folds=f.tolist(),daily=d.tolist()))
        print('target',k+1,ARMS[pick],avg[:,pick],weights[-1],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=picks,targets=reports)))
    if any(picks):
        np.savez(OUT/'model.npz',w=np.column_stack(weights),selected=picks)
    return any(picks)


def adjustment():
    with np.load(OUT/'model.npz') as archive:
        w = archive['w']
    return lambda old,p:p+(basis(old,p)@w).astype('float32')


def layer():
    earlier = history.previous.incumbent.layer()
    bridge = parent.correction()
    adjust = adjustment()
    def apply(raw,base):
        old = earlier(raw,base)
        p = old.copy()
        p[:,:,2] += bridge(raw,old)
        return adjust(old,p)
    return apply


def replay_with():
    adjust = adjustment()
    def read(d,raw):
        old,oh,_ = history.previous.read(d,raw)
        p,h,n = history.read(d,raw)
        return adjust(old,p),adjust(oh,h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        calibration.replay(sys.modules[__name__])
