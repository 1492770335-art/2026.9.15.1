"""Estimate unique short-horizon forecast information using the other stocks' current horizon geometry."""
import json
import sys

import numpy as np
import torch
from scipy.signal import lfilter

import experiment_joint_factor_order_calibration_v1 as replay
import joint_interval_covariance_scale_references as history

trial,stack,parent = history.trial,history.stack,history.incumbent
OUT = trial.clock.ART/'y3_dynamic_horizon_geometry_v1'
ARMS = ('unchanged','static short forecasts','moving unique short information','instantaneous unique information',
        'local versus running exposure','decorrelated middle forecast','consensus-weighted short forecasts',
        'changes in consensus','disjoint-stock replication','reassigned receiver control')
GROUPS = ((),(0,1),(2,3),(4,5),(6,7),(8,),(9,10),(11,12),(2,3,13,14),(15,16))


def design(p):
    v = trial.clock.center(p.astype(float))
    n = len(v)
    cols = np.empty((n,50,17),dtype='float32')
    cols[:,:,:2] = v[:,:,:2]
    for side in (0,1):
        past,now = v[:,side::2],v[:,1-side::2]
        gram = past.transpose(0,2,1)@past/25
        friend = now.transpose(0,2,1)@now/25
        def running(g):
            result = []
            for x in (g[:2401],g[2401:]):
                if len(x):
                    rho = np.exp(-1/96)
                    result.append(lfilter([1-rho],[1,-rho],x,axis=0,zi=(rho*x[:1]))[0])
            return np.concatenate(result)
        slow = running(gram)
        def split(g):
            var = g.diagonal(axis1=1,axis2=2)
            cov = g[:,:2,2]
            slope = cov/(var[:,2:]+.003)
            value = now[:,:,:2]-slope[:,None,:]*now[:,:,2:]
            cc = cov/np.sqrt((var[:,:2]+.003)*(var[:,2:]+.003))
            return value,cc
        stable,sc = split(slow)
        instant,ic = split(.75*gram+.25*slow)
        other,oc = split(.75*friend+.25*running(friend))
        pair = np.array([0,2])
        tmp = .75*gram+.25*slow
        h = tmp[:,pair][:,:,pair]
        h += np.eye(2)*(.003+.08*h.diagonal(axis1=1,axis2=2))[:,None,:]
        w = np.linalg.solve(h,tmp[:,pair,1,None])[:,:,0]
        unique = now[:,:,1]-(now[:,:,pair]*w[:,None]).sum(2)
        common = np.sign(instant)*np.minimum(abs(instant),abs(other))*(instant*other>0)
        info = (1-ic**2)[:,None,:]
        changing = (sc**2-ic**2)[:,None,:]
        parts = (stable,instant,instant-stable,unique[:,:,None],info*instant,changing*instant,common,np.roll(instant,8,axis=1))
        cols[:,1-side::2,2:] = np.concatenate(parts,axis=2)
    return cols


def records(d):
    raw = trial.clock.book.rawday(d)
    y = trial.clock.labels(d).reshape(-1,50,3)
    rows = []
    for p in history.references(d,raw):
        z = design(p)
        a = np.concatenate((np.ones_like(p[:,:,:1]),p[:,:,2:],z,y[:,:,2:]),axis=2).reshape(-1,20).astype(float)
        rows.append(a.T@a)
    return rows


def weights(rows):
    days = np.stack([np.mean([stack.covariance(m)/m[0,0] for m in day],axis=0) for day in rows])
    g = days.mean(0)
    basis = np.vstack((-g[0,1:-1]/g[0,0],np.eye(17)))
    grad = np.array([basis.T@(c[:-1,-1]/np.clip(c[0,-1]/c[0,0],.3,3)-c[:-1,0]) for c in days])
    cov = basis.T@g[:-1,:-1]@basis
    mu = grad.mean(0)
    mu *= abs(mu)/(abs(mu)+grad.std(0)/len(grad)**.5+1.e-10)
    bank = []
    for group in GROUPS:
        w = np.zeros(17)
        if group:
            ids = list(group)
            h = cov[np.ix_(ids,ids)]
            w[ids] = .5*np.linalg.solve(h+.6*np.diag(h.diagonal().clip(1.e-6)),mu[ids])
            w *= min(1.,.1*np.sqrt(g[0,0]/max(1.e-14,w@cov@w)))
        bank.append(basis@w)
    return np.stack(bank)


def gain(m,w):
    c = stack.covariance(m)
    v = w.copy()
    v[:,0] += 1
    ic = v@c[:-1,-1]/np.sqrt((v@c[:-1,:-1]*v).sum(1)*c[-1,-1])
    return ic-ic[0]


def train():
    OUT.mkdir(exist_ok=True)
    rows = [records(d) for d in range(8,16)]
    folds,days = [],[]
    for n in (2,4,6):
        w = weights(rows[:n])
        folds.append([gain(m,w) for m in np.sum(rows[n:n+2],axis=0)])
        days.extend([[gain(m,w) for m in day] for day in rows[n:n+2]])
    f,ds = np.asarray(folds),np.asarray(days)
    avg = f.mean(0)
    good = [i for i in range(2,9) if (avg[:,i]>0).all() and (f[-1,:,i]>0).all() and
            (f[:,:,i]>0).sum(0).min()>=2 and (ds[:,:,i]>0).sum(0).min()>=4 and
            all((avg[:,i]>avg[:,j]).all() for j in (1,9))]
    pick = max(good,key=lambda i:avg[:,i].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,mean=avg.tolist(),folds=f.tolist(),daily=ds.tolist())))
    print(ARMS[pick],avg,flush=True)
    if pick:
        np.savez(OUT/'model.npz',w=weights(rows)[pick])
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        w = f['w']
    return lambda raw,p:w[0]*p[:,:,2]+design(p)@w[1:]


def layer():
    return history.study.layer(sys.modules[__name__],parent)


def replay_with():
    change = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        q,head = p.copy(),h.copy()
        q[:,:,2] += change(raw,p)
        head[:,:,2] += change(raw[:n],h)
        return q,head,n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay.replay(sys.modules[__name__])
