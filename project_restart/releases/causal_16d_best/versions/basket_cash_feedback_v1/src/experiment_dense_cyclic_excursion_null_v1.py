"""Compare a realized price path with every cyclic rotation of exactly the same price increments."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_peak_reforecast_order_v1 as ref

base,strip,event,onset,rel = ref.base,ref.strip,ref.event,ref.onset,ref.rel
REFERENCE,read_p = ref.REFERENCE,ref.read_p
OUT = REFERENCE.parent/'dense_cyclic_excursion_null_v1'
ARMS = ('excess_extreme','extreme_percentile','cross_horizon_replication')
C,D = 13,25


def rotation_statistics(path):
    total = path[...,-1:]
    before = path[...,:-1]
    hi = np.maximum.accumulate(path[...,::-1],axis=-1)[...,::-1][...,:-1]
    lo = np.minimum.accumulate(path[...,::-1],axis=-1)[...,::-1][...,:-1]
    highest = np.maximum(hi,total+np.maximum.accumulate(before,axis=-1))-before
    lowest = np.minimum(lo,total+np.minimum.accumulate(before,axis=-1))-before
    result = []
    for v in (highest,-lowest):
        x = v[...,:1]
        eps = 1e-9
        order = ((v<x-eps).mean(-1)-(v>x+eps).mean(-1))
        result.append((x[...,0],v.mean(-1),order))
    return result


def features(m,ok,p):
    n = len(m)
    own = (m-(m*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1))*ok
    path = own.cumsum(0)
    gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    f = np.zeros((*m.shape,D))
    f[:,:,:3] = p
    z = strip.center(p)
    for a in range(0,n,64):
        t = np.arange(a,min(a+64,n))
        info = []
        for w in (20,40):
            origin = (t-w).clip(0)
            side = np.where(z[origin,:,1]>=0,1.,-1.)
            ix = (t[:,None]+np.arange(-w,1)).clip(0)
            cash = path[ix].transpose(0,2,1)
            cash = cash-cash[:,:,:1]
            # Normalize by the realized quadratic variation, invariant under every rotation.
            unit = np.sqrt((np.diff(cash,axis=2)**2).sum(2)+.02)
            cash = cash*side[...,None]/unit[...,None]
            (hi,havg,hr),(low,lavg,lr) = rotation_statistics(cash)
            valid = (gap[t+1]==gap[origin])&(t>=w)[:,None]
            def sign(x):
                return np.tanh(x)*valid*side
            tail = cash[:,:,-1]
            idx = 3 if w==20 else 8
            ctrl = (tail,hi,low,havg,lavg)
            f[t,:,idx:idx+5] = np.stack([sign(x) for x in ctrl],axis=2)
            info.append([sign(x) for x in (hi-havg,low-lavg,hr,lr)])
        short,long = info
        f[t,:,C:C+4] = np.stack((*short[:2],*long[:2]),axis=2)
        f[t,:,C+4:C+8] = np.stack((*short[2:],*long[2:]),axis=2)
        f[t,:,C+8:] = np.stack([np.where(x*y>0,np.sign(x)*np.minimum(abs(x),abs(y)),0.)
                                 for x,y in zip(short,long)],axis=2)
    return strip.center(f)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    gram,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%5,b-a-60,5)
        x = features(m,ok,p[a:b])[t]
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t-40]
        z = np.concatenate((x,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        gram += rows.T@rows
        n += len(rows)
    return gram/n


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    g = days.mean(0)
    slope = (days[:,[1,2],[-2,-1]]/days[:,[1,2],[1,2]]).clip(.2,4)
    err = days[:,:D,-2:]/slope[:,None]-days[:,:D,1:3]
    co = []
    for j in range(3):
        ix = np.arange(C+4*j,C+4*j+4)
        # The unrotated peak and rotation average are tested together in the controls;
        # allow for saturation and ordering differences beyond those individual levels.
        v = np.eye(D)[:,ix]
        v[:C] = -base.solve(g[:C,:C],g[:C,ix],.04)
        day = v.T@err
        mu = day.mean(0)
        st = mu**3/(mu**2+2*day.var(0)/len(days)+1e-12)
        co.append(v@base.solve(v.T@g[:D,:D]@v,st,.9))
    return np.stack(co,axis=1)


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        out[a:b] = (features(m,ok,p[a:b])@bank.reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
