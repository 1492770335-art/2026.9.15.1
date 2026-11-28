"""Predict the portion of the later book signal that survives the first reversal of pressure."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_local_relative_value_v1 as ref
import prepare_dense_confirmed_shock_reference_v1 as best

onset, rel, strip, event, base = ref.onset, ref.rel, ref.strip, ref.event, ref.base
REFERENCE, read_p = best.OUT, best.read_p
OUT = REFERENCE.parent / 'dense_pressure_survival_bridge_v1'
ARMS = ('uninterrupted_pressure', 'renewed_pressure', 'repriced_pressure')
K, C, D = 7, 3, 30


def fields(raw,m,ok,p):
    # All issued fields are one-sided; the teacher below is used only for fitting.
    q = np.tanh(base.depth(raw) @ base.axis())
    z = np.tanh(base.depth(raw))
    fast, slow = ref.smooth(z,5), ref.smooth(z,32)
    t = np.arange(len(m))[:,None]
    sign = np.sign(q)
    previous = np.vstack((sign[:1],sign[:-1]))
    last = np.maximum.accumulate(np.where((sign!=previous)|~ok,t,0),axis=0)
    age = np.tanh((t-last)/16)
    changed = abs(m)>.001
    last_price = np.maximum.accumulate(np.where(changed|~ok,t,0),axis=0)
    stale = np.tanh((t-last_price)/16)
    parts = (p,z,fast-slow,z*age[...,None],q[...,None]*stale[...,None],
             (q*age)[...,None],(q*(fast[...,0]-slow[...,0]))[...,None],
             (q*stale*age)[...,None],(q*abs(q))[...,None],(q*age*abs(q))[...,None])
    return strip.center(np.concatenate(parts,axis=2)), q


def teacher(t,q,m):
    direction = np.sign(q[t])
    alive = np.ones_like(direction,dtype=bool)
    moved = np.zeros_like(alive)
    y = np.zeros((*direction.shape,6))
    for h in range(1,61):
        v = q[t+h]
        alive &= direction*v>=0
        moved |= abs(m[t+h])>.001
        if h>20:
            band = int(h>40)
            y[...,band] += v*alive/20
            y[...,2+band] += v*~alive/20
            y[...,4+band] += v*moved/20
    return y


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    gram = 0.
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        x,q = fields(raw[a:b],m,ok,p[a:b])
        t = np.arange(40+d%5,b-a-60,5)
        path = m.cumsum(0)
        cash = np.stack((path[t+40]-path[t+20],path[t+60]-path[t+40]),axis=2)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        z = np.concatenate((x[t],teacher(t,q,m),cash,y[a+t,:,2:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        gram += rows.T @ rows
        n += len(rows)
    return gram/n


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    coeff = np.zeros((D,3))
    for split in (0,1):
        learn = days[split::2].mean(0)
        price = days[1-split::2]
        # Predict the future pressure outside the incumbent's three-dimensional forecast space.
        basis = np.eye(D)[:,C:]
        basis[:C] = -base.solve(learn[:C,:C],learn[:C,C:D],.03)
        g = basis.T@learn[:D,:D]@basis
        for arm in range(3):
            cols = D+arm*2+np.arange(2)
            proj = basis@base.solve(g,basis.T@learn[:D,cols],.40)
            # Fit only out-of-split projected teacher values. Price the later forty ticks.
            avg = price.mean(0)
            scale = (price[:,2,D+6:D+8].sum(1)/price[:,2,2])
            # Use a positive 60-tick conversion measured from the reported target.
            label_scale = (price[:,2,-1]/price[:,2,2]).clip(.2,4)
            cash_scale = np.sqrt(price[:,D+6,D+6].clip(.001)/price[:,-1,-1].clip(.001))
            pay = proj.T@(price[:,:D,D+6:D+8].sum(2)/cash_scale[:,None]/label_scale[:,None]).T
            # Remove the portion already present in the later payoff of the incumbent.
            pay -= proj.T@(price[:,:D,2]*scale[:,None]/cash_scale[:,None]/label_scale[:,None]).T
            mu = pay.mean(1)
            stable = mu**3/(mu**2+2*pay.var(1)/len(price)+1e-12)
            coeff[:,arm] += .5*proj@base.solve(proj.T@avg[:D,:D]@proj,stable,.9)
    return coeff


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x,_ = fields(raw[a:b],m,ok,p[a:b])
        out[a:b,:,:,1] = x@bank
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
