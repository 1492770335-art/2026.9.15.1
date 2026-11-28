"""Invert one-block return memory and carry forward only the part of an issued forecast attributed to new shocks."""
from functools import lru_cache
import sys
import json

import numpy as np
from scipy.signal import lfilter
import torch

import experiment_dense_waiting_book_witnesses_v1 as base

event,strip,rel,onset,REFERENCE,read_p = base.event,base.strip,base.rel,base.onset,base.REFERENCE,base.read_p
OUT = REFERENCE.parent/'dense_market_one_block_memory_v1'
ARMS = ('five_tick_memory','twenty_tick_memory','cross_basket_confirmation')
SPLIT = np.random.default_rng(91652).permutation(50).reshape(2,25)
H = (5,20,60)


def baskets(x,ok):
    return np.stack([(x[:,g]*ok[:,g]).sum(1)/ok[:,g].sum(1).clip(1) for g in SPLIT],axis=1)


def window(x,h):
    cs = np.vstack((np.zeros_like(x[:1]),x.cumsum(0)))
    t = np.arange(1,len(x)+1)
    return cs[t]-cs[(t-h).clip(0)]


@lru_cache(maxsize=16)
def stats(d):
    raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
    p = read_p(d) if d>=4 else None
    cov = np.zeros((2,4,4))
    market = np.zeros((3,3,3))
    beta = np.zeros((2,50))
    for a,b,m,ok in event.sessions(raw,grid):
        flow = baskets(m,ok)
        for j,h in enumerate(H):
            z = window(flow,h)
            t = np.arange(2*h+d%3,b-a-h,3)
            if j<2:
                f = np.concatenate((z[t],z[t-h]),axis=1)
                cov[j] += np.cov(f.T,bias=True)
            if j==2:
                cash = window(m,h)[t]
                external = np.empty_like(cash)
                for k,ids in enumerate(SPLIT):
                    external[:,ids] = z[t,1-k,None]
                beta[0] += ((cash-cash.mean(0))*(external-external.mean(0))).mean(0)
                beta[1] += ((external-external.mean(0))**2).mean(0)
            if p is not None:
                pred = baskets(p[a:b,:,j],ok)[t].mean(1)
                real = z[t+h].mean(1)
                f = np.stack((np.ones_like(pred),pred,real),axis=1)
                market[j] += f.T@f/len(f)
    return cov,market,beta


def train(end):
    c = np.mean([stats(d)[0] for d in range(end)],axis=0)
    # Cross-basket lag moments reduce noise specific to one stock. The stable invertible root has magnitude below one.
    r = ((c[:,0,3]+c[:,1,2])/(2*c[:,0,1]).clip(1e-5)).clip(-.46,.46)
    theta = 2*r/(1+np.sqrt(1-4*r*r))
    g = np.mean([stats(d)[1] for d in range(4,end)],axis=0)
    g /= g[:,:1,:1].copy()
    v = (g[:,1,1]-g[:,0,1]**2).clip(.0005)
    u = ((g[:,1,2]-g[:,0,1]*g[:,0,2])/v).clip(.05,6.)
    beta = np.mean([stats(d)[2] for d in range(end)],axis=0)
    b = ((beta[0]+.5*beta[0].mean())/(beta[1]+.5*beta[1].mean())).clip(.2,3)
    return np.r_[theta,u,v,b/b.mean()]


def estimates(m,ok,p,bank):
    theta,prior,var = bank[:2],bank[2:5],bank[5:8]
    n = len(m)
    t = np.arange(n)
    flow = baskets(m,ok)
    units,fore,observed = [],[],[]
    for j,h in enumerate(H):
        y = window(flow,h)
        f = baskets(p[...,j],ok)
        s = (t-h).clip(0)
        # Update the conversion between network and cash units only when a forecast has matured.
        x = f[s].mean(1)
        z = y.mean(1)
        valid = t>=h
        def ew(q):
            return lfilter([1.],[1,-np.exp(-1/520)],q*valid)
        mass = ew(np.ones(n))
        den = ew(x*x)-ew(x)**2/(mass+1)
        cov = ew(x*z)-ew(x)*ew(z)/(mass+1)
        unit = ((cov+180*var[j]*prior[j])/(den+180*var[j])).clip(.05,6)
        units.append(unit)
        fore.append(f*unit[:,None])
        observed.append(y)
    signals = []
    for j,h in enumerate(H[:2]):
        shock = observed[j].copy()
        for phase in range(h):
            shock[phase::h] = lfilter([1.],[1.,theta[j]],shock[phase::h],axis=0)
        expected_new = fore[j]-theta[j]*shock
        total = fore[j]+theta[j]*expected_new
        signals.append((total-fore[2])/units[2][:,None])
    a,b = (signals[1][:,k] for k in range(2))
    agreement = np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
    return np.column_stack((signals[0].mean(1),signals[1].mean(1),agreement))


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        z = estimates(m,ok,p[a:b],bank)
        out[a:b,:,:,1] = z[:,None]*bank[None,8:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    report = {d:train(d)[:5].tolist() for d in (6,8,12,16)}
    (OUT/'market_dynamics.json').write_text(json.dumps(report,indent=2))
