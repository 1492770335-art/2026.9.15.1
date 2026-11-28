"""Compare current book pressure to previous visits to a similar reconstructed price."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
from experiment_dense_lag_resolved_book_links_v1 import REFERENCE,read_p,links


OUT = REFERENCE.parent/'dense_revisited_price_pressure_v1'
ARMS = ('calendar_pressure','price_matched_pressure','after_price_excursion')
LAGS = tuple(range(8,129,8))
FIELDS = [41,42,43,44,50,51,54,55,77,78]
C = 13
D = 6


def states(raw,m,ok,p,t):
    b = np.tanh(.7*np.arcsinh(raw[:,:,FIELDS].astype(float)))
    pressure = b[...,::2]-b[...,1::2]
    path = m.cumsum(0)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    energy = np.vstack((np.zeros(50),(m*m).cumsum(0)))
    sd = np.sqrt((energy[t+1]-energy[(t-63).clip(0)]).clip(0)/np.minimum(t+1,64)[:,None]+.0004)
    sums = np.concatenate((np.zeros_like(pressure[:1]),pressure.cumsum(0)))
    sm = (sums[t+1]-sums[(t-31).clip(0)])/np.minimum(32,t+1)[:,None,None]
    ctrl = links.center(np.concatenate((p[t],pressure[t],sm),axis=2),ok[t])
    n = np.zeros((len(t),50,3))
    numer = np.zeros((len(t),50,3,D))
    for lag in LAGS:
        old = (t-lag).clip(0)
        mid = (t-lag//2).clip(0)
        valid = (gaps[t+1]==gaps[old])&(t>=lag)[:,None]
        difference = (path[t]-path[old])/sd
        excursion = (path[mid]-.5*(path[t]+path[old]))/(sd*np.sqrt(lag)*.5)
        calendar = np.exp(-lag/72.)*valid
        matched = calendar*np.exp(-.5*(difference/1.5)**2)
        away = matched*np.tanh(abs(excursion))**2
        mass = np.stack((calendar,matched,away),axis=2)
        value = np.concatenate((pressure[t]-pressure[old],np.tanh(excursion)[...,None]),axis=2)
        n += mass
        numer += mass[...,None]*value[:,:,None,:]
    summaries = numer/(n[...,None]+.65)
    # Compare the price-selected histories against ordinary temporal pressure change.
    calendar = summaries[:,:,0].copy()
    summaries[:,:,1:] -= calendar[:,:,None,:]*(n[:,:,1:]/(n[:,:,1:]+.65))[...,None]
    return ctrl,[links.center(summaries[:,:,k],ok[t]) for k in range(3)]


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    c = strip.physical()[1]
    g = np.zeros((2,3,C+D+2,C+D+2))
    count = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(32+day%5,b-a-60,5)
        ctrl,parts = states(raw[a:b],m,ok,p[a:b],t)
        e = y[a+t].astype(float)/beta-p[a+t]
        for h in range(2):
            j = h+1
            far = e[:,:,j]-e[:,:,h]*c[j]/c[h]*beta[h]/beta[j]
            labels = links.center(np.stack((e[:,:,j],far),axis=2),ok[t])
            for k,part in enumerate(parts):
                x = np.concatenate((ctrl,part,labels),axis=2)[ok[t]]
                g[h,k] += x.T@x
        count += ok[t].sum()
    return g/count


def train(end):
    gs = np.stack([record(d) for d in range(4,end)])
    co = np.zeros((2,3,C+D))
    for h in range(2):
        for k in range(3):
            rows = gs[:,h,k]
            g = rows.mean(0)
            white = links.base.whitening(g[:C,:C],.03)
            v = np.eye(C+D+2)[:,C:C+D].copy()
            v[:C] = -white@(white.T@g[:C]@v)
            v = v@links.base.whitening(v.T@g@v,.75)
            grad = v.T@rows[:,:,-2:]
            avg = grad.mean(0)
            total,far = (avg*avg**2/(avg**2+grad.var(0)/len(rows)+1e-9)).T
            co[h,k] = (v@(total*(total*far>0)))[:-2]
    return co


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        ctrl,parts = states(raw[a:b],m,ok,p[a:b],np.arange(b-a))
        for lo in range(0,b-a,240):
            t = np.arange(lo,min(lo+240,b-a))
            for k,part in enumerate(parts):
                x = np.concatenate((ctrl[t],part[t]),axis=2)
                out[a+t,:,k] = x@bank[:,k].T
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
