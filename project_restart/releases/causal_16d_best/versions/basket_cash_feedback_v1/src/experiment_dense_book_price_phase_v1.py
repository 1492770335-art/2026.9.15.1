"""Estimate whether price follows book changes by comparing their forward and reversed lags."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import prepare_dense_cyclic_excursion_reference_v1 as best
from experiment_dense_settled_market_news_v1 import GROUP

base,strip,event,onset,rel = best.model.base,best.model.strip,best.model.event,best.model.onset,best.model.rel
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_book_price_phase_v1'
ARMS = ('own_lead_direction','persistent_lead_direction','external_lead_direction')
C,D = 9,21


def window(x,w):
    cs = np.concatenate((np.zeros_like(x[:1]),x.cumsum(axis=0)))
    t = np.arange(1,len(cs))
    return cs[t]-cs[(t-w).clip(0)]


def local_phase(book,cash,h,w):
    t = np.arange(len(book))
    earlier = (t-h).clip(0)
    left = book[earlier]*cash
    right = book*cash[earlier]
    # Pair both lag directions on exactly the same set of historical observations.
    left_var = window(book[earlier]**2,w)*window(cash**2,w)
    right_var = window(book**2,w)*window(cash[earlier]**2,w)
    lag = window(left,w)/np.sqrt(left_var+1e-5)
    lead = window(right,w)/np.sqrt(right_var+1e-5)
    return (lag-lead)*(t>=w+h)[:,None]


def features(raw,m,ok,p):
    pressure = np.tanh(base.depth(raw)@base.axis())*ok
    delta = np.diff(pressure,axis=0,prepend=pressure[:1])
    q = strip.center(p.astype(float))
    cash = (m-(m*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1))*ok
    t = np.arange(len(m))
    vol = np.sqrt(window(cash**2,80)/80+.0004)
    r = np.tanh(window(cash,5)/(vol*np.sqrt(5)))
    ctrl = (pressure,delta,window(delta,5),window(delta,15),r,q[...,2]-q[...,1])
    groups = [[],[],[]]
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    for h in (2,6):
        # The most recent book impulse is measured after the calibration window closes.
        issue = (t-h).clip(0)
        short,long = [local_phase(delta,cash,h,w)[issue]*
                      (gaps[issue+1]==gaps[(issue-w-h).clip(0)]) for w in (20,80)]
        common = np.where(short*long>0,np.sign(short)*np.minimum(abs(short),abs(long)),0.)
        ext = np.zeros_like(short)
        for k,ids in enumerate(GROUP):
            other = np.roll(GROUP,-k-1,axis=0)[:4].reshape(2,20)
            a,b = [short[:,donors].mean(1) for donors in other]
            ext[:,ids] = np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)[:,None]
        pulse = np.tanh(window(delta,h))
        for parts,phase in zip(groups,(short,common,ext)):
            parts.extend((phase*pulse,phase*r))
    return strip.center(np.concatenate((p,np.stack((*ctrl,*sum(groups,[])),axis=2)),axis=2))


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    gram,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(100+d%5,b-a-60,5)
        x = features(raw[a:b],m,ok,p[a:b])[t]
        path = m.cumsum(0)
        cash = np.stack([path[t+h]-path[t+l] for l,h in ((0,5),(5,20),(20,40),(40,60))],axis=2)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        z = np.concatenate((x,cash,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        gram += rows.T@rows
        n += len(rows)
    return gram/n


def fit(days,c,dim):
    g = days.mean(0)
    slope = (days[:,[1,2],[-2,-1]]/days[:,[1,2],[1,2]]).clip(.2,4)
    err = days[:,:dim,-2:]/slope[:,None]-days[:,:dim,1:3]
    co = []
    for j in range(3):
        ix = np.array_split(np.arange(c,dim),3)[j]
        v = np.eye(dim)[:,ix]
        v[:c] = -base.solve(g[:c,:c],g[:c,ix],.035)
        day = v.T@err
        mu = day.mean(0)
        st = mu**3/(mu**2+2*day.var(0)/len(days)+1e-14)
        co.append(v@base.solve(v.T@g[:dim,:dim]@v,st,.75))
    return np.stack(co,axis=1)


def train(end):
    return fit(np.stack([record(d) for d in range(4,end)]),C,D)


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        out[a:b] = (features(raw[a:b],m,ok,p[a:b])@bank.reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


def audit():
    out = {}
    for end in (8,12):
        tr = np.mean([record(d) for d in range(4,end)],axis=0)
        te = np.mean([record(d) for d in range(end,end+4)],axis=0)
        proj = np.eye(len(te))-np.eye(len(te))[:,:C]@base.solve(tr[:C,:C],tr[:C],.01)
        g = proj.T@te@proj
        corr = g/np.sqrt(np.outer(g.diagonal(),g.diagonal()).clip(1e-18))
        out[end+1] = {arm:corr[C+4*j:C+4*j+4,D:D+4].tolist() for j,arm in enumerate(ARMS)}
    (OUT/'payoff_bands.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
