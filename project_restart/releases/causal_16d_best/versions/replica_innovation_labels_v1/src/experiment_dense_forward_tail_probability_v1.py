"""Gate forward-leg payoff regressions with an ex-ante probability of a large relative move."""
from functools import lru_cache
import json
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_paired_pressure_clock_v1 as book
import audit_dense_horizon_leg_noise_v1 as audit

base,ref,REFERENCE,read_p = audit.curve.base,audit.curve.ref,audit.curve.REFERENCE,audit.curve.read_p
OUT = REFERENCE.parent/'dense_forward_tail_probability_v1'
ARMS = ('tail_probability_update','replicated_tail_update','separate_payoffs')
TARGETS = (1,2)
R,C = 15,14


def features(raw,m,ok,t,unit):
    path = (m/unit).cumsum(0)
    count = np.vstack((np.zeros(50),ok.cumsum(0)))
    a = np.vstack((np.zeros(50),(abs(m/unit)*ok).cumsum(0)))
    sq = np.vstack((np.zeros(50),((m/unit)**2*ok).cumsum(0)))
    risk = [np.ones_like(m[t])]
    for h in (12,60,240):
        lo = (t+1-h).clip(0)
        n = (count[t+1]-count[lo]).clip(1)
        vol = np.sqrt(((sq[t+1]-sq[lo])/n).clip(0))
        drift = abs(path[t]-path[lo])/np.sqrt(h)
        volume = (a[t+1]-a[lo])/n
        global_vol = np.broadcast_to(vol.mean(1,keepdims=True),vol.shape)
        risk.extend(np.log1p(v) for v in (vol,drift,volume,global_vol))
    levels,impulse = book.pressure(raw,ok)
    risk.extend((abs(levels[t]).mean(2),abs(impulse[t]).mean(2)))
    return np.stack(risk,axis=2),levels[t],impulse[t]


@lru_cache(maxsize=16)
def examples(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    data = []
    unit = base.risk.walk.widths()
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(60+day%9,b-a-60,9)
        path = m.cumsum(0)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        live = gap[t+61]==gap[t-40]
        y = np.stack((path[t+20]-path[t+5],path[t+60]-path[t+20]),axis=2)
        y = base.links.center(y,live)
        x,levels,impulse = features(raw[a:b],m,ok,t,unit)
        data.append((a+t,live,y,x,levels,impulse))
    return data


@lru_cache(maxsize=1)
def cuts():
    scale = base.risk.walk.widths()[:,None]
    values = [(abs(y)/scale)[live] for d in range(4) for _,live,y,*_ in examples(d)]
    return np.quantile(np.concatenate(values),.9,axis=0)


@lru_cache(maxsize=16)
def risk_fit(end):
    xx,yy = [],[]
    for d in range(end):
        for _,live,y,x,*_ in examples(d):
            xx.append(x[live][::7])
            yy.append((abs(y)/base.risk.walk.widths()[:,None]>cuts())[live][::7])
    x,y = np.concatenate(xx),np.concatenate(yy)
    weights = np.zeros((R,2))
    for j in range(2):
        w = weights[:,j]
        for _ in range(9):
            p = expit(x@w)
            g = (x*(p*(1-p))[:,None]).T@x+40*np.eye(R)
            w += np.linalg.solve(g,x.T@(y[:,j]-p)-40*w)
    return weights


def direction(p,levels,impulse):
    return np.concatenate((p,levels,impulse,(p[:,:,2]-p[:,:,1])[...,None]*abs(levels).mean(2,keepdims=True)),axis=2)


@lru_cache(maxsize=12)
def record(day):
    p,y = [read_p(day),onset.settled.load(day,'y')]
    error = y/strip.daily_scale(p,y).clip(.2,4)-p
    grams = np.zeros((2,2,C+2,C+2))
    counts = np.zeros((2,2))
    for t,live,physical,_,levels,impulse in examples(day):
        x = base.links.center(direction(p[t],levels,impulse),live)
        err = base.links.center(error[t,:,1:],live)
        large = abs(physical)/base.risk.walk.widths()[:,None]>cuts()
        for h in range(2):
            z = np.concatenate((x,err[:,:,h:h+1],physical[:,:,h:h+1]),axis=2)
            for k in (0,1):
                sub = z[live&(large[:,:,h]==k)]
                grams[h,k] += sub.T@sub
                counts[h,k] += len(sub)
    return grams/counts[:,:,None,None].clip(1)


def payoffs(ds):
    g = ds.mean(0)
    gram = g[:C,:C]
    sd = gram.diagonal().clip(1e-6)**.5
    eig,v = np.linalg.eigh(gram/np.outer(sd,sd))
    axes = (v/np.sqrt(eig.clip(0)+.4))/sd[:,None]
    b = axes.T@ds[:,:C,C:]
    mean = b.mean(0)
    w = mean*mean**2/(mean**2+2*b.var(0)/len(ds)+1e-9)
    return axes@(w[:,0]*(w.prod(1)>0))


def train(end):
    ds = np.stack([record(d) for d in range(4,end)])
    co = []
    for j in range(2):
        w = []
        for subset in (ds,ds[::2],ds[1::2]):
            w.extend([payoffs(subset[:,j,k]) for k in (0,1)])
        co.append(w)
    return np.r_[base.risk.walk.widths(),risk_fit(end).ravel(),np.stack(co).ravel()]


def signals(raw,grid,p,bank):
    unit,w = bank[:50],bank[50:50+R*2].reshape(R,2)
    co = bank[50+R*2:].reshape(2,6,C)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        rr,levels,imp = features(raw[a:b],m,ok,np.arange(b-a),unit)
        chance = expit(rr@w)
        x = base.links.center(direction(p[a:b],levels,imp),ok)
        for j in range(2):
            price = x@co[j].T
            prob = chance[:,:,j]
            def shift(k):
                return (prob-.10)*(price[:,:,k+1]-price[:,:,k])
            first,second = shift(2),shift(4)
            stable = np.where(first*second>0,np.sign(first)*np.minimum(abs(first),abs(second)),0.)
            y = np.stack((shift(0),stable,(1-prob)*price[:,:,0]+prob*price[:,:,1]),axis=2)
            out[a:b,:,:,j] = base.links.center(y,ok)
    return out


def audit_risk():
    result = {}
    for start in (8,12):
        labels,forecast,baseline = [],[],[]
        for d in range(start,start+4):
            for _,live,y,x,*_ in examples(d):
                labels.append((abs(y[:,:,1])/base.risk.walk.widths()>cuts()[1])[live])
                forecast.append(expit(x[live]@risk_fit(start)[:,1]))
                baseline.append(x[live][:,5]) # realized volatility over the last 60 ticks
        y,p,vol = map(np.concatenate,(labels,forecast,baseline))
        result[start+1] = dict(auc=roc_auc_score(y,p),past_volatility_auc=roc_auc_score(y,vol),
                               prevalence=y.mean(),top_decile_rate=y[p>=np.quantile(p,.9)].mean())
    (OUT/'tail_detection.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
    audit_risk()
