"""Estimate the incremental value of ordered pressure events in five paired input channels."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_peer_crossing_payoff_v1 as parent

base,ref,REFERENCE,read_p = parent.base,parent.ref,parent.REFERENCE,parent.read_p
OUT = REFERENCE.parent/'dense_paired_pressure_clock_v1'
TARGETS = (2,)
ARMS = ('ordered_events','future_pressure_bridge','replicated_order')
I,J = np.triu_indices(5,1)
C,F,B = 16,46,10


def pressure(raw,ok):
    v = np.tanh(.7*np.arcsinh(raw[:,:,base.risk.walk.ref.model.FIELDS].astype(float)))
    book = v[...,::2]-v[...,1::2]
    prefix = np.concatenate((np.zeros_like(book[:1]),(book*ok[...,None]).cumsum(0)))
    t = np.arange(len(raw))
    count = np.vstack((np.zeros(50),ok.cumsum(0)))
    def mean(h):
        a = (t+1-h).clip(0)
        return (prefix[t+1]-prefix[a])/(count[t+1]-count[a]).clip(1)[...,None]
    return book,mean(3)-mean(21)


@lru_cache(maxsize=1)
def thresholds():
    estimates = []
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            _,imp = pressure(raw[a:b],ok)
            estimates.append(np.quantile(abs(imp[21::17][ok[21::17]]),.8,axis=0))
    return np.median(estimates,axis=0).clip(.015)


def panel(raw,m,ok,p,t,threshold,unit):
    book,imp = pressure(raw,ok)
    time = np.arange(len(raw))
    gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    available = (time>=20)[:,None]&(gap[time+1]==gap[(time-20).clip(0)])
    region = ((imp>threshold).astype(int)-(imp< -threshold))*available[...,None]
    prev = region[(time-1).clip(0)]
    signed = np.zeros((len(t),50,30))
    ages = {}
    for side in (-1,1):
        fire = (region==side)&(prev!=side)
        last = np.maximum.accumulate(np.where(fire,time[:,None,None],-100000),axis=0)[t]
        ages[side] = t[:,None,None]-last
    for side,age in ages.items():
        i,j = age[...,I],age[...,J]
        # Event clocks reset at each session. Both ordering and direction are observable at issuance.
        first,second = np.minimum(i,j),np.maximum(i,j)
        orientation = np.sign(i-j)
        live = np.exp(-first/12.)
        shared = np.exp(-second/32.)
        later = ages[-side]
        conflict = np.exp(-later[...,J]/12.)-np.exp(-later[...,I]/12.)
        signed += side*np.concatenate((orientation*live*(1-shared),orientation*live*shared,
                                      live*conflict),axis=2)
    path = m.cumsum(0)
    motion = np.stack([np.tanh((path[t]-path[(t-h).clip(0)])/(unit*np.sqrt(h/5))) for h in (5,20,60)],axis=2)
    ctrl = np.concatenate((p[t],book[t],imp[t],motion),axis=2)
    use = (gap[t+1]==gap[(t-60).clip(0)]) & (t>=21)[:,None]
    x = base.links.center(np.concatenate((ctrl,signed),axis=2),use)
    return x,book,path,gap


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    c = strip.physical()[1]
    err = y/beta-p
    far = err[...,2]-err[...,1]*c[2]/c[1]*beta[1]/beta[2]
    g = np.zeros((F+B+4,)*2)
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(60+day%6,b-a-60,6)
        x,book,path,gap = panel(raw[a:b],m,ok,p[a:b],t,thresholds(),base.risk.walk.widths())
        good = gap[t+61]==gap[t-60]
        future = np.concatenate((book[t+20],book[t+40]),axis=2)
        legs = np.stack(((path[t+40]-path[t+20])/beta[2],(path[t+60]-path[t+40])/beta[2]),axis=2)
        labels = np.concatenate((future,err[a+t,:,2:3],far[a+t,:,None],legs),axis=2)
        v = np.concatenate((x,base.links.center(labels,good)),axis=2)[good]
        g += v.T@v
        n += len(v)
    return g/n


def operators(dates):
    g = dates.mean(0)
    white = base.links.base.whitening
    ctrl = white(g[:C,:C],.02)
    v = np.eye(F+B+4)[:,C:F].copy()
    v[:C] = -ctrl@(ctrl.T@g[:C]@v)
    v = v@white(v.T@g@v,.8)
    pay = v.T@dates[:,:,F+B:F+B+2]
    mu = pay.mean(0)
    trust = mu**2/(mu**2+2*pay.var(0)/len(dates)+1e-9)
    direct = v[:F]@(mu[:,0]*trust[:,0]*(mu.prod(1)>0))
    return v,direct


def train(end):
    dates = np.stack([record(d) for d in range(4,end)])
    _,direct = operators(dates)
    bridged = np.zeros(F)
    halves = []
    for side in range(2):
        src,dst = dates[side::2],dates[1-side::2]
        v,part = operators(src)
        halves.append(part)
        g = dst.mean(0)
        for h in range(2):
            f = slice(F+5*h,F+5*(h+1))
            w = base.links.base.whitening(g[f,f],.6)
            teacher = w@(w.T@g[f,F+B+2+h])
            flow = v.T@src[:,:,f]@teacher
            avg = flow.mean(0)
            stability = avg**2/(avg**2+2*flow.var(0)/len(src)+1e-9)
            bridged += .5*v[:F]@(avg*stability)
    return np.r_[thresholds(),base.risk.walk.widths(),np.stack((direct,bridged,*halves)).ravel()]


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,1))
    co = bank[55:].reshape(4,F)
    for a,b,m,ok in event.sessions(raw,grid):
        x,_,_,_ = panel(raw[a:b],m,ok,p[a:b],np.arange(b-a),bank[:5],bank[5:55])
        y = x@co.T
        left,right = y[...,2],y[...,3]
        agreement = np.where(left*right>0,np.sign(left)*np.minimum(abs(left),abs(right)),0.)
        out[a:b,:,:,0] = np.stack((y[...,0],y[...,1],agreement),axis=2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
