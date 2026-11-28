"""Estimate common price news from unexpected updates while accounting for different update rates."""
from functools import lru_cache
import sys

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit

import experiment_dense_basket_cash_feedback_v1 as base
import prepare_dense_basket_release_reference_v1 as official

strip,event,onset,rel=base.strip,base.event,base.onset,base.rel
REFERENCE,read_p=official.OUT,official.read_with_short_market
OUT=REFERENCE.parent/'dense_price_update_selection_v1'
ARMS=('unexpected_update_price','delay_adjusted_price','independent_basket_confirmation')
GROUP=np.random.default_rng(40721).permutation(50).reshape(2,25)
W=13
C=8
D=C+9

def smooth(x,n):
    decay=np.exp(-1/n)
    return lfilter([1-decay],[1,-decay],x-x[:1],axis=0)+x[:1]

def state(raw,m,ok):
    t=np.arange(len(m))[:,None]
    up=(abs(m)>.001)&ok
    recent=smooth(up.astype(float),20)
    slow=smooth(up.astype(float),100)
    age=np.log1p(t-np.maximum.accumulate(np.where(up|~ok,t,0),axis=0))/3
    flow=np.tanh(np.arcsinh(raw[...,41].astype(float))-np.arcsinh(raw[...,42].astype(float)))
    deeper=np.tanh(np.arcsinh(raw[...,54].astype(float))-np.arcsinh(raw[...,55].astype(float)))
    ret=np.tanh(smooth(m,7)*7)
    market=np.broadcast_to(ret.mean(1,keepdims=True),ret.shape)
    x=np.stack((np.ones_like(m),age,recent,slow,abs(flow),flow,deeper,ret,market,age*flow,flow*recent,
                ret*recent,slow*age),axis=-1)
    return x[np.maximum(0,np.arange(len(m))-1)],up

@lru_cache(maxsize=1)
def encoder():
    xx=[];cash=[];obs=[]
    for d in range(4):
        raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            x,up=state(raw[a:b],m,ok)
            t=np.arange(10+d,b-a,7)
            live=ok[t]&ok[t-1]
            xx.append(x[t][live]);cash.append(m[t][live]);obs.append(up[t][live])
    x,y,z=map(np.concatenate,(xx,cash,obs))
    def objective(w):
        logit=x@w
        loss=(np.logaddexp(0,logit)-z*logit).mean()+.001*(w[1:]@w[1:])
        grad=x.T@(expit(logit)-z)/len(z)
        grad[1:]+=.002*w[1:]
        return loss,grad
    fit=minimize(objective,np.zeros(W),jac=True,method='L-BFGS-B',options={'maxiter':140})
    if not fit.success:raise RuntimeError(fit.message)
    xm=x[z];g=xm.T@xm/len(xm)
    sign=np.linalg.solve(g+.12*np.diag(g.diagonal().clip(1e-5)),xm.T@y[z]/len(xm))
    return np.r_[fit.x,sign]

def design(raw,m,ok,p,weights):
    x,hit=state(raw,m,ok)
    propensity=expit(x@weights[:W]).clip(.08,.95)
    mean=x@weights[W:]
    surprise=(m-mean)*hit
    valid=ok&np.vstack((np.zeros((1,50),bool),ok[:-1]))
    def average(v):
        z=(v*valid)[:,GROUP].sum(2)/valid[:,GROUP].sum(2).clip(1)
        return z
    regular=average(surprise)
    adjusted=average(surprise*(1/propensity-1))
    # Compare independent groups at the same time after smoothing price news.
    controls=np.column_stack((p.mean(1,dtype=float),m.mean(1),smooth(m.mean(1),20),
                              mean.mean(1),regular.mean(1),smooth(regular.mean(1),32)))
    outputs=[]
    for tau in (3,18,80):
        u,v=smooth(regular,tau),smooth(adjusted,tau)
        agreement=np.where(v[:,0]*v[:,1]>0,np.sign(v[:,0])*np.minimum(abs(v[:,0]),abs(v[:,1])),0)
        outputs.append(np.column_stack((u.mean(1),v.mean(1),agreement)))
    return np.column_stack((controls,*outputs))

@lru_cache(maxsize=12)
def record(d):
    raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
    p=read_p(d)
    q=p[...,1:].astype(float);v=y[...,1:].astype(float)
    q-=q.mean((0,1));v-=v.mean((0,1))
    slope=((q*v).mean((0,1))/(q*q).mean((0,1))).clip(.2,4)
    err=(v/slope-q).mean(1)
    parts=[]
    for a,b,m,ok in event.sessions(raw,grid):
        t=np.arange(20+d%4,b-a-60,4)
        x=design(raw[a:b],m,ok,p[a:b],encoder())
        parts.append(np.column_stack((x[t],err[a+t])))
    z=np.concatenate(parts)
    mean=z.mean(0);z-=mean
    return z.T@z/len(z),mean[:D]

def train(end):
    gs,means=map(np.stack,zip(*(record(d) for d in range(4,end))))
    g=gs.mean(0)[:D,:D]
    def solve(a,b,k):
        return np.linalg.solve(a+k*np.diag(a.diagonal().clip(1e-9)),b)
    co=[]
    for j in range(3):
        ids=C+j+3*np.arange(3)
        nuisance=np.arange(C)
        if j:nuisance=np.r_[nuisance,C+3*np.arange(3)]
        v=np.eye(D)[:,ids]
        v[nuisance]=-solve(g[np.ix_(nuisance,nuisance)],g[np.ix_(nuisance,ids)],.08)
        r=v.T@gs[:,:D,D:]
        avg=r.mean(0)
        stable=avg**3/(avg**2+3*r.var(0)/len(r)+1e-12)
        co.append(v@solve(v.T@g@v,stable,.85))
    return np.r_[np.stack(co,axis=1).ravel(),means.mean(0),encoder()]

def signals(raw,grid,p,bank):
    co=bank[:D*6].reshape(D,6)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x=design(raw[a:b],m,ok,p[a:b],bank[D*7:])
        z=((x-bank[D*6:D*7])@co).reshape(b-a,3,2)
        out[a:b]=z[:,None]
    return out

def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)

if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
