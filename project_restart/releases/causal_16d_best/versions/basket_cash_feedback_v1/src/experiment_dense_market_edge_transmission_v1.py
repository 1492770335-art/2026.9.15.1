"""Price common book transitions by measuring payment in other stock baskets on training dates."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_market_edge_breadth_v1 as market

peer,direct,strip,event,onset,rel = market.peer,market.direct,market.strip,market.event,market.onset,market.rel
REFERENCE,read_p = market.REFERENCE,market.read_p
OUT = REFERENCE.parent/'dense_market_edge_transmission_v1'
ARMS = ('other_basket_cash','external_cash_after_own_book','receiver_response')
C,D = 6,15
GROUP = np.random.default_rng(24810).permutation(50).reshape(5,10)
LOADING = (3,5,12)


def half(x):
    return x[...,:12]-x[...,::-1][...,:12]


@lru_cache(maxsize=16)
def price_data(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    gram = np.zeros((5,25,25))
    for a,b,m,ok in event.sessions(raw,grid):
        f,code,vol = direct.fields(raw[a:b],m)
        t = np.arange(40+day%5,len(m)-5,5)
        path = m.cumsum(0)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = (gap[t+6]==gap[t])[:,GROUP]
        qty = valid.sum(2).clip(1)
        idx = (5*code[t]+code[t+5])[:,GROUP]
        realized = half(np.eye(25)[idx])
        own = (realized*(vol[t][:,GROUP]*valid)[...,None]).sum(2)/qty[...,None]
        market_x = (own.sum(1,keepdims=True)-own)/4
        cash = ((path[t+5]-path[t])[:,GROUP]*valid).sum(2)/qty
        z = np.concatenate((market_x,own,cash[...,None]),axis=2)
        gram += np.einsum('tgi,tgj->gij',z,z,optimize=True)/len(t)
    return gram


def laws(end):
    def estimate(g):
        h = g.mean(0)
        def external(mat,partial):
            if not partial:
                return peer.base.solve(mat[:12,:12],mat[:12,-1],.3)
            # Remove the recipient's simultaneous book changes from the donor basket.
            v = np.eye(24)[:,:12]
            v[12:] = -peer.base.solve(mat[12:24,12:24],mat[12:24,:12],.07)
            return peer.base.solve(v.T@mat[:24,:24]@v,v.T@mat[:24,-1],.45)
        common,unique = external(h,False),external(h,True)
        local = np.stack([external(mat,True)-unique for mat in g])
        return np.stack(np.broadcast_arrays(common,unique,local))
    a,b = [estimate(np.mean([price_data(d) for d in range(s,end,2)],0)) for s in (0,1)]
    return np.where(a*b>0,(a+b)/2,0.)


def features(raw,m,ok,p,net,law):
    f,code,vol = direct.fields(raw,m)
    out = np.zeros((len(m),D))
    out[:,:3] = p.mean(1)
    out[:,3:6] = f[...,-4:-1].mean(1)
    for start in range(0,len(m),64):
        sl = slice(start,start+64)
        pr,prior = direct.density(f[sl],code[sl],net)
        local = half(pr-prior)*vol[sl,...,None,None]
        good = ok[sl][:,GROUP]
        own = (local[:,GROUP]*good[...,None,None]).sum(2)/good.sum(2).clip(1)[...,None,None]
        other = (own.sum(1,keepdims=True)-own)/4
        for k in range(3):
            predicted = (other*law[k,:,None,:]).sum(-1).mean(1)
            out[sl,C+3*k:C+3*k+3] = predicted*np.array([1,3,8])
    return out


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    law = laws(day)
    observations,labels,guesses = [],[],[]
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%4,b-a-60,4)
        observations.append(features(raw[a:b],m,ok,p[a:b],direct.encoder(),law)[t])
        labels.append(y[a+t,:,1:].astype(float))
        guesses.append(p[a+t,:,1:].astype(float))
    x,y,p = map(np.concatenate,(observations,labels,guesses))
    x -= x.mean(0)
    y -= y.mean((0,1))
    p -= p.mean((0,1))
    slope = ((p*y).sum((0,1))/(p*p).sum((0,1))).clip(.2,4)
    err = (y/slope-p).mean(1)
    return x.T@x/len(x),x.T@err/len(x)


def train(end):
    cov,err = map(np.stack,zip(*(record(d) for d in range(4,end))))
    g = cov.mean(0)
    co = []
    for ix in np.array_split(np.arange(C,D),3):
        v = np.eye(D)[:,ix]
        v[:C] = -peer.base.solve(g[:C,:C],g[:C,ix],.07)
        rhs = v.T@err
        avg = rhs.mean(0)
        safe = avg**2/(avg**2+2*rhs.var(0)/len(rhs)+1e-14)
        co.append(v@peer.base.solve(v.T@g@v,avg*safe,.9))
    return np.r_[laws(end).ravel(),direct.encoder(),np.stack(co,axis=1).ravel()]


def signals(raw,grid,p,params):
    cut = int(np.prod(LOADING))
    law = params[:cut].reshape(LOADING)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],m,ok,p[a:b],params[cut:-D*6],law)
        out[a:b] = (x@params[-D*6:].reshape(D,6)).reshape(b-a,1,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
