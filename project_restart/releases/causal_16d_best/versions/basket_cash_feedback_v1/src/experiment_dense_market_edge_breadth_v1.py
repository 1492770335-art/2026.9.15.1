"""Read market-wide repricing, liquidity concentration and breadth from predicted book transitions."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_peer_impact_shrinkage_v1 as peer

direct,strip,event,onset,rel = peer.direct,peer.strip,peer.event,peer.onset,peer.rel
REFERENCE,read_p = peer.REFERENCE,peer.read_p
OUT = REFERENCE.parent / 'dense_market_edge_breadth_v1'
ARMS = ('shared_edge_shift','volatility_concentration','confirmed_breadth')
C,D = 6,33
source,dest = np.indices((5,5)).reshape(2,25)/2-1
basis = np.stack((dest, dest-source, dest*source**2),axis=1)


def features(raw,m,ok,p,net):
    f,code,vol = direct.fields(raw,m)
    out = np.zeros((len(m),D))
    out[:,:3] = p.mean(1)
    out[:,3:6] = f[...,-4:-1].mean(1)
    for start in range(0,len(m),64):
        sl = slice(start,start+64)
        pr,prior = direct.density(f[sl],code[sl],net)
        local = ((pr-prior)@basis).reshape(len(pr),50,-1)
        good = ok[sl,...,None]
        qty = good.sum(1).clip(1)
        avg = (local*good).sum(1)/qty
        vol_mean = (vol[sl,...,None]*good).sum(1)/qty
        weighted = (vol[sl,...,None]*local*good).sum(1)/qty
        def basket(offset):
            v = local[:,offset::2]
            g = good[:,offset::2]
            return (g*v).sum(1)/g.sum(1).clip(1)
        a,b = basket(0),basket(1)
        confirmed = np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
        out[sl,C:] = np.concatenate((avg*vol_mean,weighted-avg*vol_mean,confirmed*vol_mean),axis=1)
    return out


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    grams, norm = [],[]
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%4,b-a-60,4)
        x = features(raw[a:b],m,ok,p[a:b],direct.encoder())[t]
        pred = p[a+t,:,1:].astype(float)
        label = y[a+t,:,1:].astype(float)
        xx = np.column_stack((x,pred.mean(1),label.mean(1)))
        grams.append(xx)
        norm.append(np.stack((pred,label)))
    pred,label = np.concatenate(norm,axis=1)
    pred -= pred.mean((0,1))
    label -= label.mean((0,1))
    slope = (np.mean(pred*label,axis=(0,1))/np.mean(pred**2,axis=(0,1))).clip(.2,4)
    xx = np.concatenate(grams)
    xx -= xx.mean(0)
    res = xx[:,-2:]/slope-xx[:,-4:-2]
    x = xx[:,:D]
    return x.T@x/len(x),x.T@res/len(x)


def train(end):
    cov,err = map(np.stack,zip(*(record(d) for d in range(4,end))))
    g = cov.mean(0)
    coeff = []
    for ix in np.array_split(np.arange(C,D),3):
        v = np.eye(D)[:,ix]
        v[:C] = -peer.base.solve(g[:C,:C],g[:C,ix],.07)
        rhs = v.T@err
        mean = rhs.mean(0)
        reliable = mean**2/(mean**2+2*rhs.var(0)/len(rhs)+1.e-14)
        coeff.append(v@peer.base.solve(v.T@g@v,mean*reliable,.9))
    return np.r_[direct.encoder(),np.stack(coeff,axis=1).ravel()]


def signals(raw,grid,p,params):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],m,ok,p[a:b],params[:-D*6])
        z = (x@params[-D*6:].reshape(D,6)).reshape(b-a,3,2)
        out[a:b] = z[:,None,:,:]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
