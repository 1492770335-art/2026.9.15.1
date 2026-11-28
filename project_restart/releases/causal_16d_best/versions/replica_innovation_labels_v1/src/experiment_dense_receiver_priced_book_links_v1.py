"""Estimate stock-specific prices of future book innovations and use them to select incoming peer links."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import REFERENCE, read_p, event, onset, rel, strip
import experiment_dense_peer_book_response_graph_v1 as links


OUT = REFERENCE.parent / 'dense_receiver_priced_book_links_v1'
ARMS = ('pooled_receiver_price','receiver_specific_price','price_before_graph_selection')
D = links.D
C = links.C


@lru_cache(maxsize=16)
def book_cash(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    sums = np.zeros((2,50,D+1,D+1))
    counts = np.zeros(50)
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+day%6,b-a-60,6)
        fast,delta,mean = links.raw_view(raw[a:b],ok,t,links.base.coordinates()[:,-D:])
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+61]==gaps[t-34]
        for h,(near,full) in enumerate(links.base.BANDS):
            later = mean(t+near-4,t+near+1)-fast
            cash = path[t+full]-path[t+near]
            marks = links.center(np.concatenate((later,cash[...,None]),axis=2),valid)
            own = links.center(np.concatenate((fast,delta,(path[t+near]-path[t])[...,None]),axis=2),valid)
            x = own[valid]
            white = links.base.whitening(x.T@x/len(x),.01)
            fit = white@(white.T@(x.T@marks[valid]/len(x)))
            v = (marks-own@fit).transpose(1,0,2)
            sums[h] += v.transpose(0,2,1)@v
        counts += valid.sum(0)
    return sums/counts.clip(1)[None,:,None,None]


def receiver_prices(rows):
    g = rows.mean(0)
    shared = g.mean(0)
    root = links.base.whitening(shared[:D,:D],.65)
    pooled = root@(root.T@shared[:D,-1])
    price = []
    for i,stock in enumerate(g):
        white = links.base.whitening(.5*stock[:D,:D]+.5*shared[:D,:D],.65)
        gradients = rows[:,i,:D,-1] - rows[:,i,:D,:D]@pooled
        value = gradients@white
        mu = value.mean(0)
        trust = mu**2/(mu**2+value.var(0)/len(rows)+.00002)
        price.append(pooled + white@(mu*trust))
    return pooled,np.stack(price)


@lru_cache(maxsize=16)
def maps(end):
    days = np.stack([book_cash(d) for d in range(end)])
    supervised = np.stack([links.relation(d) for d in range(end)])
    output = []
    for h in range(2):
        structured = links.graphs(end)[h].reshape(50,D,50,D)
        pooled,individual = receiver_prices(days[:,h])
        common = np.einsum('sird,d->sir',structured,pooled)
        local = np.einsum('sird,rd->sir',structured,individual)
        halves = []
        for k in range(2):
            response = supervised[k::2,h].mean(0).reshape(50,50,D,D)
            _,price = receiver_prices(days[1-k::2,h])
            halves.append(np.einsum('srid,rd->sri',response,price).reshape(2500,D))
        a,b = halves
        vals,u = np.linalg.eigh(.5*(a.T@b+b.T@a))
        noise = .25*np.square((a-b)@u).sum(0)
        weight = vals.clip(0)/(vals.clip(0)+noise+.000001)
        prioritized = (((a+b)*.5)@u*weight)@u.T
        output.append(np.stack((common,local,prioritized.reshape(50,50,D).transpose(0,2,1)),axis=-1))
    return np.stack(output)


def features(raw,ok,p,t,axes,mats):
    fast,delta,_ = links.raw_view(raw,ok,t,axes)
    valid = ok[t] & (t>=34)[:,None]
    ctrl = links.center(np.concatenate((fast,delta,p[t]),axis=2),valid)
    x = links.center(delta,valid).reshape(len(t),50*D)
    for w in mats:
        z = (x@w.reshape(50*D,150)).reshape(len(t),50,3)
        yield np.concatenate((ctrl,links.center(z,valid)),axis=2)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    coeff = strip.physical()[1]
    grams = np.zeros((2,C+5,C+5))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%6,b-a-60,6)
        for h,x in enumerate(features(raw[a:b],ok,p[a:b],t,links.base.coordinates()[:,-D:],maps(d))):
            j = h+1
            yy = y[a+t,:,j].astype(float)/beta[j]
            far = yy-y[a+t,:,j-1]*coeff[j]/(beta[j]*coeff[j-1])
            labels = links.center(np.stack((yy-p[a+t,:,j],far),axis=2),ok[t])
            obs = np.concatenate((x,labels),axis=2)[ok[t]]
            grams[h] += obs.T@obs
        n += ok[t].sum()
    return grams/n


def train(end):
    ds = np.stack([record(d) for d in range(4,end)])
    co = np.zeros((2,C+3,3))
    for h in range(2):
        g = ds[:,h].mean(0)
        root = links.base.whitening(g[:C,:C],.035)
        for k in range(3):
            v = np.eye(C+5)[:,C+k].copy()
            v[:C] = -root@(root.T@g[:C]@v)
            grad = v@ds[:,h,:,-2:]
            mu,far = grad.mean(0)
            trust = mu**2/(mu**2+grad[:,0].var()/len(ds)+1e-12)
            co[h,:,k] = v[:-2]*trust*mu/(v@g@v+1e-7)*(mu*far>0)
    return np.r_[links.base.coordinates()[:,-D:].ravel(),maps(end).ravel(),co.ravel()]


def signals(raw,grid,p,bank):
    off = 100*D+2*50*D*50*3
    axes = bank[:100*D].reshape(100,D)
    mats = bank[100*D:off].reshape(2,50,D,50,3)
    co = bank[off:].reshape(2,C+3,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        for h,x in enumerate(features(raw[a:b],ok,p[a:b],np.arange(b-a),axes,mats)):
            out[a:b,:,:,h] = x@co[h]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
