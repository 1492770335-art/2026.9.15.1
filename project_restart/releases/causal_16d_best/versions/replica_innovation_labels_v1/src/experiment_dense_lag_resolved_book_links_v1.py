"""Separate age-specific peer book responses from an equal-age mixture and relabeled delays."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, rel, strip, official
import prepare_dense_receiver_priced_links_reference_v1 as ref


priced = ref.model
links = priced.links
D,C = links.D,links.C
AGES = (0,10,30)
REFERENCE = ref.OUT
OUT = REFERENCE.parent / 'dense_lag_resolved_book_links_v1'
ARMS = ('matched_delay','permuted_delay','delay_above_common')


def read_p(day):
    p = ref.read_p(day)
    if 8<=day<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[day][:,None]
    return p


@lru_cache(maxsize=16)
def relation(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    total = np.zeros((2,len(AGES),50,D,50,D))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        fast,delta,mean = links.raw_view(raw[a:b],ok,np.arange(b-a),links.base.coordinates()[:,-D:])
        t = np.arange(70+day%6,b-a-20,6)
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gaps[t+21]==gaps[t-64]
        source = [links.center(delta[t-age],valid).reshape(len(t),-1) for age in AGES]
        for h,near in enumerate((5,20)):
            later = links.center(mean(t+near-4,t+near+1)-fast[t],valid)
            own = links.center(np.concatenate((fast[t],delta[t],(path[t+near]-path[t])[...,None]),axis=2),valid)
            x = own[valid]
            u = links.base.whitening(x.T@x/len(x),.01)
            beta = u@(u.T@(x.T@later[valid]/len(x)))
            y = (later-own@beta).reshape(len(t),-1)
            for k,z in enumerate(source):
                total[h,k] += (z.T@y).reshape(50,D,50,D)
        n += len(t)
    tensor = (total/n).transpose(0,1,2,4,3,5)
    ids = np.arange(50)
    tensor[:,:,ids,ids] = 0.
    mask = (1.-np.eye(50))[...,None,None]
    tensor -= mask*tensor.sum(2,keepdims=True)/49
    return tensor


@lru_cache(maxsize=16)
def maps(end):
    dates = np.stack([priced.book_cash(d) for d in range(end)])
    result = []
    for h in range(2):
        halves = []
        for k in range(2):
            days = range(k,end,2)
            response = sum(relation(d)[h] for d in days)/len(days)
            _,price = priced.receiver_prices(dates[1-k::2,h])
            halves.append(np.einsum('asrid,rd->asri',response,price))
        a,b = (v.reshape(-1,D) for v in halves)
        val,u = np.linalg.eigh(.5*(a.T@b+b.T@a))
        noise = .25*np.square((a-b)@u).sum(0)
        strength = val.clip(0)/(val.clip(0)+noise+.000001)
        w = (((a+b)*.5)@u*strength)@u.T
        result.append(w.reshape(len(AGES),50,50,D).transpose(0,1,3,2))
    return np.stack(result)


def features(raw,ok,p,t,axes,weights):
    fast,delta,_ = links.raw_view(raw,ok,np.arange(len(raw)),axes)
    complete = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    valid = (complete[t+1]==complete[(t-64).clip(0)]) & (t>=64)[:,None]
    ctrl = links.center(np.concatenate((fast[t],delta[t],p[t]),axis=2),valid)
    for matrices in weights:
        common = matrices.mean(0)
        aligned = np.zeros(valid.shape)
        shuffled = np.zeros_like(aligned)
        equal = np.zeros_like(aligned)
        for k,age in enumerate(AGES):
            x = links.center(delta[(t-age).clip(0)],valid).reshape(len(t),-1)
            def apply(w):
                return x@w.reshape(50*D,50)
            aligned += apply(matrices[k])
            shuffled += apply(matrices[(k+1)%len(AGES)])
            equal += apply(common)
        z = np.stack((aligned,shuffled,aligned-equal),axis=2)/len(AGES)
        yield np.concatenate((ctrl,links.center(z,valid)),axis=2)


@lru_cache(maxsize=12)
def record(day,design=features,graphbank=maps):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    scale = strip.physical()[1]
    gs = np.zeros((2,C+5,C+5))
    count = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(70+day%6,b-a-60,6)
        for h,x in enumerate(design(raw[a:b],ok,p[a:b],t,links.base.coordinates()[:,-D:],graphbank(day))):
            j = h+1
            yy = y[a+t,:,j].astype(float)/beta[j]
            tail = yy-y[a+t,:,j-1]*scale[j]/(beta[j]*scale[j-1])
            target = links.center(np.stack((yy-p[a+t,:,j],tail),axis=2),ok[t])
            obs = np.concatenate((x,target),axis=2)[ok[t]]
            gs[h] += obs.T@obs
        count += ok[t].sum()
    return gs/count


def train(end,reader=record,graphbank=maps):
    gs = np.stack([reader(d) for d in range(4,end)])
    co = np.zeros((2,C+3,3))
    for h in range(2):
        g = gs[:,h].mean(0)
        w = links.base.whitening(g[:C,:C],.035)
        for k in range(3):
            v = np.eye(C+5)[:,C+k].copy()
            v[:C] = -w@(w.T@g[:C]@v)
            grad = v@gs[:,h,:,-2:]
            mu,far = grad.mean(0)
            trust = mu**2/(mu**2+grad[:,0].var()/len(gs)+1e-12)
            co[h,:,k] = v[:-2]*trust*mu/(v@g@v+1e-7)*(mu*far>0)
    return np.r_[links.base.coordinates()[:,-D:].ravel(),graphbank(end).ravel(),co.ravel()]


def signals(raw,grid,p,bank,design=features,num_ages=len(AGES)):
    off = 100*D+2*num_ages*50*D*50
    w = bank[100*D:off].reshape(2,num_ages,50,D,50)
    axes = bank[:100*D].reshape(100,D)
    co = bank[off:].reshape(2,C+3,3)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        for h,x in enumerate(design(raw[a:b],ok,p[a:b],np.arange(b-a),axes,w)):
            out[a:b,:,:,h] = x@co[h]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
