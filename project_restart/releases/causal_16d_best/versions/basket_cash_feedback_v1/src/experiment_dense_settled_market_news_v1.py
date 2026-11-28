"""Measure delayed returns of a separate basket after the same paid forecast surprise is seen by two donor panels."""
from functools import lru_cache
import json
import sys

import numpy as np
from scipy.signal import lfilter
import torch

import experiment_dense_market_one_block_memory_v1 as market

base,event,strip,rel,onset,REFERENCE,read_p = market.base,market.event,market.strip,market.rel,market.onset,market.REFERENCE,market.read_p
OUT = REFERENCE.parent/'dense_settled_market_news_v1'
ARMS = ('terminal_surprise','pre_terminal_revision','late_session_replication')
GROUP = np.random.default_rng(164132).permutation(50).reshape(5,10)
C,D = 8,20


def means(x,ok):
    return (x[:,GROUP]*ok[:,GROUP]).sum(2)/ok[:,GROUP].sum(2).clip(1)


def ew(x,span):
    return lfilter([1.],[1,-np.exp(-1/span)],x,axis=0)


def design(m,ok,p):
    t = np.arange(len(m))
    path = means(m,ok).cumsum(0)
    news,issued = [],[]
    for j,h in enumerate((5,20,60)):
        q = means(p[...,j],ok)
        s = (t-h).clip(0)
        real = path-path[s]
        active = t>=h
        x = q[s].mean(1)
        y = real.mean(1)
        mass = ew(active.astype(float),520)+1
        a,b = [ew(z*active,520) for z in (x,y)]
        unit = ((ew(x*y*active,520)-a*b/mass+.7)/(ew(x*x*active,520)-a*a/mass+1.)).clip(.1,5)
        surprise = (real-q[s]*unit[s,None])*active[:,None]
        news.append(surprise)
        issued.append(q)
    channels = np.stack((news[0],(news[1]-news[0])*(t>=20)[:,None],(news[2]-news[1])*(t>=60)[:,None]),axis=2)
    f = []
    def agree(x,y):
        return np.where(x*y>0,np.sign(x)*np.minimum(abs(x),abs(y)),0.)
    for k in range(5):
        ids = [(k+i)%5 for i in (1,2,3,4)]
        a,b = channels[:,ids[:2]].mean(1),channels[:,ids[2:]].mean(1)
        shared = agree(a,b)
        avg = (a+b)/2
        phase = (1-np.exp(-t/250))[:,None]
        past = market.window(means(m,ok),20)
        ctrl = np.column_stack((np.ones(len(t)),*(q[:,k] for q in issued),*(q[:,ids].mean(1) for q in issued[1:]),
                                past[:,k],past[:,ids].mean(1)))
        f.append(np.concatenate((ctrl,avg,shared,avg*phase,shared*phase),axis=1))
    return np.stack(f,axis=1)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    gram = 0.
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(60+d%3,b-a-60,3)
        x = design(m,ok,p[a:b])[t]
        path = means(m,ok).cumsum(0)
        cash = np.stack([path[t+h+20]-path[t+h] for h in (0,20,40)],axis=2)
        label = np.stack([means(y[a:b,:,j],ok)[t] for j in (1,2)],axis=2)
        z = np.concatenate((x,cash,label),axis=2).reshape(-1,D+5)
        gram += z.T@z/len(z)
    return gram/2


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    g = days.mean(0)
    coeff = []
    for idx in ([8,11,14,17],[9,10,12,13,15,16,18,19],[17,18,19]):
        ctrl = np.arange(C if idx[0]!=17 else 17)
        v = np.eye(D)[:,idx]
        v[ctrl] = -base.solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.04)
        def centered_cov(i,k):
            return days[:,i,k]-days[:,0,i]*days[:,0,k]/days[:,0,0]
        beta = np.stack([centered_cov(2+j,D+3+j)/centered_cov(2+j,2+j).clip(.001) for j in range(2)],axis=1)
        rhs = days[:,:D,D+3:]/beta[:,None].clip(.2,4)-days[:,:D,2:4]
        cross = v.T@rhs
        mu = cross.mean(0)
        stable = mu**3/(mu**2+2*cross.var(0)/len(days)+1e-11)
        coeff.append(v@base.solve(v.T@g[:D,:D]@v,stable,1.3))
    return np.stack(coeff,axis=1)


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = design(m,ok,p[a:b])
        z = (x@bank.reshape(D,6)).reshape(b-a,5,3,2)
        out[a:b] = np.repeat(z,10,axis=1)[:,np.argsort(GROUP.ravel())]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    report = {}
    for s in (8,12):
        g = np.mean([record(d) for d in range(s,s+4)],axis=0)
        g = g-np.outer(g[:,0],g[0])/g[0,0]
        idx = [8,9,10]
        report[s+1] = (g[idx,D:D+3]/np.sqrt(g.diagonal()[idx,None]*g.diagonal()[None,D:D+3])).tolist()
    (OUT/'external_legs.json').write_text(json.dumps(report,indent=2))
