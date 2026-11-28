"""Value peer settlements against the distance left on the receiver's currently open price break."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import experiment_dense_passage_probability_news_v1 as news

pathmod,parent,event = news.pathmod,news.parent,news.event
strip,rel,onset,REFERENCE,read_p = news.strip,news.rel,news.onset,news.REFERENCE,news.read_p
OUT = REFERENCE.parent/'dense_live_barrier_receivers_v1'
ARMS = ('remaining_barrier_distance','cash_while_break_is_open','cash_after_exit')
C = 15
N = C+12


def active(move,ok,floor):
    path,signed,width = pathmod.crossings(move,ok,floor)
    n = len(move)
    age = np.zeros((n,50),int)
    sign = signed.copy()
    fraction = np.zeros_like(path)
    live = signed!=0
    for lag in range(1,min(n,60)):
        dist = signed[:-lag]*(path[lag:]-path[:-lag])/width[:-lag].clip(1e-10)
        live = live[:-1]&(signed[lag:]==0)&ok[lag:]&(abs(dist)<1)
        age[lag:][live] = lag
        sign[lag:][live] = signed[:-lag][live]
        fraction[lag:][live] = dist[live]
    s = np.arange(n)[:,None]-age
    width_now = np.take_along_axis(width,s,axis=0)*(sign!=0)
    return s,age,sign,fraction,width_now


def design(move,ok,p,floor,edges,odds,condition,notices):
    origin,age,sign,progress,width = condition
    opened = sign!=0
    ix = np.searchsorted([2,6,15,32],age)*9+np.searchsorted([-.8,-.55,-.3,-.08,.08,.3,.55,.8],progress)
    prob = odds[ix]
    # Remaining displacement in the direction of the first break, accounting for distance already traveled.
    unit = (width/floor.clip(.001)).clip(0,12)
    left = (1-progress)*prob[...,0]
    right = (1+progress)*prob[...,1]
    intrinsic = np.stack((sign,sign*progress,sign*unit,sign*age/60,sign*left,sign*right,
                          sign*unit*(left-right)),axis=-1)
    cs = np.concatenate((np.zeros_like(notices[:1]),notices.cumsum(0)))
    t = np.arange(len(move))+1
    own = (cs[t]-cs[(t-12).clip(0)])/np.sqrt(12)
    def route(w):
        return (own.transpose(0,2,1)@w.T).transpose(0,2,1)
    peer = route(edges)
    placebo = route(edges[event.PERM][:,event.PERM])
    ctrl = np.concatenate((p,intrinsic,own,placebo,move[...,None]/floor.clip(.001)[:,None]),axis=2)
    parts = [peer*gain[...,None] for gain in (opened, opened*left, opened*right,
                      opened*(unit/4).clip(0,3), opened*age/60, opened*progress)]
    return strip.center(np.concatenate((ctrl,*parts),axis=2))


@lru_cache(maxsize=12)
def record(d):
    floor,edges = event.geometry()
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    grams = np.zeros((2,N+6,N+6))
    counts = np.zeros(2)
    coverage = []
    for a,b,move,ok in event.sessions(raw,grid):
        t = np.arange(60+d%5,b-a-60,5)
        data = pathmod.resolution(move,ok,floor)
        path,_,end,result,notice,gaps = data
        cond = active(move,ok,floor)
        x = design(move,ok,p[a:b],floor,edges,news.table(),cond,notice)[t]
        issued,_,sign,frac,width = [v[t] for v in cond]
        opened = sign!=0
        finish = end[issued,np.arange(50)]
        side = result[issued,np.arange(50)]
        for j,h in enumerate((20,60)):
            expiry = t[:,None]+h
            resolved = opened&(finish<=expiry)
            price = path[t]
            stop = np.minimum(np.minimum(finish,issued+60),expiry)
            # Continue tracking the chosen break to its own expiry even if a newer break appears.
            during = (path[stop,np.arange(50)]-price)*opened
            full = path[t+h]-price
            marks = np.stack((sign*width*(1-frac)*(resolved&(side==1)),
                              -sign*width*(1+frac)*(resolved&(side==-1))),axis=2)
            later = (full-during)*opened
            valid = gaps[t+h+1]==gaps[t]
            row = np.concatenate((x,marks,during[...,None],later[...,None],full[...,None],y[a+t,:,j+1:j+2]),axis=2)
            row -= ((row*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
            r = row[valid]
            grams[j] += r.T@r
            counts[j] += len(r)
        coverage.append(opened.mean())
    return grams/counts[:,None,None],float(np.mean(coverage))


def coefficients(g,h,j):
    v = np.eye(N)[:,C:]
    v[:C] = -parent.solve(g[:C,:C],g[:C,C:N],.04)
    cov = v.T@g[:N,:N]@v
    # Train directions for the nominal endpoint amount and both legs of the realized cash flow.
    cash = g[:N,N:N+2].sum(1)
    target = np.column_stack((cash,g[:N,N+2],g[:N,N+3]))
    unit = h[j+1,N+4]/max(1e-8,h[j+1,j+1])
    return v@parent.solve(cov,v.T@target,.8)/np.clip(unit,.05,10)


def train(end):
    gs = np.stack([record(d)[0] for d in range(4,end)])
    a,b = [gs[i::2].mean(0) for i in (0,1)]
    w = np.stack([.5*(coefficients(a[j],b[j],j)+coefficients(b[j],a[j],j)) for j in (0,1)],axis=2)
    floor,edges = event.geometry()
    return np.r_[floor,edges.ravel(),news.table().ravel(),w.ravel()]


def signals(raw,grid,p,bank):
    floor,edges = bank[:50],bank[50:2550].reshape(50,50)
    odds = bank[2550:2640].reshape(news.CELLS,2)
    w = bank[2640:].reshape(N,6)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        x = design(move,ok,p[a:b],floor,edges,odds,active(move,ok,floor),pathmod.settled_notices(move,ok,floor))
        out[a:b] = (x@w).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    (OUT/'training_coverage.json').write_text(json.dumps({d+1:record(d)[1] for d in range(4,16)},indent=2))
