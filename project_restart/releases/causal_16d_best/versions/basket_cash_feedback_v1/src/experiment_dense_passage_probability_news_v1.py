"""Use revisions in the conditional odds of extending or retracing a price break as peer messages."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import experiment_dense_passage_resolution_v1 as pathmod

parent,event,strip,rel,onset = pathmod.parent,pathmod.event,pathmod.strip,pathmod.rel,pathmod.onset
REFERENCE,read_p = pathmod.REFERENCE,pathmod.read_p
OUT = REFERENCE.parent/'dense_passage_probability_news_v1'
ARMS = ('pending_odds_revisions','settlement_surprises','both_types_of_news')
CELLS = 5*9
N = pathmod.N+8


def cell(age,progress):
    a = np.searchsorted([2,6,15,32],age)
    b = np.searchsorted([-.8,-.55,-.3,-.08,.08,.3,.55,.8],progress)
    return a*9+b


@lru_cache(maxsize=1)
def table():
    floor,_ = event.geometry()
    counts = np.zeros((CELLS,3))
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for _,_,move,ok in event.sessions(raw,grid):
            path,signed,width = pathmod.crossings(move,ok,floor)
            _,_,finish,kind,_,gaps = pathmod.resolution(move,ok,floor)
            s,stock = np.nonzero(signed[:-60])
            valid = gaps[s+61,stock]==gaps[s,stock]
            s,stock = s[valid],stock[valid]
            stop = finish[s,stock]
            final = kind[s,stock]+1
            for age in range(60):
                take = s+age<stop
                start,col = s[take],stock[take]
                dist = signed[start,col]*(path[start+age,col]-path[start,col])/width[start,col]
                idx = cell(age,dist)*3+final[take]
                counts += np.bincount(idx,minlength=CELLS*3).reshape(CELLS,3)
    coarse = counts.reshape(5,9,3).sum(0)
    prior = (coarse+.5)/(coarse.sum(1)[:,None]+1.5)
    return ((counts+80*np.tile(prior,(5,1)))/(counts.sum(1)[:,None]+80))[:,[2,0]]


def updates(move,ok,floor,odds):
    path,signed,width = pathmod.crossings(move,ok,floor)
    # At each age every update is written to the current clock, using the saved state at issuance.
    n = len(path)
    start,col = np.nonzero(signed)
    direction = signed[start,col]
    prob = np.tile(odds[cell(0,0.)],(len(start),1))
    early = np.zeros((n,50,2))
    final = np.zeros_like(early)
    receipts = np.zeros_like(early)
    for age in range(1,min(61,n)):
        valid = start+age<n
        start,col,direction,prob = start[valid],col[valid],direction[valid],prob[valid]
        now = start+age
        progress = direction*(path[now,col]-path[start,col])/width[start,col]
        settle = abs(progress)>=1
        open_ = (~settle)&ok[now,col]&(age<60)
        paid = (settle|(age==60))&ok[now,col]
        y = np.stack((progress>=1,progress<=-1),axis=1)
        np.add.at(receipts,(now[paid],col[paid]),direction[paid,None]*y[paid])
        np.add.at(final,(now[paid],col[paid]),direction[paid,None]*(y[paid]-prob[paid]))
        next_prob = odds[cell(age,progress[open_])]
        np.add.at(early,(now[open_],col[open_]),direction[open_,None]*(next_prob-prob[open_]))
        start,col,direction,prob = start[open_],col[open_],direction[open_],next_prob
    return receipts.clip(-1,1),early,final


def designs(move,ok,p,floor,edges,odds):
    raw,early,final = updates(move,ok,floor,odds)
    common = pathmod.features(move,ok,p,floor,edges,raw)
    for news in (early,final,early+final):
        news = news.clip(-2,2)
        cs = np.concatenate((np.zeros_like(news[:1]),news.cumsum(0)))
        t = np.arange(len(p))+1
        cols = []
        for width in (6,32):
            own = (cs[t]-cs[(t-width).clip(0)])/np.sqrt(width)
            peer = (own.transpose(0,2,1)@edges.T).transpose(0,2,1)
            cols.extend((strip.center(own),strip.center(peer)))
        yield np.concatenate((common,*cols),axis=2)


@lru_cache(maxsize=12)
def record(d):
    floor,edges = event.geometry()
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    grams = np.zeros((3,2,N+9,N+9))
    counts = np.zeros(2)
    for a,b,move,ok in event.sessions(raw,grid):
        data = pathmod.resolution(move,ok,floor)
        t = np.arange(55+d%5,b-a-60,5)
        targets = [pathmod.targets(t,h,data) for h in pathmod.HORIZONS]
        for arm,x in enumerate(designs(move,ok,p[a:b],floor,edges,table())):
            for j,(out,valid) in enumerate(targets):
                z = np.concatenate((x[t],out,y[a+t,:,j+1:j+2]),axis=2)
                z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
                r = z[valid]
                grams[arm,j] += r.T@r
                if arm==0:
                    counts[j] += len(r)
    return grams/counts[None,:,None,None]


def fit(g,h,j):
    at = [N,N+1]+([N+2] if j==1 else [])
    mi = [N+3,N+4]
    base = list(range(8))+at
    v = np.eye(N+5)[:,mi]
    v[base] = -parent.solve(h[np.ix_(base,base)],h[np.ix_(base,mi)],.04)
    pay = parent.solve(v.T@h[:N+5,:N+5]@v,v.T@h[:N+5,N+5],.35)
    pay /= np.clip(h[j+1,N+7]/h[j+1,j+1],.05,10)
    # Condition on both the observed barrier receipts and the receiver's own probability updates.
    ctrl = list(range(pathmod.N))+[26,27,30,31]
    idx = [28,29,32,33]
    proj = np.eye(N)[:,idx]
    proj[ctrl] = -parent.solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.05)
    residual = g[:N,mi]-g[:N,at]@parent.solve(g[np.ix_(at,at)],g[np.ix_(at,mi)],.02)
    return proj@parent.solve(proj.T@g[:N,:N]@proj,proj.T@residual,.7)@pay


def train(end):
    gs = np.stack([record(d) for d in range(4,end)])
    halves = [gs[k::2].mean(0) for k in (0,1)]
    w = np.stack([[.5*sum(fit(halves[k][arm,j],halves[1-k][arm,j],j) for k in (0,1))
                    for j in range(2)] for arm in range(3)])
    floor,edges = event.geometry()
    return np.r_[floor,edges.ravel(),table().ravel(),w.ravel()]


def signals(raw,grid,p,bank):
    floor,edges = bank[:50],bank[50:2550].reshape(50,50)
    odds = bank[2550:2640].reshape(CELLS,2)
    w = bank[2640:].reshape(3,2,N)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        for arm,x in enumerate(designs(move,ok,p[a:b],floor,edges,odds)):
            out[a:b,:,arm] = x@w[arm].T
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
