"""Estimate directed peer edges from subsequent price crossings, controlling for receiver history."""
from functools import lru_cache
import sys

import numpy as np
import torch

import prepare_dense_peer_passage_reference_v1 as best


parent = best.model
event,strip,rel,onset = parent.event,parent.strip,parent.rel,parent.onset
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_directed_passage_graph_v1'
ARMS = ('directed_passage_neighbors','beyond_contemporaneous_neighbors','beyond_reversed_edges')
UNIT = parent.N
N = 3*UNIT


@lru_cache(maxsize=1)
def graphs():
    floor,price_graph = event.geometry()
    days = []
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        xs,ys,controls = [],[],[]
        for _,_,move,ok in event.sessions(raw,grid):
            hit = event.signed_events(move,ok,floor)
            impulse = hit[...,0].astype(float)-hit[...,1]
            flow = np.vstack((np.zeros(50),impulse.cumsum(0)))
            level = move.cumsum(0)
            t = np.arange(50+d%3,len(move)-60,3)
            x = flow[t+1]-flow[t-4]
            marks,_,_,valid = parent.marked_prices(move,ok,floor,t)
            y = (marks[...,:2]@np.array([1.,.5]))*valid
            age = t[:,None,None]-np.maximum.accumulate(np.where(hit,np.arange(len(move))[:,None,None],-1),axis=0)[t]
            v = np.stack((np.ones_like(x),x,np.broadcast_to(x.mean(1,keepdims=True),x.shape),
                          np.tanh((level[t]-level[t-15])/floor.clip(.0001)/4),
                          (age[...,0].clip(0,40)-age[...,1].clip(0,40))/40),axis=2)
            xs.append(x-x.mean(1,keepdims=True))
            ys.append(y)
            controls.append(v)
        x,y,v = np.concatenate(xs),np.concatenate(ys),np.concatenate(controls)
        vt = v.transpose(1,2,0)
        cov = vt@v.transpose(1,0,2)
        coef = np.linalg.solve(cov+.001*np.eye(v.shape[-1]),vt@y.T[...,None])
        residual = y-np.einsum('tsc,sc->ts',v,coef[...,0])
        days.append((residual.T@x)/np.sqrt(np.outer((residual**2).sum(0),(x*x).sum(0))).clip(1e-5))
    odd,even = [np.mean(days[j::2],axis=0) for j in (0,1)]
    signed = .5*(odd+even)
    strength = signed.clip(0)/(1.+abs(odd-even)/(abs(signed)+.005))
    def weights(r):
        matrix = r.copy()
        np.fill_diagonal(matrix,-np.inf)
        idx = np.argsort(matrix,axis=1)[:,-8:]
        w = np.zeros_like(matrix)
        np.put_along_axis(w,idx,np.take_along_axis(matrix,idx,axis=1).clip(.001),axis=1)
        return w/w.sum(1,keepdims=True)
    return floor,np.stack((weights(strength),price_graph,weights(strength.T)))


def design(move,ok,p,floor,banks):
    return np.concatenate([parent.design(move,ok,p,floor,w) for w in banks],axis=2)


@lru_cache(maxsize=12)
def gram(d):
    floor,net = graphs()
    raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
    p = read_p(d)
    total = 0.
    count = 0
    for a,b,move,ok in event.sessions(raw,grid):
        t = np.arange(55+d%5,b-a-60,5)
        x = design(move,ok,p[a:b],floor,net)[t]
        m,_,_,valid = parent.marked_prices(move,ok,floor,t)
        row = np.concatenate((x,m),axis=2)
        row -= ((row*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        r = row[valid]
        total += r.T@r
        count += len(r)
    return total/count


def propensity(g,j):
    idx = np.array([8,9,12,13,14,15,16,17])
    ctrl = [0,1,2,3,4,5,6,7,10,11]
    if j:
        ctrl += list(idx+UNIT*j)
    proj = np.eye(N)[:,idx]
    proj[ctrl] = -parent.solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,idx)],.05)
    cov = proj.T@g[:N,:N]@proj
    return proj@parent.solve(cov,proj.T@g[:N,N:],.65)


def train(end):
    days = list(range(4,end))
    out = np.zeros((N,3,2))
    for j in (0,1):
        g = np.mean([gram(d) for d in days[j::2]],axis=0)
        h = np.mean([parent.gram(d) for d in days[1-j::2]],axis=0)
        cost = parent.price(h)[0]
        for k in range(3):
            out[:,k] += .5*propensity(g,k)@cost
    floor,net = graphs()
    return np.r_[floor,net.ravel(),out.ravel()]


def signals(raw,grid,p,bank):
    floor,net = bank[:50],bank[50:7550].reshape(3,50,50)
    w = bank[7550:].reshape(N,6)
    z = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        z[a:b] = (design(move,ok,p[a:b],floor,net)@w).reshape(b-a,50,3,2)
    return z


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
