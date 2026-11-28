"""Route peer price changes after a signed book imbalance and before the receiver reprices."""
from functools import lru_cache
import sys
import json

import numpy as np
import torch

import prepare_dense_live_barrier_reference_v1 as best

event,strip,rel,onset = best.model.event,best.model.strip,best.model.rel,best.model.onset
solve = best.model.parent.solve
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_waiting_book_witnesses_v1'
ARMS = ('next_price_change','after_next_change','residual_return')
BID,ASK = [41,43,50,52,54,56,79],[42,44,51,53,55,57,80]
C,D = 15,23


def depth(raw):
    return np.arcsinh(raw[...,BID].astype(np.float32))-np.arcsinh(raw[...,ASK].astype(np.float32))


@lru_cache(maxsize=1)
def axis():
    gram = np.zeros((8,8))
    mass = 0
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
        for a,b,move,ok in event.sessions(raw,grid):
            t = np.arange(4+d%3,b-a-5,3)
            valid = np.logical_and.reduce([ok[t+j] for j in range(6)])
            path = move.cumsum(0)
            z = np.concatenate((depth(raw[a+t]),(path[t+5]-path[t])[...,None]),axis=2)[valid]
            gram += z.T@z
            mass += len(z)
    w = solve(gram[:7,:7],gram[:7,7],.35)
    return w/np.sqrt(w@gram[:7,:7]@w/mass)


def issue(raw,move,ok,floor,w):
    n = len(move)
    t = np.arange(n)[:,None]
    pressure = depth(raw)@w
    side = np.sign(pressure)
    strong = abs(pressure)>=.85
    prev = np.vstack((np.zeros(50),pressure[:-1]))
    changed = abs(move)>.001
    start = strong & (prev*side<.85) & ok & ~changed
    last = np.maximum.accumulate(np.where(start,t,-64),axis=0)
    s = last.clip(0)
    clock = t-last
    # A new opposite or weak book or a tick in the receiver price ends its outstanding pressure.
    closed = np.maximum.accumulate(np.where(changed|~ok|(abs(pressure)<.32)|(side!=np.sign(prev)),t,-65),axis=0)
    active = (last>=0)&(last>=closed)&(clock>0)&(clock<=32)
    return pressure,active,clock.clip(0,32),s,changed


def features(raw,move,ok,p,floor,edges,w):
    pressure,active,age,s,changed = issue(raw,move,ok,floor,w)
    t = np.arange(len(move))[:,None]
    cols = np.arange(50)
    z = move/floor.clip(.004)
    def stream(weights):
        path = (z@weights.T).cumsum(0)
        endpoint = path[s,cols]
        post = (path-endpoint)/np.sqrt(age.clip(1))
        pre = (endpoint-path[(s-age).clip(0),cols])/np.sqrt(age.clip(1))
        return post.clip(-8,8)*active,pre.clip(-8,8)*active
    peer,prior = stream(edges)
    random,_ = stream(edges[event.PERM][:,event.PERM])
    market,_ = stream((1-np.eye(50))/49)
    launch = pressure[s,cols]
    direction = np.sign(launch)
    state = np.stack((pressure,active*direction,active*launch,active*pressure,active*direction*age/16,
                      z.clip(-8,8),random,market,prior,abs(random)*direction,abs(market)*direction,
                      abs(prior)*direction),axis=2)
    aligned = (peer*direction>0)
    extra = np.stack((peer,abs(peer)*direction,peer*age/16,peer*np.minimum(abs(launch),3),
                      peer*(pressure*direction).clip(0,3),peer*aligned*(age>=4),
                      peer*~aligned*(age>=4),peer*np.tanh(prior*direction)),axis=2)
    return strip.center(np.concatenate((p,state,extra),axis=2)),active,changed


@lru_cache(maxsize=12)
def record(d):
    floor,edges = event.geometry()
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    gram = np.zeros((2,D+4,D+4))
    count = np.zeros(2)
    audit = np.zeros(3)
    for a,b,move,ok in event.sessions(raw,grid):
        t = np.arange(32+d%5,b-a-60,5)
        x,active,changed = features(raw[a:b],move,ok,p[a:b],floor,edges,axis())
        n = len(move)
        first = np.minimum.accumulate(np.where(changed,np.arange(n)[:,None],n)[::-1],axis=0)[::-1][t+1]
        path = move.cumsum(0)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        for j,h in enumerate((20,60)):
            valid = gap[t+h+1]==gap[t]
            mask = active[t]&valid
            end = np.minimum(first,t[:,None]+h)
            first_cash = path[end,np.arange(50)]-path[t]
            total = path[t+h]-path[t]
            split = np.stack((first_cash*active[t],(total-first_cash)*active[t],total,y[a+t,:,j+1]),axis=2)
            rows = np.concatenate((x[t],split),axis=2)
            rows -= ((rows*valid[...,None]).sum(1)/valid.sum(1).clip(1)[:,None])[:,None]
            z = rows[valid]
            gram[j] += z.T@z
            count[j] += len(z)
            if j==1:
                audit += [mask.sum(),valid.sum(),(mask&(first<=t[:,None]+h)).sum()]
    return gram/count[:,None,None],audit


def train(end):
    days = np.stack([record(d)[0] for d in range(4,end)])
    weights = []
    for j in range(2):
        g = days[:,j].mean(0)
        v = np.eye(D)[:,C:]
        v[:C] = -solve(g[:C,:C],g[:C,C:D],.04)
        rows = days[:,j]
        units = rows[:,j+1,D+2]/rows[:,j+1,j+1].clip(1e-8)
        beta = rows[:,j+1,D+3]/rows[:,j+1,j+1].clip(1e-8)
        rhs = np.concatenate((rows[:,:D,D:D+2]/units[:,None,None].clip(.05,10),
            (rows[:,:D,D+3]/beta[:,None].clip(.2,4)-rows[:,:D,j+1])[...,None]),axis=2)
        cross = v.T@rhs
        mu = cross.mean(0)
        shrunk = mu**3/(mu**2+2*cross.var(0)/len(days)+1e-11)
        weights.append(v@solve(v.T@g[:D,:D]@v,shrunk,.7))
    floor,edges = event.geometry()
    return np.r_[axis(),floor,edges.ravel(),np.stack(weights,axis=2).ravel()]


def signals(raw,grid,p,bank):
    axis_w,floor,edges = bank[:7],bank[7:57],bank[57:2557].reshape(50,50)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        x = features(raw[a:b],move,ok,p[a:b],floor,edges,axis_w)[0]
        out[a:b] = (x@bank[2557:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    (OUT/'coverage.json').write_text(json.dumps({d+1:record(d)[1].tolist() for d in range(4,16)},indent=2))
