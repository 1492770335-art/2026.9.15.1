"""Compare absolute distal returns with amount and direction estimates on issue-time matched stock pairs."""
from functools import lru_cache
import json
import sys

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtr
import torch

import experiment_dense_exit_book_trajectory_v1 as book
import experiment_dense_waiting_book_witnesses_v1 as base

event,strip,rel,onset,REFERENCE,read_p = base.event,base.strip,base.rel,base.onset,base.REFERENCE,base.read_p
OUT = REFERENCE.parent/'dense_local_relative_value_v1'
ARMS = ('individual_distal_return','matched_distal_spread','matched_win_probability')
K,C,D = 8,9,41
NEIGHBORS = 4


def smooth(x,span):
    k = np.exp(-1/span)
    return lfilter([1-k],[1,-k],x,axis=0)


def state(raw,move,ok,p,floor,axes):
    n = len(move)
    t = np.arange(n)
    path = move.cumsum(0)
    def leg(lo,hi):
        return (path[(t-lo).clip(0)]-path[(t-hi).clip(0)])/np.sqrt(hi-lo)
    short,mid,long = [leg(*h)/floor.clip(.004) for h in ((0,5),(5,20),(20,60))]
    vol = np.sqrt(smooth(move*move,100)+.25*floor**2).clip(.004)
    risk = np.log(vol/floor.clip(.004))
    desc = np.stack((short,mid,long,risk,np.log(floor.clip(.004))[None,:]+risk,p[...,0],p[...,1],p[...,2]),axis=2)
    desc = np.tanh(desc*np.array([.6,.6,.6,.8,.8,1.8,1.8,1.8]))
    coord = book.coordinates(raw,t)@axes
    current = coord
    quick = coord-smooth(coord,6)
    slow = smooth(coord,6)-smooth(coord,48)
    ctrl = np.stack((short,mid,long,risk,vol,p[...,2]*np.tanh(risk)),axis=2)
    x = strip.center(np.concatenate((p,ctrl,current,quick,slow,current*np.tanh(mid[...,None])),axis=2)).clip(-8,8)
    return x,desc,vol


def matches(desc,ok):
    diff = desc[:,:,None,:]-desc[:,None,:,:]
    distance = (diff**2).sum(3)
    exclude = ~ok[:,None,:]|np.eye(50,dtype=bool)
    distance = np.where(exclude,np.inf,distance)
    idx = np.argsort(distance,axis=2,kind='stable')[:,:,:NEIGHBORS]
    dist = np.take_along_axis(distance,idx,axis=2)
    weight = np.exp(-dist)*ok[...,None]
    return idx,weight


def edge_values(v,idx):
    t = np.arange(len(v))[:,None,None]
    return v[:,:,None]-v[t,idx]


def integrate(delta,idx,weights):
    # Combine outgoing and incoming opinions; apply Jacobi updates of the ridge-regularized graph Laplacian.
    n = len(idx)
    row = np.arange(n)[:,None,None]*50+np.arange(50)[None,:,None]
    dest = np.arange(n)[:,None,None]*50+idx
    recv = lambda z:np.bincount(dest.ravel(),weights=z.ravel(),minlength=n*50).reshape(n,50)
    b = (weights*delta).sum(2)-recv(weights*delta)
    degree = weights.sum(2)+recv(weights)
    x = np.zeros_like(b)
    for i in range(8):
        near = (weights*x[np.arange(n)[:,None,None],idx]).sum(2)+recv(weights*x[:,:,None])
        x = .45*x+.55*(b+near)/(degree+1.2)
    return x-x.mean(1,keepdims=True)


@lru_cache(maxsize=12)
def record(d):
    floor,_ = event.geometry()
    axes = book.dictionary()[:,-K:]
    raw,grid,p = base.onset.settled.load(d,'x'),base.onset.settled.load(d,'time'),read_p(d)
    batches = []
    g = np.zeros((3,D+1,D+1))
    audit = np.zeros(5)
    for a,b,m,ok in event.sessions(raw,grid):
        x,desc,vol = state(raw[a:b],m,ok,p[a:b],floor,axes)
        t = np.arange(30+d%5,b-a-60,5)
        path = m.cumsum(0)
        leg = np.stack((path[t+20]-path[t],path[t+60]-path[t]),axis=2)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        batches.append((x[t],desc[t],vol[t],ok[t],valid,leg,p[a+t]))
    x,desc,vol,ok,valid,leg,p = [np.concatenate(v) for v in zip(*batches)]
    def center(z):
        return (z-(z*valid[...,None]).sum(1,keepdims=True)/valid.sum(1)[:,None,None].clip(1))*valid[...,None]
    y,q = center(leg),center(p[...,1:])
    unit = ((y*q).sum((0,1))/(q*q).sum((0,1))).clip(.05,10)
    actual = (leg[:,:,1]-leg[:,:,0])/unit[1]
    predicted = p[:,:,2]-p[:,:,1]*unit[0]/unit[1]
    err = center((actual-predicted)[...,None])[...,0]
    row = np.concatenate((x,err[...,None]),axis=2)[valid]
    g[0] = row.T@row/len(row)
    for start in range(0,len(x),96):
        sl = slice(start,start+96)
        idx,w = matches(desc[sl],ok[sl])
        ii = np.arange(len(idx))[:,None,None]
        good = valid[sl,:,None]&valid[sl][ii,idx]
        w *= good
        dx = edge_values(x[sl],idx)
        dp,dy = (edge_values(z[sl],idx) for z in (predicted,actual))
        sd = np.sqrt(40*(vol[sl,:,None]**2+vol[sl][ii,idx]**2))/unit[1]
        sd = sd.clip(.08,6)
        paid = dy-dp
        # One update of the probit mean score, in return units. Each pair has its own expected spread.
        sign = ((dy>0)+.5*(dy==0)-ndtr(dp/sd))*sd*np.sqrt(2*np.pi)
        for j,target in enumerate((paid,sign),1):
            z = np.concatenate((dx,target[...,None]),axis=3).reshape(-1,D+1)
            g[j] += z.T@(w.reshape(-1,1)*z)
        control = np.roll(actual[sl],19,axis=1)
        audit += [w.sum(),(w*(dy**2)).sum(),(w*((actual[sl,:,None]-control[:,:,None])**2)).sum(),
                  (w*(abs(dy)>0)).sum(),(w*paid**2).sum()]
    g[1:] /= audit[0]
    return g,audit


def train(end):
    rows = np.stack([record(d)[0] for d in range(4,end)])
    co = []
    for j in range(3):
        days = rows[:,j]
        g = days.mean(0)
        v = np.eye(D)[:,C:]
        v[:C] = -base.solve(g[:C,:C],g[:C,C:D],.05)
        directions = days[:,:D,-1]@v
        mu = directions.mean(0)
        pay = mu**3/(mu**2+2*directions.var(0)/len(days)+1e-10)
        co.append(v@base.solve(v.T@g[:D,:D]@v,pay,.9))
    floor,_ = event.geometry()
    return np.r_[floor,book.dictionary()[:,-K:].ravel(),np.stack(co).ravel()]


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    w = bank[-3*D:].reshape(3,D)
    for a,b,m,ok in event.sessions(raw,grid):
        x,desc,_ = state(raw[a:b],m,ok,p[a:b],bank[:50],bank[50:-3*D].reshape(83,K))
        z = x@w.T
        out[a:b,:,0,1] = z[:,:,0]
        for start in range(0,len(x),64):
            sl = slice(start,min(start+64,len(x)))
            idx,weight = matches(desc[sl],ok[sl])
            for j in (1,2):
                out[a+sl.start:a+sl.stop,:,j,1] = integrate(edge_values(z[sl,:,j],idx),idx,weight)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    report = {}
    for s in (8,12):
        n,spread,plain,moved,res = sum(record(d)[1] for d in range(s,s+4))
        report[s+1] = dict(matched_rms=(spread/n)**.5,shifted_rms=(plain/n)**.5,non_tie=moved/n)
    (OUT/'pair_spreads.json').write_text(json.dumps(report,indent=2))
