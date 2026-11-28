"""Compare reproducible cash during a book transition with cash earned strictly after the destination."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import experiment_dense_book_cycle_rewards_v1 as cycle

base,strip,event,onset,rel = cycle.base,cycle.strip,cycle.event,cycle.onset,cycle.rel
REFERENCE,read_p = cycle.REFERENCE,cycle.read_p
OUT = REFERENCE.parent/'dense_after_book_transition_v1'
ARMS = ('future_only_edge','ordered_arrival','retained_over_two_transitions')
S,C,D = 45,9,18
coords = np.stack(np.unravel_index(np.arange(S),(5,3,3)),axis=1)
mirror = S-1-np.arange(S)
distance = np.abs(coords[:,None]-coords[None])
kernel = np.exp(-2.0*distance[:,:,0]-1.7*distance[:,:,1]-2.5*distance[:,:,2])


def state(raw,m,ok):
    _,vol,q = cycle.state(raw,m,ok)
    slow = cycle.ref.window(q,20)/np.minimum(np.arange(1,len(q)+1),20)[:,None]
    move = cycle.ref.window(m,5)
    code = (np.digitize(q,[-.65,-.22,.22,.65])*3+np.digitize(q-slow,[-.15,.15]))*3
    return code+np.digitize(move,[-.002,.002]),vol,q


@lru_cache(maxsize=16)
def moments(d):
    raw,grid = [onset.settled.load(d,k) for k in ('x','time')]
    totals = np.zeros((5,S,S))
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(30+d%3,b-a-60,3)
        ids,vol,q = state(raw[a:b],m,ok)
        path = m.cumsum(0)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t-5]
        ix = (ids[t-5]*S+ids[t])[valid]
        totals[0] += np.bincount(ix,minlength=S*S).reshape(S,S)
        for j,(start,end) in enumerate(((-5,0),(0,5),(5,20),(20,60))):
            v = path[t+end]-path[t+start]
            v -= (v*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)
            val = v/(vol[t]*np.sqrt(5))
            totals[j+1] += np.bincount(ix,weights=val[valid],minlength=S*S).reshape(S,S)
    return totals


def table(days):
    stats = sum(moments(d) for d in days)
    # Flip buy/sell and the sign of the move. The flat-price category is invariant.
    rev = stats[:,mirror][:,:,mirror]
    count = stats[0]+rev[0]
    payoff = stats[1:]-rev[1:]
    pooled_counts = kernel@count@kernel.T
    expected = (kernel@payoff@kernel.T)/(pooled_counts+1.)
    mean = (payoff+50*expected)/(count+50)
    destination = payoff.sum(1)/(count.sum(0)+1.)
    return mean-destination[:,None,:],destination,count


def bank(end):
    halves = [table(range(i,end,2)) for i in (0,1)]
    a,b = [v[0] for v in halves]
    edge = np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
    dest = (halves[0][1]+halves[1][1])/2
    return np.r_[edge.ravel(),dest.ravel()]


def features(raw,m,ok,p,params):
    edge,dest = params[:4*S*S].reshape(4,S,S),params[4*S*S:].reshape(4,S)
    code,vol,q = state(raw,m,ok)
    t = np.arange(len(m))
    s,v = (t-5).clip(0),(t-10).clip(0)
    now = np.moveaxis(edge[:,code[s],code],0,2)
    prev = np.moveaxis(edge[:,code[v],code[s]],0,2)
    reverse = np.moveaxis(edge[:,code,code[s]],0,2)
    destination = np.moveaxis(dest[:,code],0,2)
    scale = vol*np.sqrt(5)
    gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    valid = (gap[t+1]==gap[v])&(t>=10)[:,None]
    def spread(x):
        return x*(valid*scale)[...,None]
    cur = now[:,:,1:]
    order = (cur-reverse[:,:,1:])*.5
    before = prev[:,:,1:]
    both = np.where(cur*before>0,np.sign(cur)*np.minimum(abs(cur),abs(before)),0.)
    controls = np.stack((q,np.tanh(cycle.ref.window(m,5)/scale),now[:,:,0]*scale),axis=2)
    return strip.center(np.concatenate((p,controls,spread(destination[:,:,1:]),spread(cur),spread(order),spread(both)),axis=2))


@lru_cache(maxsize=12)
def record(d):
    params = bank(d)
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    g,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(30+d%5,b-a-60,5)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        f = features(raw[a:b],m,ok,p[a:b],params)[t]
        z = np.concatenate((f,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        g += rows.T@rows
        n += len(rows)
    return g/n


def train(end):
    return np.r_[bank(end),cycle.ref.fit(np.stack([record(d) for d in range(4,end)]),C,D).ravel()]


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],m,ok,p[a:b],bank[:-D*6])
        out[a:b] = (x@bank[-D*6:].reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    out = {}
    for d in (8,12):
        a = table(range(d))[0]
        b,_,w = table(range(d,d+4))
        out[d+1] = ((a*b*w).sum((1,2))/np.sqrt((a*a*w).sum((1,2))*(b*b*w).sum((1,2)))).tolist()
    (OUT/'timing_replication.json').write_text(json.dumps(out,indent=2))
