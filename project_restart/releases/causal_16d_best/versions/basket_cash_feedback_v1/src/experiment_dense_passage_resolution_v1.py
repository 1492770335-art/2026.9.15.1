"""Price the extension and complete retracement of a directional price break as distinct outcomes."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch
from scipy.ndimage import minimum_filter1d, maximum_filter1d

import prepare_dense_directed_passage_reference_v1 as best

parent = best.model.parent
event,strip,rel,onset = parent.event,parent.strip,parent.rel,parent.onset
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_passage_resolution_v1'
ARMS = ('peer_resolution_outcomes','newly_settled_peer_resolutions','return_after_resolution')
N = parent.N+8
HORIZONS = (20,60)


def crossings(move,ok,floor):
    path = move.cumsum(0)
    hits = event.signed_events(move,ok,floor)
    signed = hits[...,0].astype(int)-hits[...,1]
    low = minimum_filter1d(path,13,axis=0,origin=6,mode='nearest')
    high = maximum_filter1d(path,13,axis=0,origin=6,mode='nearest')
    width = np.where(signed>0,path-low,high-path)
    return path,signed,width


def settled_notices(move,ok,floor):
    path,signed,width = crossings(move,ok,floor)
    notices = np.zeros((*signed.shape,2))
    # Arrays are indexed by receipt time. At age k, only the price at receipt and the
    # stored barrier k ticks earlier enter; an already paid or interrupted crossing stays closed.
    live = signed!=0
    for age in range(1,min(61,len(move))):
        live = live[:-1]&ok[age:]
        side = signed[:-age]
        change = side*(path[age:]-path[:-age])
        paid = live&(abs(change)>=width[:-age])
        notices[age:,:,0] += side*(paid&(change>0))
        notices[age:,:,1] += side*(paid&(change<0))
        live &= ~paid
    return notices.clip(-1,1)


def resolution(move,ok,floor):
    n = len(move)
    path,signed,widths = crossings(move,ok,floor)
    start,stock = np.nonzero(signed)
    side = signed[start,stock]
    width = widths[start,stock]
    finish = np.full(len(start),n)
    kind = np.zeros(len(start),int)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    for lag in range(1,61):
        at = start+lag
        rows = np.flatnonzero((finish==n)&(at<n))
        u,s = at[rows],stock[rows]
        progress = side[rows]*(path[u,s]-path[start[rows],s])
        complete = (abs(progress)>=width[rows])&(gaps[u+1,s]==gaps[start[rows],s])
        dest = rows[complete]
        finish[dest] = at[dest]
        kind[dest] = np.sign(progress[complete])
    endings = np.full_like(signed,n)
    outcomes = np.zeros_like(signed)
    endings[start,stock] = finish
    outcomes[start,stock] = kind
    # Future classification is mapped back to its origin ONLY for training targets.
    # Online features receive the notice at the time the second barrier actually crosses.
    notices = np.zeros((n,50,2))
    for j,value in enumerate((1,-1)):
        mask = kind==value
        np.add.at(notices,(finish[mask],stock[mask],j),side[mask])
    return path,signed,endings,outcomes,notices.clip(-1,1),gaps


def features(move,ok,p,floor,edges,notices):
    original = parent.design(move,ok,p,floor,edges)
    t = np.arange(len(p))+1
    cs = np.concatenate((np.zeros_like(notices[:1]),notices.cumsum(0)))
    parts = []
    for length in (6,32):
        own = (cs[t]-cs[(t-length).clip(0)])/np.sqrt(length)
        neighbor = (own.transpose(0,2,1)@edges.T).transpose(0,2,1)
        parts.extend((strip.center(own),strip.center(neighbor)))
    return np.concatenate((original,*parts),axis=2)


def targets(t,h,data):
    path,sign,finish,kind,_,gaps = data
    n = len(path)
    stock = np.arange(50)
    future = np.minimum.accumulate(np.where(sign!=0,np.arange(n)[:,None],n)[::-1],axis=0)[::-1]
    first = future[t+1]
    safe = first.clip(max=n-1)
    end = t[:,None]+h
    within = first<=end
    direction = sign[safe,stock]*within
    resolved = within & (finish[safe,stock]<=end)
    outcome = kind[safe,stock]*resolved
    delay = first-t[:,None]
    timing = np.stack([direction*((delay>lo)&(delay<=hi)) for lo,hi in ((0,6),(6,20),(20,60))],axis=2)
    marks = np.stack((direction*(outcome==1),direction*(outcome==-1)),axis=2)
    at1 = np.minimum(first,end)
    at2 = np.minimum(finish[safe,stock],end)
    during = path[at2,stock]-path[at1,stock]
    after = path[t+h]-path[at2,stock]
    total = path[t+h]-path[t]
    return np.concatenate((timing,marks,during[...,None],after[...,None],total[...,None]),axis=2),gaps[t+h+1]==gaps[t]


@lru_cache(maxsize=12)
def record(d):
    floor,edges = event.geometry()
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    gs = np.zeros((2,N+9,N+9))
    counts = np.zeros(2)
    for a,b,move,ok in event.sessions(raw,grid):
        data = resolution(move,ok,floor)
        t = np.arange(55+d%5,b-a-60,5)
        notices = settled_notices(move,ok,floor)
        np.testing.assert_array_equal(notices,data[4])
        x = features(move,ok,p[a:b],floor,edges,notices)[t]
        for j,h in enumerate(HORIZONS):
            out,valid = targets(t,h,data)
            z = np.concatenate((x,out,y[a+t,:,j+1:j+2]),axis=2)
            z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
            row = z[valid]
            gs[j] += row.T@row
            counts[j] += len(row)
    return gs/counts[:,None,None]


def parameters(g,h,j,which):
    # A signed return or extension changes the price of a first crossing only beyond its direction and arrival time.
    mi = [N+3,N+4]
    ci = list(range(8))+[N,N+1]+([N+2] if j==1 else [])
    mark = np.eye(N+5)[:,mi]
    mark[ci] = -parent.solve(h[np.ix_(ci,ci)],h[np.ix_(ci,mi)],.04)
    cash = N+6 if which==2 else N+5
    price = parent.solve(mark.T@h[:N+5,:N+5]@mark,mark.T@h[:N+5,cash],.35)
    price /= np.clip(h[j+1,N+7]/h[j+1,j+1],.05,10)
    # First estimate the excess outcome beyond the usual mix for a given first-crossing time.
    at = [N,N+1]+([N+2] if j==1 else [])
    residual = g[:N,mi]-g[:N,at]@parent.solve(g[np.ix_(at,at)],g[np.ix_(at,mi)],.02)
    controls = [0,1,2,3,4,5,6,7,10,11,18,19,22,23]
    indices = [8,9,12,13,14,15,16,17]
    if which==1:
        controls += indices
        indices = [20,21,24,25]
    if which==2:
        indices += [20,21,24,25]
    v = np.eye(N)[:,indices]
    v[controls] = -parent.solve(g[np.ix_(controls,controls)],g[np.ix_(controls,indices)],.05)
    co = v@parent.solve(v.T@g[:N,:N]@v,v.T@residual,.7)
    return co@price,co,price


def train(end):
    gs = np.stack([record(d) for d in range(4,end)])
    halves = [gs[j::2].mean(0) for j in (0,1)]
    coef = np.stack([[.5*sum(parameters(halves[k][j],halves[1-k][j],j,arm)[0]
                                 for k in (0,1)) for j in range(2)] for arm in range(3)])
    floor,edges = event.geometry()
    return np.r_[floor,edges.ravel(),coef.transpose(2,0,1).ravel()]


def signals(raw,grid,p,bank):
    floor,edges = bank[:50],bank[50:2550].reshape(50,50)
    w = bank[2550:].reshape(N,6)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in event.sessions(raw,grid):
        x = features(move,ok,p[a:b],floor,edges,settled_notices(move,ok,floor))
        out[a:b] = (x@w).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


def audit():
    report = {}
    for end in (8,12):
        g = np.mean([record(d)[1] for d in range(4,end)],axis=0)
        test = np.mean([record(d)[1] for d in range(end,end+4)],axis=0)
        w,co,price = parameters(g,g,1,0)
        at = [N,N+1,N+2]
        mi = [N+3,N+4]
        residual = test[:N,mi]-test[:N,at]@parent.solve(g[np.ix_(at,at)],g[np.ix_(at,mi)],.02)
        corr = np.diag(co.T@residual)/np.sqrt(np.diag(co.T@test[:N,:N]@co)*test.diagonal()[mi]).clip(1e-8)
        report[end] = dict(price_during=price.tolist(),price_after=parameters(g,g,1,2)[2].tolist(),
                           heldout_partial_mark_cov_over_sd=corr.tolist())
    (OUT/'resolution_audit.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
