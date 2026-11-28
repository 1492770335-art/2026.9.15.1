"""Forecast distal market risk from simultaneous and delayed unexpected price refreshes."""
from functools import lru_cache
import sys
import numpy as np
import torch

import prepare_dense_conditional_price_age_reference_v1 as ref
from experiment_dense_market_and_relative_distal_v1 import strip, event, onset, rel, official

REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_clock_coincidence_risk_v1'
ARMS=('simultaneous_price_clocks','delayed_price_clocks','short_minus_delayed_coincidence')
BLOCK=120
AGES=(1,3,7,15,31)
SHAPE=(8,50,len(AGES)+1)


def read_p(day):
    p=ref.read_p(day)
    if 8<=day<16:
        p=p.copy()
        p[:,:,0]+=official.short_market_changes()[day][:,None]
    return p


def clock(move,ok,grid):
    t=np.arange(len(move))[:,None]
    changed=(abs(move)>.003)&ok
    anchor=np.maximum.accumulate(np.where(changed|~ok,t,0),axis=0)
    previous=np.vstack((anchor[:1],anchor[:-1]))
    age=np.searchsorted(AGES,t-previous)
    slot=np.minimum(grid.astype(int)//600,7)[:,None]
    ids=np.ravel_multi_index(np.broadcast_arrays(slot,np.arange(50)[None],age),SHAPE)
    return ids,changed


@lru_cache(maxsize=1)
def hazard():
    table=np.zeros((np.prod(SHAPE),2))
    for day in range(4):
        raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
        for a,b,m,ok in event.sessions(raw,grid):
            ids,changed=clock(m,ok,grid[a:b])
            table[:,0]+=np.bincount(ids[ok],weights=changed[ok],minlength=len(table))
            table[:,1]+=np.bincount(ids[ok],minlength=len(table))
    table=table.reshape(*SHAPE,2)
    parent=table.sum(0,keepdims=True)
    hazard=(table[...,0]+80*(parent[...,0]+.5)/(parent[...,1]+1))/(table[...,1]+80)
    return hazard.ravel()


def cross_pair(x,k):
    left=x[:len(x)-k];right=x[k:]
    pairs=left.sum(1)*right.sum(1)-(left*right).sum(1)
    return pairs.mean()/50


def keys(move,ok,grid,haz):
    ids,hit=clock(move,ok,grid)
    p=haz[ids]
    miss=(hit-p)*ok
    smooth=np.zeros(6)
    for lo in range(BLOCK,len(move),BLOCK):
        sl=slice(lo-BLOCK,lo)
        e=miss[sl]
        e=e-e.mean(0)
        risk=(p[sl]*(1-p[sl])*ok[sl]).mean()+1e-7
        coinc=cross_pair(e,0)/risk
        short=np.mean([cross_pair(e,k) for k in (1,2,4)])/risk
        late=np.mean([cross_pair(e,k) for k in (8,12,16)])/risk
        m=move[sl]
        common=m.mean(1)
        frac=np.log((common.var()+1e-6)/(m.var(0).mean()+1e-6))
        val=np.array((hit[sl].mean(),frac,coinc,short,late,short-late))
        smooth=.72*smooth+.28*val
        if lo>=2*BLOCK:
            x=np.r_[1.,lo/1200,smooth/(1-.72**(lo//BLOCK))]
            yield lo,min(lo+BLOCK,len(move)),x


@lru_cache(maxsize=16)
def record(day):
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    rows=[]
    for a,b,m,ok in event.sessions(raw,grid):
        path=m.cumsum(0)
        for lo,hi,x in keys(m,ok,grid[a:b],hazard()):
            if hi-lo!=BLOCK:
                continue
            # Cash densities use only the future portion of historical blocks
            # during fitting. Feature extraction in deployment needs no targets.
            t=np.arange(lo,hi-60,5)
            ret=path[t+60]-path[t+20]
            v=ret.mean(1)
            y=np.log((v@v/len(t)+.0003)/(((ret-v[:,None])**2).mean()+.0003))
            rows.append(np.r_[x,np.clip(y,-7,2)])
    return np.stack(rows)


def train(end):
    rows=[record(d) for d in range(4,end)]
    grams=np.stack([x.T@x/len(x) for x in rows])
    g=grams.mean(0)
    ans=[]
    for ids in ((4,),(5,6),(4,7)):
        v=np.eye(8)[:,ids]
        ctrl=g[:4,:4]+.025*np.diag(g.diagonal()[:4].clip(1e-5))
        v[:4]=-np.linalg.solve(ctrl,g[:4,ids])
        k=v.T@g[:8,:8]@v
        rhs=grams[:,:8,-1]@v
        mean=rhs.mean(0)
        shrink=mean**2/(mean**2+rhs.var(0)/len(rows)+1e-10)
        ans.append(v@np.linalg.solve(k+np.diag(k.diagonal().clip(.001)),mean*shrink))
    return np.r_[hazard(),np.stack(ans).ravel()]


def signals(raw,grid,p,bank):
    split=np.prod(SHAPE)
    coeff=bank[split:].reshape(3,8)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        market=p[a:b,:,2].mean(1,dtype=float)
        for lo,hi,x in keys(m,ok,grid[a:b],bank[:split]):
            scale=.3*np.tanh(coeff@x)
            out[a+lo:a+hi,...,1]=market[lo:hi,None,None]*scale
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
