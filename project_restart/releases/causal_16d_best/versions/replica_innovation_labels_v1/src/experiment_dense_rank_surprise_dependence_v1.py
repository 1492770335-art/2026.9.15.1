"""Separate marginal tail size from lagged rank dependence of settled five-tick forecast surprises."""
from functools import lru_cache
import sys
import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import rankdata

from experiment_dense_directional_market_persistence_v1 import BLOCK, REFERENCE, read_p, event, onset, rel, strip, unit

OUT=REFERENCE.parent/'dense_rank_surprise_dependence_v1'
ARMS=('rank_persistence','persistence_beyond_ranks','extreme_rank_continuation')


def statistics(error):
    common=error.mean(1,keepdims=True)
    parts=(common,error-common)
    result=[]
    for v in parts:
        # Every rank is computed on a closed block; the next decision cannot
        # observe ranks or extrema of its own unfinished block.
        ranked=(rankdata(v,axis=0)-.5)/len(v)
        gauss=ndtri(ranked)
        raw=v-v.mean(0)
        tail=np.where(ranked>.8,1.,np.where(ranked<.2,-1.,0.))
        states=np.stack((gauss,raw,tail),axis=-1)
        variance=(states**2).mean((0,1))
        lagged=np.zeros(3)
        for k in range(1,7):
            lagged+=2*(1-k/12)*(1-k/7)*(states[:-k]*states[k:]).mean((0,1))
        result.append(np.stack((lagged,variance)))
    return np.stack(result)


def block_statistics(move,ok,p,scale,estimator):
    path=move.cumsum(0)
    gap=(~ok).cumsum(0)
    acc=np.zeros((2,2,3))
    for lo in range(BLOCK,len(p),BLOCK):
        t=np.arange(lo-BLOCK+5,lo,5)
        good=(gap[t]==gap[t-5])&ok[t-5]
        err=path[t]-path[t-5]-scale*p[t-5,:,0]
        # Fill unobserved surprises with their within-block stock mean, so they
        # contribute no stock-centered surprise.
        mean=(err*good).sum(0)/good.sum(0).clip(1)
        err=(err-mean)*good
        acc=.77*acc+estimator(err)
        if lo>=2*BLOCK:
            ratio=acc[:,0]/acc[:,1].clip(1e-6)
            yield lo,min(lo+BLOCK,len(p)),(ratio[0]-ratio[1]).clip(-5,5)


def observe(move,ok,p,scale):
    for lo,hi,(rank,raw,tail) in block_statistics(move,ok,p,scale,statistics):
        yield lo,hi,np.array((rank,raw-rank,tail-rank))


def day_center(day,scale,observer):
    p=read_p(day)
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    rows=[z for a,b,m,ok in event.sessions(raw,grid) for _,_,z in observer(m,ok,p[a:b],scale)]
    return np.median(rows,axis=0)


@lru_cache(maxsize=48)
def record(day,scale):
    return day_center(day,scale,observe)


def bank_parameters(end,recorder):
    scale=float(np.median([unit(d) for d in range(4,end)]))
    return np.r_[scale,np.median([recorder(d,scale) for d in range(4,end)],axis=0)]


def train(end):
    return bank_parameters(end,record)


def readout(raw,grid,p,bank,observer):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        market=p[a:b,:,1:].mean(1,dtype=float)
        for lo,hi,z in observer(m,ok,p[a:b],bank[0]):
            weight=.17*np.tanh(z-bank[1:])
            out[a+lo:a+hi]=market[lo:hi,None,None,:]*weight[None,None,:,None]
    return out


def signals(raw,grid,p,bank):
    return readout(raw,grid,p,bank,observe)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
