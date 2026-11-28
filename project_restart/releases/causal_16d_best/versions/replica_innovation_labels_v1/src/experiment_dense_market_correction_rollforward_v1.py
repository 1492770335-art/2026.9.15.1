"""Extend the verified medium market correction across only the future legs where its timing is known."""
from functools import lru_cache
import sys
import numpy as np
import torch

import prepare_dense_conditional_book_kernel_reference_v1 as ref
import experiment_dense_analogue_block_payoff_v1 as medium
from experiment_dense_clock_coincidence_risk_v1 import event,onset,rel,strip,official

REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_market_correction_rollforward_v1'
ARMS=('first_twenty_only','remaining_known_legs','complete_rollforward')


def read_p(day):
    p=ref.read_p(day)
    if 8<=day<16:
        p=p.copy()
        p[:,:,0]+=official.short_market_changes()[day][:,None]
    return p


@lru_cache(maxsize=12)
def state_fit(day):
    p=read_p(day)
    grid=onset.settled.load(day,'time')
    rows=[]
    for a,b in rel.fields.parent.sessions(grid):
        q=p[a:b].mean(1,dtype=float)
        t=np.arange(day%5,b-a-60,5)
        if len(t):
            rows.append(np.column_stack((q[t],q[t+20,1],q[t+40,1])))
    x=np.concatenate(rows)
    return x.T@x/len(x)


def train(end):
    gs=np.stack([state_fit(d) for d in range(4,end)])
    g=gs.mean(0)
    sd=g.diagonal()[:3].clip(1e-6)**.5
    w=np.linalg.solve(g[:3,:3]/np.outer(sd,sd)+.25*np.eye(3),g[:3,3:]/sd[:,None])/sd[:,None]
    return np.r_[w.ravel(),medium.train(end)]


def signals(raw,grid,p,bank):
    w=bank[:6].reshape(3,2)
    unit=bank[6:9]
    sd=bank[9:9+medium.K]
    history=bank[9+medium.K:].reshape(-1,medium.K+4)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b].mean(1,dtype=float)
        forecast=q@w
        for lo,hi,key in medium.descriptors(m,ok,p[a:b],unit):
            matched,vol,_=medium.lookup(key,history,sd)
            coef=.18*np.tanh(matched[0]-vol[0])*unit[1]/unit[2]
            # The block coefficient remains available for the next forecast
            # strictly before the next scheduled update, even on truncated prefixes.
            t=np.arange(lo,hi)
            known=(t[:,None]+np.array([20,40]))<(lo+medium.BLOCK)
            first=coef*q[t,1]
            later=coef*(forecast[t]*known).sum(1)
            out[a+t,...,1]=np.stack((first,later,first+later),axis=1)[:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
