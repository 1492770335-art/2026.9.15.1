"""Retrieve separated, fully settled episodes at a stock's current price level."""
from functools import partial
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, rel, strip, official
import prepare_dense_price_visit_gru_reference_v1 as ref


gru = ref.model
base = gru.visit.base
REFERENCE = ref.OUT
OUT = REFERENCE.parent/'dense_settled_visit_memory_v1'
ARMS = ('book_difference','settled_peer_free_cash','settled_forecast_error')
LEGS = ((0,5),(5,20),(20,60))
STOCK = np.arange(50)


def read_p(day):
    p = ref.read_p(day)
    if 8<=day<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[day][:,None]
    return p


def episodes(raw,m,ok,p,t,unit,match_state=None):
    path = m.cumsum(0)
    cell = np.rint(path/unit).astype(np.int64)
    levels = np.tanh(.7*np.arcsinh(raw[...,base.risk.walk.ref.model.FIELDS].astype(float)))
    book = levels[...,::2]-levels[...,1::2]
    key_state = cell if match_state is None else match_state(book,p)
    start,_ = gru.visit.history(key_state,ok)
    changed = start == np.arange(len(raw))[:,None]
    origins,names = np.nonzero(changed & ok)
    n = len(raw)
    key = key_state[origins,names]*50+names
    packed = key*(n+1)+origins
    order = np.argsort(packed)
    ordered = packed[order]
    target = key_state[t]*50+STOCK
    cutoff = np.minimum(start[t]-1,t[:,None]-60)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    ctrl = base.links.center(np.concatenate((p[t],book[t]),axis=2),ok[t])
    c = strip.physical()[1]
    market = (m*ok).sum(1)/ok.sum(1).clip(1)
    mark = path-market.cumsum(0)[:,None]
    curve = strip.center(p.astype(float))*c[2]/c
    prior = np.diff(curve,axis=2,prepend=np.zeros_like(curve[:,:,:1]))
    if not len(order):
        return
    for _ in range(5):
        pos = (np.searchsorted(ordered,target*(n+1)+cutoff,side='right')-1).clip(0)
        donor = origins[order[pos]]
        due = (donor+60).clip(max=n-1)
        valid = (key[order[pos]]==target)&(donor<=cutoff)&(t[:,None]-donor<=1200)
        valid &= (gaps[due+1,STOCK]==gaps[donor,STOCK]) & ok[t]
        change = book[t]-book[donor,STOCK]
        w = valid*np.exp(-(t[:,None]-donor).clip(0)/540.)*np.exp(-np.mean(change**2,axis=2)/.13)
        pay = np.stack([mark[(donor+h).clip(max=n-1),STOCK]-mark[(donor+l).clip(max=n-1),STOCK] for l,h in LEGS],axis=2)
        yield ctrl,change,w,pay,prior[donor,STOCK],donor
        cutoff = np.minimum(cutoff,donor-60)


def states(raw,m,ok,p,t,unit):
    accum = np.zeros((len(t),50,3,3))
    total = np.zeros((len(t),50))
    ctrl = np.zeros((len(t),50,8))
    for ctrl,change,w,pay,issued,_ in episodes(raw,m,ok,p,t,unit):
        book = np.stack((change[...,0],change[...,1],change[...,2:].mean(2)),axis=2)
        accum += w[...,None,None]*np.stack((book,pay,pay-issued),axis=2)
        total += w
    accum /= (1.5+total)[...,None,None]
    return ctrl,[base.links.center(accum[:,:,j],ok[t]) for j in range(3)]


def fit_states(raw,m,ok,p,t):
    return states(raw,m,ok,p,t,base.risk.walk.widths())


def record(d):
    return base.record(d,fit_states,read_p)


def train(end):
    return np.r_[base.risk.walk.widths(),base.train(end,record).ravel()]


def signals(raw,grid,p,bank):
    return base.signals(raw,grid,p,bank[50:].reshape(2,3,11),partial(states,unit=bank[:50]))


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
