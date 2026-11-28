"""Recover the last completed visit to the current price level and compare its exit book to its re-entry book."""
from functools import partial
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, rel, strip
from experiment_dense_risk_scaled_peer_events_v1 import ref, base, read_p, REFERENCE


OUT = REFERENCE.parent / 'dense_returning_pressure_spells_v1'
ARMS = ('exit_to_entry_pressure', 'reversed_pressure', 'pressure_against_excursion')
STOCK = np.arange(50)


def history(cell,ok):
    n = len(cell)
    stamp = np.arange(n)[:,None]
    changed = np.vstack((np.ones((1,50),bool), (cell[1:]!=cell[:-1]) | (~ok[1:]) | (~ok[:-1])))
    entry = np.maximum.accumulate(np.where(changed,stamp,0),axis=0)
    # Organize visits by stock and price; each link points strictly backwards.
    stocks, starts = np.nonzero(changed.T)
    ends = np.r_[starts[1:]-1,n-1]
    ends[np.r_[stocks[1:]!=stocks[:-1],True]] = n-1
    levels = cell[starts,stocks]
    order = np.lexsort((starts,levels,stocks))
    before,after = order[:-1],order[1:]
    pair = (stocks[before]==stocks[after]) & (levels[before]==levels[after])
    previous = np.full_like(cell,-1)
    previous[starts[after[pair]],stocks[after[pair]]] = ends[before[pair]]
    return entry,previous


def state_inputs(raw,m,ok,p,t,unit):
    path = m.cumsum(0)
    cell = np.rint(path/unit).astype(np.int64)
    entry,last = history(cell,ok)
    origin = entry[t]
    prev = last[origin,STOCK]
    prior = prev.clip(0)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    valid = (prev>=0) & (gaps[t+1]==gaps[prior,STOCK])
    fields = base.risk.walk.ref.model.FIELDS
    b = np.tanh(.7*np.arcsinh(raw[...,fields].astype(float)))
    pressure = b[...,::2]-b[...,1::2]
    control = base.links.center(np.concatenate((p[t],pressure[t]),axis=2),ok[t])
    components = np.stack((pressure[...,:2].mean(2),pressure[...,2:].mean(2),pressure[...,0]-pressure[...,1]),axis=2)
    prefix = np.concatenate((np.zeros_like(components[:1]),components.cumsum(0)))
    def recent(at):
        start = (at-3).clip(0)
        return (prefix[at+1,STOCK]-prefix[start,STOCK])/(at-start+1)[...,None]
    old,arrived = recent(prior),recent(origin)
    # Signed mean price displacement while away, in units of the stock's own price grid.
    sums = np.vstack((np.zeros(50),cell.cumsum(0)))
    elapsed = (origin-prior-1).clip(1)
    displacement = (sums[origin,STOCK]-sums[prior+1,STOCK])/elapsed - cell[t]
    weight = valid * np.exp(-elapsed/240.-(t[:,None]-origin)/35.)*(1-np.exp(-elapsed/6.))
    return control,old,arrived,weight,displacement


def states(raw,m,ok,p,t,unit):
    control,old,arrived,weight,excursion = state_inputs(raw,m,ok,p,t,unit)
    change = arrived-old
    reverse = np.maximum(-arrived*old,0)
    against = np.maximum(-arrived*excursion[...,None],0)
    variants = (change, change*np.tanh(4*reverse), change*np.tanh(against))
    return control,[base.links.center(weight[...,None]*v,ok[t]) for v in variants]


def fit_states(raw,m,ok,p,t):
    return states(raw,m,ok,p,t,base.risk.walk.widths())


def record(day):
    return base.record(day,fit_states,read_p)


def train(end):
    return np.r_[base.risk.walk.widths(),base.train(end,record).ravel()]


def signals(raw,grid,p,bank):
    return base.signals(raw,grid,p,bank[50:].reshape(2,3,11),partial(states,unit=bank[:50]))


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
