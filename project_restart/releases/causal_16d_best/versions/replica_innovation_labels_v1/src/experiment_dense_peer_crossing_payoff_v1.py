"""Model expected terminal displacement by whether a lagging stock subsequently hits a barrier."""
from functools import lru_cache, partial
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip,official
import experiment_dense_peer_crossing_clock_v1 as clock
import prepare_dense_peer_crossing_clock_reference_v1 as ref

base,REFERENCE = clock.base,ref.OUT
OUT = REFERENCE.parent/'dense_peer_crossing_payoff_v1'
ARMS = ('changes_in_hitting_probability','conditional_terminal_move','combined_terminal_move')
SHAPE = (3,2,50,50,3)


def read_p(day):
    p = ref.read_p(day)
    if 8<=day<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[day][:,None]
    return p


@lru_cache(maxsize=16)
def cash(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    displacements = []
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(30+day%8,b-a-60,8)
        path = m.cumsum(0)/base.risk.walk.widths()
        r = np.stack([path[t+h]-path[t] for h in clock.LAGS],axis=2)
        # Align outcomes to the two directional pseudo-observations at each time.
        displacements.extend((-r,r))
    return displacements


@lru_cache(maxsize=16)
def payouts(day):
    totals = np.zeros((clock.BINS,50,3,2))
    for (_,key,_,_,hit,seen),r in zip(clock.examples(day),cash(day)):
        for h in range(3):
            for k in range(2):
                np.add.at(totals[:,:,h,k],(key,clock.NAMES),r[...,h]*seen[...,h]*(hit[...,h]==k))
    return totals


def expected(totals,count):
    all_obs,hit = count[...,0],count[...,1]
    n = np.stack((all_obs-hit,hit),axis=3)
    prior = totals.sum(1,keepdims=True)/n.sum(1,keepdims=True).clip(1)
    return (totals+32*prior)/(n+32)


@lru_cache(maxsize=16)
def edges(end):
    count = np.stack([clock.counts(d) for d in range(end)])
    totals = np.stack([payouts(d) for d in range(end)])
    all_count,all_cash = count.sum(0),totals.sum(0)
    coef = []
    for d in range(end):
        hist = all_count-count[d]
        prob = clock.probs(hist)
        mean = expected(all_cash-totals[d],hist)
        daily = np.zeros(SHAPE)
        power = np.zeros(clock.WEIGHT_SHAPE)
        for (_,key,donors,_,hit,seen),r in zip(clock.examples(d),cash(d)):
            mu = mean[key,clock.NAMES]
            surprise = (hit-prob[key,clock.NAMES])*(mu[...,1]-mu[...,0])
            # The three pieces sum to realized displacement minus its baseline expectation.
            errors = [surprise,hit*(r-mu[...,1]),(~hit)*(r-mu[...,0])]
            for k,x in enumerate(donors):
                for h in range(3):
                    mask = seen[...,h].astype(float)
                    for j,error in enumerate(errors):
                        daily[j,k,:,:,h] += x.T@(error[...,h]*mask)
                    power[k,:,:,h] += (x*x).T@mask
        coef.append(daily/(power+30))
    coef = np.stack(coef)
    mu = coef.mean(0)
    edge = mu*mu**2/(mu**2+2*coef.var(0)/end+1e-7)/np.sqrt(49)
    edge[:,:,clock.NAMES,clock.NAMES] = 0.
    return edge


def states(raw,m,ok,p,t,unit,graph):
    path = m.cumsum(0)
    moves = [path[t]-path[(t-h).clip(0)] for h in clock.LAGS]
    levels = np.tanh(.7*np.arcsinh(raw[t][:,:,base.risk.walk.ref.model.FIELDS[:4]].astype(float)))
    ctrl = base.links.center(np.concatenate((p[t],np.stack(moves,axis=2),levels[...,::2]-levels[...,1::2]),axis=2),ok[t])
    z = np.zeros((3,len(t),50,3))
    for sign,_,donors,risk,_,_ in clock.observations(m,ok,t,unit,False):
        for k,x in enumerate(donors):
            for j in range(3):
                move = (x@graph[j,k].reshape(50,-1)).reshape(len(t),50,3)
                z[j] += sign*risk[...,None]*move*unit[:,None]
    return ctrl,[base.links.center(part,ok[t]) for part in (z[0],z[1]+z[2],z.sum(0))]


@lru_cache(maxsize=12)
def record(day):
    return base.record(day,partial(states,unit=base.risk.walk.widths(),graph=edges(day)),read_p)


def train(end):
    return np.r_[base.risk.walk.widths(),edges(end).ravel(),base.train(end,record).ravel()]


def signals(raw,grid,p,bank):
    designer = partial(states,unit=bank[:50],graph=bank[50:-66].reshape(SHAPE))
    return base.signals(raw,grid,p,bank[-66:].reshape(2,3,11),designer)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
