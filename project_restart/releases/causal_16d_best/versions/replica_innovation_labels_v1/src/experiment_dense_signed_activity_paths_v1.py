"""Preserve price direction in an ordered path of activity and signed forecast revisions."""
from functools import lru_cache
import sys
import numpy as np
import torch

import experiment_dense_ordered_pressure_paths_v1 as paths
from experiment_dense_directional_market_persistence_v1 import REFERENCE, read_p, strip, event, onset, rel

OUT=REFERENCE.parent/'dense_signed_activity_paths_v1'
ARMS=('local_activity_precedes_pressure','peer_activity_precedes_revisions','direction_of_ordered_loop')


def features(raw,move,ok,p):
    q=paths.center(p.astype(float))
    panel=np.tanh(.65*np.arcsinh(raw[:,:,[26,38,41,42,54,55]].astype(float)))
    signed=paths.center(panel[:,:,::2]-panel[:,:,1::2])
    liquidity=(panel[:,:,::2]+panel[:,:,1::2])*.5
    def delta(x):
        return np.diff(x,axis=0,prepend=x[:1])*(ok & np.vstack((ok[:1],ok[:-1])))[...,None]
    book=delta(signed)
    revisions=delta(q)
    flow=delta(liquidity)
    peer=(flow.sum(1,keepdims=True)-flow)/49
    banks=[[],[],[]]
    for tau in (8.,32.):
        banks[0].append(paths.areas(flow,book,tau))
        banks[1].append(paths.areas(peer,revisions,tau))
        banks[2].append(paths.areas(revisions,book,tau)*np.tanh(2*q))
    return np.concatenate((q,signed),axis=2),[paths.center(np.concatenate(x,axis=2)) for x in banks]


@lru_cache(maxsize=12)
def moments(day):
    return paths.day_moments(day,features)


def train(end):
    return paths.train_weights([moments(d) for d in range(4,end)])


def signals(raw,grid,p,bank):
    return paths.readout(raw,grid,p,bank,features)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
