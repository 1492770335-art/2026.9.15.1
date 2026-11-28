"""Combine the price-magnitude persistence of the market and ordered activity of individual names."""
from functools import lru_cache
import sys
import numpy as np
import torch

import prepare_dense_analogue_block_reference_v1 as ref
import experiment_dense_rank_surprise_dependence_v1 as rank
import experiment_dense_block_information_weight_v1 as weights
import experiment_dense_signed_activity_paths_v1 as activity
from experiment_dense_directional_market_persistence_v1 import event, onset, rel, strip, official

REFERENCE=ref.OUT
OUT=REFERENCE.parent/'dense_market_and_relative_distal_v1'
ARMS=('common_tail_and_volatility','relative_ordered_activity','common_and_relative')


def read_p(day):
    p=ref.read_p(day)
    if 8<=day<16:
        p=p.copy()
        p[:,:,0]+=official.short_market_changes()[day][:,None]
    return p


@lru_cache(maxsize=12)
def local_record(day):
    # The relative path readout uses the updated medium forecast at every fold.
    raw,grid,y=[onset.settled.load(day,k) for k in ('x','time','y')]
    p=read_p(day)
    return activity.paths.panel_moments(day,p,raw,grid,y,activity.features)


def train(end):
    local=activity.paths.train_weights([local_record(d) for d in range(4,end)])[1]
    return np.r_[rank.train(end),weights.train(end),local.ravel()]


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b]
        ctrl,design=activity.features(raw[a:b],m,ok,q)
        local=(np.concatenate((ctrl,design[1]),axis=2)@bank[8:].reshape(12,2))[:,:,1]
        local-=local.mean(1,keepdims=True)
        common=np.zeros((len(q),1))
        market=q[:,:,2].mean(1,dtype=float)
        ranks=rank.observe(m,ok,q,bank[0])
        weighted=weights.observe(m,ok,q,bank[4])
        for (lo,hi,z),(first,last,other) in zip(ranks,weighted):
            assert (lo,hi)==(first,last)
            # Combine the amplitude information lost under rank normalization with
            # the independent-basket estimate of volatility-weighted persistence.
            coefficient=.085*(np.tanh(z[1]-bank[2])+np.tanh(other[2]-bank[7]))
            common[lo:hi,0]=market[lo:hi]*coefficient
        out[a:b,:,0,1]=common
        out[a:b,:,1,1]=local
        out[a:b,:,2,1]=common+local
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
