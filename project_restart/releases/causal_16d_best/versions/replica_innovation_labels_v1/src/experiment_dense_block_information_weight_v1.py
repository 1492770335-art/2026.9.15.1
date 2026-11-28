"""Test how much settled price persistence changes when earlier blocks are weighted by realized variance."""
from functools import lru_cache
import sys
import numpy as np
import torch

import experiment_dense_rank_surprise_dependence_v1 as rank
from experiment_dense_directional_market_persistence_v1 import BLOCK, REFERENCE, read_p, event, onset, rel, strip

OUT=REFERENCE.parent/'dense_block_information_weight_v1'
ARMS=('variance_weighted_blocks','relative_to_market_precision','independent_basket_weights')
GROUP=np.random.default_rng(34319).permutation(50).reshape(2,25)


def stats(error):
    common=error.mean(1,keepdims=True)
    var,lag=[],[]
    for v in (common,error-common):
        var.append((v*v).mean())
        lag.append(sum(2*(1-k/12)*(1-k/7)*(v[:-k]*v[k:]).mean() for k in range(1,7)))
    return np.array(var),np.array(lag)


def observe(move,ok,p,scale):
    path=move.cumsum(0)
    gap=(~ok).cumsum(0)
    total=np.zeros((3,2,3))
    ratios=np.zeros_like(total)
    for lo in range(BLOCK,len(p),BLOCK):
        t=np.arange(lo-BLOCK+5,lo,5)
        good=(gap[t]==gap[t-5])&ok[t-5]
        err=path[t]-path[t-5]-scale*p[t-5,:,0]
        err=(err-(err*good).sum(0)/good.sum(0).clip(1))*good
        samples=[stats(err),stats(err[:,GROUP[0]]),stats(err[:,GROUP[1]])]
        for j,(var,lag) in enumerate(samples):
            local=lag/(var+1e-6)
            # The alternate panel supplies the interval weight. An outlier
            # in a single stock cannot mechanically weight its own lag product.
            other=samples[3-j][0] if j else var[::-1]
            w=np.stack((np.ones(2),var,other),axis=1)
            ratios[j]=.77*ratios[j]+w*local[:,None]
            total[j]=.77*total[j]+w
        if lo>=2*BLOCK:
            est=ratios/total.clip(1e-7)
            change=est-est[:,:,:1]
            own=change[0,0,1]-change[0,1,1]
            alternate=change[0,0,2]-change[0,1,2]
            held_out=(change[1:,0,2]-change[1:,1,2]).mean()
            yield lo,min(lo+BLOCK,len(p)),np.array((own,own-alternate,held_out))


@lru_cache(maxsize=48)
def record(day,scale):
    return rank.day_center(day,scale,observe)


def train(end):
    return rank.bank_parameters(end,record)


def signals(raw,grid,p,bank):
    return rank.readout(raw,grid,p,bank,observe)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
