"""Translate matched peer responses from donor risk units to receiver risk units at event onset."""
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, rel, strip, official
import prepare_dense_staggered_book_events_reference_v1 as ref


base = ref.model
REFERENCE = ref.OUT
OUT = REFERENCE.parent / 'dense_risk_scaled_peer_events_v1'
ARMS = ('fast_volatility_units', 'slow_volatility_units', 'relative_volatility_units')


def read_p(day):
    p = ref.read_p(day)
    if 8 <= day < 16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[day][:,None]
    return p


def scales(motion, valid):
    # Absolute variation is less sensitive to a single outlier than squared returns.
    n = np.vstack((np.zeros(50), valid.cumsum(0)))
    summed = np.vstack((np.zeros(50), (abs(motion)*valid).cumsum(0)))
    t = np.arange(len(motion))+1
    risks = []
    for w in (40,320):
        old = (t-w).clip(0)
        total = summed[t]-summed[old]
        count = n[t]-n[old]
        common = (total.sum(1,keepdims=True)/count.sum(1,keepdims=True).clip(1)).clip(.002)
        local = (total+10*common)/(count+10)
        risks.append(common * (local/common).clip(.25,4.))
    return *risks, np.sqrt(risks[0]/risks[1])


def states(raw,m,valid,p,t):
    return base.states(raw,m,valid,p,t,scales)


def record(day):
    return base.record(day,states,read_p)


def train(end):
    return base.train(end,record)


def signals(raw,grid,p,bank):
    return base.signals(raw,grid,p,bank,states)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
