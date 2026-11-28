"""Align other stocks' observed returns to the age of the receiver's latest pressure event."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_market_conditional_graph_risk_v1 import event, onset, rel, strip
import experiment_dense_book_jump_hurdle_v1 as risk


REFERENCE, read_p, links = risk.REFERENCE, risk.read_p, risk.links
OUT = REFERENCE.parent / 'dense_staggered_book_events_v1'
ARMS = ('age_aligned_peers', 'state_matched_peers', 'state_match_increment')
C, D = 8, 3
IDS = np.arange(50)


def states(raw, motion, valid, p, t, scale_builder=None, payout_builder=None):
    time = np.arange(len(raw))
    levels = np.tanh(.7*np.arcsinh(raw[:, :, risk.walk.ref.model.FIELDS].astype(float)))
    imbalance = levels[..., ::2] - levels[..., 1::2]
    # The trigger compares a fast and a slow pressure average at the same timestamp.
    pressure = imbalance[..., :2].mean(2)
    path = motion.cumsum(0)
    prefix = np.vstack((np.zeros(50), pressure.cumsum(0)))
    def mean(size):
        return (prefix[time+1]-prefix[(time+1-size).clip(0)])/np.minimum(time+1,size)[:,None]
    impulse = mean(4) - mean(24)
    side = np.sign(impulse)
    gaps = np.vstack((np.zeros(50,int), (~valid).cumsum(0)))
    recent = (time-5).clip(0)
    region = side * (np.abs(impulse) > .09)
    fire = (region != 0) & (region != region[(time-1).clip(0)]) & (gaps[time+1]==gaps[(time-23).clip(0)])
    last = np.maximum.accumulate(np.where(fire,time[:,None],-10000),axis=0)
    origin = last[t].clip(0)
    age = t[:,None] - last[t]
    sign = side[origin,IDS]
    deep = imbalance[...,2:].mean(2)
    # Two binary fields describe book shape and whether price had already moved at onset.
    code = ((deep*side > 0).astype(int) + 2*((path-path[recent])*side > 0))
    key = code[origin,IDS]
    ctrl = links.center(np.concatenate((p[t], imbalance[t]),axis=2),valid[t])
    payouts = payout_builder(p, path) if payout_builder else None
    scales = scale_builder(motion, valid) if scale_builder else None
    pool = np.zeros((len(t),50,3,D))
    for phase, limit in ((0,5),(5,10),(10,20),(20,40)):
        query = (age>=phase) & (age<limit) & (gaps[t+1]==gaps[origin,IDS])
        for j,(start,end) in enumerate(((0,5),(5,20),(20,60))):
            cutoff = t-phase-end
            prev = last[cutoff.clip(0)]
            donor = prev.clip(0)
            lo = (donor+phase+start).clip(max=len(raw)-1)
            hi = (donor+phase+end).clip(max=len(raw)-1)
            available = (prev>=0) & (cutoff[:,None]>=prev) & (hi<=t[:,None]) & (gaps[hi+1,IDS]==gaps[donor,IDS])
            confidence = available*np.exp(-np.maximum(cutoff[:,None]-prev,0)/30.)
            cash = (path[hi,IDS]-path[lo,IDS]) * side[donor,IDS]
            if payouts is not None:
                at = (donor+phase).clip(max=len(raw)-1)
                values = payouts(cash,side[donor,IDS],at,j)
                donor_key = code[donor,IDS]
                for category in range(4):
                    weights = confidence*(donor_key==category)
                    mass = weights.sum(1,keepdims=True)-weights
                    paid = values*weights[...,None]
                    pooled = (paid.sum(1,keepdims=True)-paid)/(mass+3)[...,None]
                    pool[:,:,:,j] += ((key==category)*query*sign)[...,None]*pooled
                continue
            total = confidence.sum(1,keepdims=True)-confidence
            value = ((cash*confidence).sum(1,keepdims=True)-cash*confidence)/(total+3)
            matched = np.zeros_like(value)
            support = np.zeros_like(value)
            donor_key = code[donor,IDS]
            for category in range(4):
                weights = confidence*(donor_key==category)
                mass = weights.sum(1,keepdims=True)-weights
                amount = (cash*weights).sum(1,keepdims=True)-cash*weights
                matched += (key==category)*amount/(mass+3)
                support += (key==category)*mass/(mass+3)
            gate = query*sign
            if scales is not None:
                for k,scale in enumerate(scales):
                    amount_per_risk = cash / scale[donor,IDS]
                    transfer = np.zeros_like(value)
                    for category in range(4):
                        weights = confidence*(donor_key==category)
                        mass = weights.sum(1,keepdims=True)-weights
                        paid = weights*amount_per_risk
                        transfer += (key==category)*(paid.sum(1,keepdims=True)-paid)/(mass+3)
                    pool[:,:,k,j] += gate*(transfer*scale[origin,IDS]-matched)
                continue
            pool[:,:,0,j] += gate*value
            pool[:,:,1,j] += gate*matched
            pool[:,:,2,j] += gate*(matched-support*value)
    return ctrl,[links.center(pool[:,:,k],valid[t]) for k in range(3)]


@lru_cache(maxsize=12)
def record(day, designer=states, reader=read_p):
    raw, grid, y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = reader(day)
    beta = strip.daily_scale(p,y).clip(.2,4.)
    c = strip.physical()[1]
    moments = np.zeros((2,3,C+D+2,C+D+2))
    n = 0
    for a,b,m,valid in event.sessions(raw,grid):
        t = np.arange(50+day%5,b-a-60,5)
        ctrl,parts = designer(raw[a:b],m,valid,p[a:b],t)
        error = y[a+t].astype(float)/beta - p[a+t]
        for h in range(2):
            j = h+1
            far = error[...,j]-error[...,h]*c[j]/c[h]*beta[h]/beta[j]
            labels = links.center(np.stack((error[...,j],far),axis=2),valid[t])
            for k,part in enumerate(parts):
                x = np.concatenate((ctrl,part,labels),axis=2)[valid[t]]
                moments[h,k] += x.T@x
        n += valid[t].sum()
    return moments/n


def train(end, reader=record):
    dates = np.stack([reader(d) for d in range(4,end)])
    weights = np.zeros((2,3,C+D))
    for h in range(2):
        for k in range(3):
            days = dates[:,h,k]
            g = days.mean(0)
            whitener = links.base.whitening(g[:C,:C],.03)
            v = np.eye(C+D+2)[:,C:C+D].copy()
            v[:C] = -whitener@(whitener.T@g[:C]@v)
            v = v@links.base.whitening(v.T@g@v,.75)
            responses = v.T@days[:,:,-2:]
            avg = responses.mean(0)
            total,far = (avg*avg**2/(avg**2+responses.var(0)/len(days)+1e-9)).T
            weights[h,k] = v[:-2]@(total*(total*far>0))
    return weights


def signals(raw,grid,p,bank,designer=states):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,valid in event.sessions(raw,grid):
        ctrl,parts = designer(raw[a:b],m,valid,p[a:b],np.arange(b-a))
        for k, part in enumerate(parts):
            out[a:b,:,k] = np.concatenate((ctrl,part),axis=2)@bank[:,k].T
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
