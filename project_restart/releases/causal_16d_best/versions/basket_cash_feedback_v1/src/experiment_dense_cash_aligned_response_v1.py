"""Estimate horizon-response differences by comparing old and new signals against exactly the same price payment."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_hedged_stopped_teacher_v1 as hedged

pre, strip, event, onset, rel = hedged.pre, hedged.strip, hedged.event, hedged.onset, hedged.rel
REFERENCE, read_p, C, D = hedged.REFERENCE, hedged.read_p, hedged.C, hedged.D
OUT = REFERENCE.parent/'dense_cash_aligned_response_v1'
ARMS = ('direct_payments','relative_to_remote_information','multiscale_payment_differences')
K = 32
S = D-C


@lru_cache(maxsize=12)
def record(day):
    raw, grid = [onset.settled.load(day,k) for k in ('x','time')]
    p = read_p(day)
    bank = pre.encoder()
    net = pre.restore(bank)
    law = np.zeros((K,S))
    cov = np.zeros((S,S))
    counts = np.zeros(K)
    size = 0
    for a,b,m,ok in event.sessions(raw,grid):
        q = np.arange(50+day%5,len(m),5)
        x = hedged.hedge(pre.views(raw[a:b],p[a:b],bank,net)[q],ok[q])[:,:,C:]
        path = m.cumsum(0)
        gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        # Keep the payment rows fixed as the information origin moves backward in time.
        ix = np.arange(K-1,len(q)-1)
        paid = path[q[ix+1]]-path[q[ix]]
        valid = gaps[q[ix+1]+1]==gaps[q[ix-K+1]]
        paid = (paid-(paid*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1))*valid
        y = paid.ravel()
        for j in range(K):
            v = x[ix-j].reshape(-1,S)
            law[j] += v.T@y
            counts[j] += valid.sum()
        v = x.reshape(-1,S)
        cov += v.T@v
        size += len(v)
    return cov/size, law/counts[:,None].clip(1)


def shrink(samples):
    mu = samples.mean(0)
    noise = samples.var(0)/len(samples)
    return mu*(mu**2/(mu**2+2*noise+1e-13))


def curves(grad):
    direct = shrink(grad)
    remote = grad[:,20:].mean(1,keepdims=True)
    centered = shrink(grad-remote)
    # Fit a shared response from finite differences of the kernel at lags 5, 20 and 60.
    eye = np.eye(K)
    A, y = [], []
    for lag in (1,4,12):
        diff = grad[:,:-lag]-grad[:,lag:]
        A.extend(eye[:-lag]-eye[lag:])
        y.extend(shrink(diff))
    # Fix the otherwise undetermined common level by removing the remote-lag mean.
    A.append(eye[20:].mean(0)*K)
    y.append(np.zeros(grad.shape[-1]))
    curve = np.linalg.lstsq(np.stack(A),np.stack(y),rcond=None)[0]
    return np.stack((direct,centered,curve),axis=0)


def train(end):
    cov, grads = map(np.stack,zip(*(record(d) for d in range(4,end))))
    g = cov.mean(0)
    s = np.sqrt(g.diagonal().clip(1e-7))
    val,u = np.linalg.eigh(g/np.outer(s,s))
    white = (u/np.sqrt(val.clip(0)+.65))/s[:,None]
    response = curves(grads@white)
    cal = np.stack([hedged.teacher.live.calibration(d) for d in range(4,end)])
    unit = np.median(cal[:,:,0]/cal[:,:,1].clip(1e-8),axis=0).clip(.03,6.)
    co = np.zeros((D,3,2))
    for j,h in enumerate((4,12)):
        co[C:,:,j] = white@response[:,:h].sum(1).T/unit[j]
    return np.r_[pre.encoder(),co.ravel()]


signals = hedged.signals


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
