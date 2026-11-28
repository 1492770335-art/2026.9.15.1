"""Condition market forward risk on near-horizon cash in independent panels of horizon-specific stock portfolios."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_horizon_graph_filter_v1 as graph
import experiment_dense_spectral_forward_cash_v1 as forward
import experiment_dense_multitaper_common_surprises_v1 as spec
import prepare_dense_horizon_graph_reference_v1 as ref
from experiment_dense_market_correction_rollforward_v1 import event, onset, rel, strip, official
from experiment_dense_disjoint_settled_error_transfer_v1 import units


REFERENCE = ref.OUT
OUT = REFERENCE.parent/'dense_market_conditional_graph_risk_v1'
ARMS = ('conditional_market_risk','factor_to_market_hedge','two_panel_agreement')
SPLIT = np.random.default_rng(7419).permutation(50).reshape(2,25)
K = 3


def read_p(d):
    p = ref.read_p(d)
    if 8<=d<16:
        p = p.copy()
        p[...,0] += official.short_market_changes()[d][:,None]
    return p


@lru_cache(maxsize=1)
def modes():
    g = np.mean([graph.record(d) for d in range(4)],axis=0)
    near = g[:50,:50]
    sd = near.diagonal().clip(.0001)**.5
    val,vec = np.linalg.eigh(near/np.outer(sd,sd))
    axes = (vec/sd[:,None])/np.sqrt(val.clip(0)+.5)
    val,vec = np.linalg.eigh(axes.T@(graph.covariance(g,2,True)-near)@axes)
    modes = axes@vec[:,[-1,-2,0]]
    weights = np.zeros((2,50,K+1))
    for side,ids in enumerate(SPLIT):
        v = modes[ids]-modes[ids].mean(0)
        weights[side,ids,1:] = v/abs(v).sum(0).clip(.0001)
        weights[side,ids,0] = 1/len(ids)
    return weights


def readings(panel,j):
    # panel[side, taper, frequency, portfolio]. Both sides have disjoint stocks.
    near = forward.LEGS[:,:j].sum(1)
    late = forward.LEGS[:,j]
    transforms = np.stack([np.concatenate((p*near[None,:,None],p[:,:,:1]*late[None,:,None]),axis=2) for p in panel])
    def gram(a,b):
        return np.einsum('kfi,kfj,f->ij',a.conj(),b,spec.MULT,optimize=True).real/len(a)
    aa,bb = (gram(z,z) for z in transforms)
    ab = gram(*transforms)
    # Positive semidefinite and shared across the two independent portfolios.
    shared = .25*(aa+bb+ab+ab.T)
    def solve(g):
        var = g.diagonal().clip(.00001)
        beta = np.linalg.solve(g[:-1,:-1]+.35*np.diag(var[:-1]),g[:-1,-1])
        alone = g[0,-1]/(1.35*var[0])
        risk = var[-1]-2*beta@g[:-1,-1]+beta@g[:-1,:-1]@beta
        base = var[-1]-2*alone*g[0,-1]+alone**2*var[0]
        return np.r_[np.log((risk+.001)/(base+.001)),beta[1:]]
    u,v = solve(aa),solve(bb)
    combined = solve(shared)
    agree = np.sign(u[1:])*np.minimum(abs(u[1:]),abs(v[1:]))*(u[1:]*v[1:]>0)
    return np.r_[combined,agree]


def windows(move,ok,p,unit,w):
    path = move.cumsum(0)
    gaps = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t = np.arange(5,len(p),5)
    good = gaps[t+1]==gaps[t-5]
    err = (path[t]-path[t-5]-p[t-5,:,0]*unit)*good
    values = np.stack([err@axes for axes in w])
    for lo in range(485,len(move),120):
        end = t.searchsorted(lo)
        f = np.stack([spec.spectral(z[end-spec.N:end]) for z in values])
        yield lo,min(len(move),lo+120),np.stack([readings(f,j) for j in (1,2)])


@lru_cache(maxsize=48)
def record(day,unit):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    p = read_p(day)
    values = [s for a,b,m,ok in event.sessions(raw,grid) for _,_,s in windows(m,ok,p[a:b],unit,modes())]
    return np.median(values,axis=0)


def train(end):
    unit = np.median([units(d) for d in range(4,end)],axis=0)
    prior = np.median([record(d,float(unit[0])) for d in range(4,end)],axis=0)
    return np.r_[unit,modes().ravel(),prior.ravel()]


def signals(raw,grid,p,bank):
    unit = bank[:3]
    weights = bank[3:-14].reshape(2,50,K+1)
    center = bank[-14:].reshape(2,7)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q = p[a:b]*unit
        for lo,hi,state in windows(m,ok,p[a:b],unit[0],weights):
            diff = state-center
            for j in (1,2):
                near = q[lo:hi,:,j-1]
                tail = (q[lo:hi,:,j]-near).mean(1)
                f = near@weights.mean(0)[:,1:]
                row = diff[j-1]
                z = np.column_stack((tail*np.tanh(3*row[0]),f@np.tanh(row[1:4]),f@np.tanh(row[4:])))
                out[a+lo:a+hi,:,:,j-1] = .20*z[:,None]/unit[j]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
