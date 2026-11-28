"""Price cross-stock portfolio modes whose cash risk persists from five to forty ticks."""
from functools import lru_cache
import sys

import numpy as np
import torch

from experiment_dense_relative_payoff_timing_v1 import REFERENCE, read_p, event, onset, rel, strip, units


OUT = REFERENCE.parent/'dense_horizon_graph_filter_v1'
ARMS = ('unconditional_risk_modes','near_conditioned_risk_modes','replicated_horizon_modes')
D = 50
LEGS = ((0,5),(5,20),(20,60))


@lru_cache(maxsize=16)
def record(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    gram = np.zeros((3*D,3*D))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        path = m.cumsum(0)
        missing = np.vstack((np.zeros(D,int),(~ok).cumsum(0)))
        t = np.arange(30+day%5,len(m)-60,5)
        blocks = []
        for lo,hi in LEGS:
            valid = missing[t+hi+1]==missing[t+lo]
            v = (path[t+hi]-path[t+lo])/np.sqrt((hi-lo)/5)
            v = (v-(v*valid).sum(1,keepdims=True)/valid.sum(1).clip(1)[:,None])*valid
            blocks.append(v)
        z = np.column_stack(blocks)
        gram += z.T@z
        n += len(z)
    return gram/n


def covariance(g,j,conditional):
    s = j*D
    target = g[s:s+D,s:s+D]
    if not conditional:
        return target
    near = g[:s,:s]
    w = np.linalg.solve(near+.45*np.diag(near.diagonal().clip(.0001)),g[:s,s:s+D])
    return target-g[:s,s:s+D].T@w-w.T@g[:s,s:s+D]+w.T@near@w


@lru_cache(maxsize=16)
def filters(end):
    days = np.stack([record(d) for d in range(end)])
    g = days.mean(0)
    # Whiten the short covariance, retaining regularization in both the long
    # and short terms so modes with small short variance do not appear persistent.
    near = g[:D,:D]
    sd = np.sqrt(near.diagonal().clip(.0001))
    eig,vec = np.linalg.eigh(near/np.outer(sd,sd))
    whitening = vec/np.sqrt(eig.clip(0)+.45)
    inverse = (vec*np.sqrt(eig.clip(0)+.45)).T
    profiles = []
    for j in (1,2):
        def matrix(c):
            return whitening.T@(c/np.outer(sd,sd))@whitening
        def offdiag(basis,weight):
            # Transform the unitless stock projection back to raw forecast units.
            filt = (whitening@((basis*weight)@basis.T)@inverse)*sd[None,:]/sd[:,None]
            np.fill_diagonal(filt,0.)
            return filt
        current = []
        for conditional in (False,True):
            contrast = matrix(covariance(g,j,conditional)-near)
            lam,basis = np.linalg.eigh(contrast)
            current.append(offdiag(basis,np.tanh(lam)))
            if conditional:
                folds = [k.mean(0) for k in (days[::2],days[1::2])]
                # Test each mode in partitions of dates with the same basis.
                measure = [np.diag(basis.T@matrix(covariance(k,j,True)-k[:D,:D])@basis) for k in folds]
                u,v = measure
                stable = np.sign(u)*np.minimum(abs(u),abs(v))*(u*v>0)
                current.append(offdiag(basis,np.tanh(stable)))
        profiles.append(current)
    return np.stack(profiles)


def features(p,ok,w,unit):
    q = p.astype(float)
    q -= (q*ok[...,None]).sum(1,keepdims=True)/ok.sum(1).clip(1)[:,None,None]
    q *= ok[...,None]
    near = q[:,:,:-1]*unit[:-1]/unit[1:]
    tail = q[:,:,1:]-near
    out = np.zeros((*ok.shape,3,2))
    for j in range(2):
        for k in range(3):
            out[:,:,k,j] = tail[:,:,j]@w[j,k]
    controls = np.concatenate((ok[...,None],q),axis=2)
    ct = controls.swapaxes(1,2)
    g = ct@controls
    reg = .002*g.diagonal(axis1=1,axis2=2).clip(.001)[:,None,:]*np.eye(4)
    z = (out*ok[...,None,None]).reshape(len(p),D,6)
    return (z-controls@np.linalg.solve(g+reg,ct@z)).reshape(out.shape)


@lru_cache(maxsize=40)
def calibration(day,unit):
    raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
    p = read_p(day)
    slope = strip.daily_scale(p,y)[1:].clip(.2,4.)
    g = np.zeros((3,2,2))
    n = 0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(day%5,len(m),5)
        z = features(p[a+t],ok[t],filters(day),np.asarray(unit))
        target = y[a+t,:,1:]/slope-p[a+t,:,1:]
        g[...,0] += (z*z).sum((0,1))
        g[...,1] += (z*target[:,:,None,:]).sum((0,1))
        n += len(z)*D
    return g/n


def train(end):
    unit = tuple(np.median([units(d) for d in range(4,end)],axis=0))
    ds = np.stack([calibration(d,unit) for d in range(4,end)])
    g = ds.mean(0)
    xy = g[...,1]
    trust = xy**2/(xy**2+ds[...,1].var(0)/len(ds)+1e-12)
    slope = (xy/(g[...,0]+.00002)).clip(-3,3)*trust
    return np.r_[unit,filters(end).ravel(),slope.ravel()]


def signals(raw,grid,p,bank):
    w = bank[3:-6].reshape(2,3,D,D)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        out[a:b] = features(p[a:b],ok,w,bank[:3])*bank[-6:].reshape(3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
