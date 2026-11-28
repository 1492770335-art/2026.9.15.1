"""Track revisions of a common terminal price before and after an observed twenty-tick extreme."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_excursion_state_mixture_v1 as parent

base,strip,event,onset,rel = parent.base,parent.strip,parent.event,parent.onset,parent.rel
REFERENCE,read_p = parent.REFERENCE,parent.read_p
OUT = REFERENCE.parent/'dense_peak_reforecast_order_v1'
ARMS = ('revision_at_peak','relative_deadline_revision','revision_price_loop')
C,D = 13,25


def features(m,ok,p):
    _,coef,_ = strip.physical()
    curve = strip.center(p.astype(float))*coef[2]/coef
    move = (m-(m*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1))*ok
    path = move.cumsum(0)
    vol = np.sqrt(parent.ref.ref.smooth(move**2,150)+.0004)
    gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    levels = np.arange(-20,1)
    def remaining_curve(age,horizon):
        rem = horizon-age
        # Linear interpolation for an old deadline using the three live forecast horizons and spot.
        anchors = [0,5,20,60]
        return np.stack([np.interp(rem,anchors,np.eye(4)[i]) for i in (1,2,3)],axis=1)
    short,long = [remaining_curve(levels+20,h) for h in (20,60)]
    out = np.zeros((*m.shape,D))
    out[:,:,:3] = p
    for start in range(20,len(m),128):
        t = np.arange(start,min(start+128,len(m)))
        ix = t[:,None]+levels
        prices = (path[ix]-path[ix[:,:1]]).transpose(0,2,1)
        curves = curve[ix]
        near,far = [prices+np.einsum('tksj,kj->tsk',curves,w) for w in (short,long)]
        origin = t-20
        side = np.where(curve[origin,:,1]>=0,1.,-1.)
        unit = vol[origin]*np.sqrt(20)
        positions = prices*side[...,None]/unit[...,None]
        near = (near-near[:,:,:1])*side[...,None]/unit[...,None]
        far = (far-far[:,:,:1])*side[...,None]/unit[...,None]
        peak = positions.argmax(2)
        trough = positions.argmin(2)
        def at(mat,idx):
            return np.take_along_axis(mat,idx[...,None],axis=2)[...,0]
        high,low = at(positions,peak),at(positions,trough)
        end = positions[:,:,-1]
        after = high-end
        rev = far[:,:,-1]
        pn,pf,tf = at(near,peak),at(far,peak),at(far,trough)
        pull = np.tanh(after)
        advanced = np.tanh(high)
        shift = np.tanh(at(far-near,peak))
        keep = (gap[t+1]==gap[origin])*side
        # Center the oriented area on its endpoint chord. Swapping the order of price and
        # forecast revisions reverses this area even when both total revisions are unchanged.
        def area(a,b):
            return np.tanh(.5*np.sum(a[:,:,:-1]*np.diff(b,axis=2)-b[:,:,:-1]*np.diff(a,axis=2),axis=2))
        terms = (end,high,low,after,rev,near[:,:,-1],rev*advanced,np.tanh(curve[origin,:,1]*side/unit),
                 np.tanh(curve[origin,:,2]*side/unit),(peak-trough)/20,
                 pf*pull,(rev-pf)*advanced,tf*(1-np.exp(-abs(low))),(rev-tf)*np.tanh(end-low),
                 shift*pull,(pf-pn)*advanced,(rev-pf-near[:,:,-1]+pn)*pull,shift*(peak-trough)/20,
                 area(positions,far),area(positions,near),area(near,far)*advanced,area(positions,far)*pull)
        out[t,:,3:] = np.tanh(np.stack(terms,axis=2))*keep[...,None]
    return strip.center(out)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    g,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(20+d%5,b-a-60,5)
        x = features(m,ok,p[a:b])[t]
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t-20]
        z = np.concatenate((x,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        g += rows.T@rows
        n += len(rows)
    return g/n


def train(end):
    days = np.stack([record(d) for d in range(4,end)])
    g = days.mean(0)
    unit = (days[:,[1,2],[-2,-1]]/days[:,[1,2],[1,2]]).clip(.2,4)
    err = days[:,:D,-2:]/unit[:,None]-days[:,:D,1:3]
    co = []
    for j in range(3):
        ix = np.arange(C+4*j,C+4*j+4)
        ctrl = np.arange(C)
        v = np.eye(D)[:,ix]
        v[ctrl] = -base.solve(g[np.ix_(ctrl,ctrl)],g[np.ix_(ctrl,ix)],.03)
        day = v.T@err
        mu = day.mean(0)
        st = mu**3/(mu**2+2*day.var(0)/len(days)+1e-12)
        co.append(v@base.solve(v.T@g[:D,:D]@v,st,.8))
    return np.stack(co,axis=1)


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        out[a:b] = (features(m,ok,p[a:b])@bank.reshape(D,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
