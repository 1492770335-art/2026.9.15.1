"""Estimate observable timing corrections to common forecasts with unity-gain causal phase filters."""
from functools import lru_cache
import sys

import numpy as np
import torch
from scipy.signal import lfilter

import experiment_dense_price_update_selection_v1 as source

strip,event,onset,rel=source.strip,source.event,source.onset,source.rel
REFERENCE,read_p=source.REFERENCE,source.read_p
OUT=REFERENCE.parent/'dense_common_phase_correction_v1'
ARMS=('forecast_phase','terminal_price_phase','replicated_basket_phase')
TAU=(2.,8.,32.)
WIDTH=9


def delay(x,tau):
    # Stable first-order all-pass filter. Initialize at the first input to avoid opening transients.
    a=(1.-tau)/(1.+tau)
    return lfilter([a,1.],[1.,a],x-x[:1],axis=0)+x[:1]

def design(p,m,units):
    panels=p[:,source.GROUP].mean(2,dtype=float)
    path=m[:,source.GROUP].mean(2).cumsum(0)
    terminal=panels+path[:,:,None]/units
    groups=[[],[],[]]
    for tau in TAU:
        phase=(delay(panels,tau)-panels)
        # A delayed terminal estimate refers to an earlier maturity. Extrapolate its
        # expiry using the far slope observable in the current forecast curve.
        near=panels[...,0]*units[0]/5
        middle=(panels[...,1]*units[1]-panels[...,0]*units[0])/15
        far=(panels[...,2]*units[2]-panels[...,1]*units[1])/40
        carry=np.stack((near,middle,far),axis=2)/units
        level=delay(terminal,tau)-terminal+tau*carry
        left,right=phase[:,0],phase[:,1]
        agree=np.where(left*right>0,np.sign(left)*np.minimum(abs(left),abs(right)),0.)
        groups[0].append(phase.mean(1))
        groups[1].append(level.mean(1))
        groups[2].append(agree)
    return np.stack([np.concatenate(parts,axis=1) for parts in groups],axis=1)

@lru_cache(maxsize=12)
def unit(d):
    p=read_p(d)
    y=onset.settled.load(d,'y')
    _,co,_=strip.physical()
    return strip.daily_scale(p,y).clip(.2,4)*co[2]/co

@lru_cache(maxsize=48)
def record(d,scale):
    raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
    p=read_p(d)
    q=p[...,1:].astype(float);v=y[...,1:].astype(float)
    q-=q.mean((0,1));v-=v.mean((0,1))
    slope=((q*v).mean((0,1))/(q*q).mean((0,1))).clip(.2,4)
    r=(v/slope-q).mean(1)
    xs=[];ys=[];ps=[]
    for a,b,m,_ in event.sessions(raw,grid):
        t=np.arange(32+d%4,b-a-60,4)
        xs.append(design(p[a:b],m,np.array(scale))[t])
        ys.append(r[a+t]);ps.append(p[a+t].mean(1))
    x,y,p=map(np.concatenate,(xs,ys,ps))
    # Project correlations attributable merely to a different market amplitude during fitting.
    z=np.stack([np.column_stack((p,x[:,j],y)) for j in range(3)])
    z-=z.mean(1,keepdims=True)
    return z.transpose(0,2,1)@z/z.shape[1]

def train(end):
    scale=tuple(np.median(np.stack([unit(d) for d in range(4,end)]),axis=0))
    grams=np.stack([record(d,scale) for d in range(4,end)])
    weights=[]
    for days in grams.transpose(1,0,2,3):
        g=days.mean(0)
        proj=np.linalg.solve(g[:3,:3]+.02*np.diag(g.diagonal()[:3].clip(1e-8)),g[:3,3:-2])
        rhs=days[:,3:-2,-2:]-proj.T@days[:,:3,-2:]
        h=g[3:-2,3:-2]
        scale_x=np.sqrt(h.diagonal().clip(1e-10))
        eig,rot=np.linalg.eigh(h/np.outer(scale_x,scale_x))
        axes=rot/(scale_x[:,None]*np.sqrt(eig.clip(0)+.7))
        r=axes.T@rhs
        avg=r.mean(0)
        stable=avg**3/(avg**2+3*r.var(0)/len(r)+1e-10)
        weights.append(axes@stable)
    return np.r_[scale,np.stack(weights).ravel()]

def signals(raw,grid,p,bank):
    w=bank[3:].reshape(3,WIDTH,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,_ in event.sessions(raw,grid):
        x=design(p[a:b],m,bank[:3])
        out[a:b]=np.einsum('tjk,jkh->tjh',x,w)[:,None]
    return out

def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)

if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
