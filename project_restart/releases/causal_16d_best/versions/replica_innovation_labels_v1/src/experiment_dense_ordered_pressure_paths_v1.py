"""Learn distal relative-return signals from the order of cash, book pressure and forecast revisions."""
from functools import lru_cache
import sys
import numpy as np
import torch
from scipy.signal import lfilter

from experiment_dense_directional_market_persistence_v1 import REFERENCE, read_p, strip, event, onset, rel

OUT = REFERENCE.parent / 'dense_ordered_pressure_paths_v1'
ARMS = ('own_price_pressure_order','peer_price_pressure_order','forecast_pressure_order')


def center(x):
    return x-x.mean(1,keepdims=True)


def areas(left,right,tau):
    # Both inputs are contemporaneously observed increments. The strictly
    # trailing displacement makes the antisymmetric product depend on order.
    decay=np.exp(-1/tau)
    l=lfilter([0.,1.],[1.,-decay],left,axis=0)
    r=lfilter([0.,1.],[1.,-decay],right,axis=0)
    return lfilter([1.],[1.,-decay],l*right-r*left,axis=0)


def features(raw,move,ok,p):
    q=center(p.astype(float))
    mid=np.tanh(.65*np.arcsinh(raw[:,:,[26,38,41,42,54,55]].astype(float)))
    pressure=center(mid[:,:,::2]-mid[:,:,1::2])
    def change(x):
        increments=np.diff(x,axis=0,prepend=x[:1])
        observed=ok & np.vstack((ok[:1],ok[:-1]))
        return increments*observed[...,None]
    dp=change(pressure)
    price=center(move)[...,None]
    df=change(q)
    market=move.mean(1,keepdims=True)[...,None]
    local=np.concatenate((q,pressure),axis=2)
    banks=[[],[],[]]
    for tau in (8.,32.):
        banks[0].append(areas(price,dp,tau))
        banks[1].append(areas(market,dp,tau))
        banks[2].append(areas(df,dp,tau))
    return local,[center(np.concatenate(arm,axis=2)) for arm in banks]


def day_moments(day,designer):
    p=read_p(day)
    raw,grid,y=[onset.settled.load(day,k) for k in ('x','time','y')]
    return panel_moments(day,p,raw,grid,y,designer)


def panel_moments(day,p,raw,grid,y,designer):
    grams=np.zeros((3,15,15))
    n=0
    for a,b,m,ok in event.sessions(raw,grid):
        t=np.arange(32+day%4,b-a-60,4)
        if not len(t):
            continue
        ctrl,arms=designer(raw[a:b],m,ok,p[a:b])
        missing=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid=missing[t+61]==missing[t]
        labels=center(y[a+t].astype(float))
        for j,arm in enumerate(arms):
            obs=np.concatenate((ctrl[t],arm[t],labels),axis=2)[valid]
            grams[j]+=obs.T@obs
        n+=valid.sum()
    return grams/n


@lru_cache(maxsize=12)
def moments(day):
    return day_moments(day,features)


def train_weights(history):
    gs=np.stack(history)
    _,coef,_=strip.physical()
    weights=[]
    for j in range(3):
        g=gs[:,j].mean(0)
        sd=np.sqrt(g.diagonal()[:12].clip(1e-7))
        gram=g[:12,:12]/np.outer(sd,sd)
        v=np.eye(12)[:,6:]
        v[:6]=-np.linalg.solve(gram[:6,:6]+.1*np.eye(6),gram[:6,6:])
        v/=sd[:,None]
        pred_slope=(gs[:,j,(1,2),(13,14)]/gs[:,j,(1,2),(1,2)]).clip(.2,4.)
        target=gs[:,j,:12,13:]/pred_slope[:,None,:]-gs[:,j,:12,1:3]
        # Give the far component its own test of direction across training dates.
        grad=v.T@target
        factor=(coef[2]/coef[1])*pred_slope[:,0]/pred_slope[:,1]
        far=grad[:,:,1]-factor[:,None]*grad[:,:,0]
        mu=grad.mean(0)
        far_mu=far.mean(0)
        shrink=mu**2/(mu**2+2*grad.var(0)/len(gs)+1e-10)
        same=far_mu*mu[:,1]>0
        shrink[:,1]*=same*far_mu**2/(far_mu**2+far.var(0)/len(gs)+1e-10)
        gg=v.T@g[:12,:12]@v
        weights.append(v@np.linalg.solve(gg+1.1*np.diag(gg.diagonal().clip(.01)),mu*shrink))
    return np.stack(weights)


def train(end):
    return train_weights([moments(d) for d in range(4,end)])


def readout(raw,grid,p,bank,designer):
    out=np.empty((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        ctrl,arms=designer(raw[a:b],m,ok,p[a:b])
        for j,arm in enumerate(arms):
            out[a:b,:,j]=np.concatenate((ctrl,arm),axis=2)@bank[j]
    return out


def signals(raw,grid,p,bank):
    return readout(raw,grid,p,bank,features)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
