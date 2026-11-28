"""Estimate horizon-specific common error persistence from cross-spectra of disjoint stock baskets."""
from functools import lru_cache
import sys

import numpy as np
import torch
from scipy.signal.windows import dpss

from experiment_dense_market_correction_rollforward_v1 import REFERENCE,read_p,event,onset,rel,strip
from experiment_dense_disjoint_settled_error_transfer_v1 import units


OUT=REFERENCE.parent/'dense_multitaper_common_surprises_v1'
ARMS=('covariance_beyond_short_taper','common_low_frequency_excess','replicated_slow_band_response')
BASKETS=np.random.default_rng(37891).permutation(50).reshape(5,10)
N=96
TAPERS=dpss(N,2.5,Kmax=4)
FREQ=np.fft.rfftfreq(N)
LAGS=np.arange(1,12)
COS=np.cos(2*np.pi*FREQ[:,None]*LAGS)
MULT=np.r_[1.,np.full(N//2-1,2.),1.]


def spectral(x):
    x=x-x.mean(0)
    return np.fft.rfft(TAPERS[:,:,None]*x[None],axis=1)


def spectra(err,good):
    z=err*good
    market=z.sum(1,keepdims=True)/good.sum(1,keepdims=True).clip(1)
    mean=z[:,BASKETS].sum(2)/good[:,BASKETS].sum(2).clip(1)
    panel=spectral(mean)
    local=spectral((err-market)*good)
    pairs=[(panel[:,:,i]*panel[:,:,j].conj()).real.mean(0) for i in range(5) for j in range(i)]
    # Use signed cross-power. Averaging two disjoint groups prevents an
    # independent fluctuation of a single stock from inflating common power.
    return np.stack(pairs)*MULT, (abs(local)**2).mean((0,2))*MULT


def ratio(s,w):
    return np.sum(s*w,axis=-1)/(np.sum(s,axis=-1).clip(.002))


def features(cross,within,h):
    common=cross.mean(0)
    taper=(1-LAGS/h).clip(0)
    full=1+2*COS@taper
    short=1+2*COS@(taper*(1-LAGS/7).clip(0))
    long=(ratio(common,full-short)-ratio(within,full-short))/h
    # Compare low and middle bands using frequency cutoffs scaled to the horizon.
    low=(FREQ>0)&(FREQ<=.5/h)
    mid=(FREQ>.5/h)&(FREQ<=1.5/h)
    contrast=low.astype(float)-mid*(low.sum()/mid.sum())
    balance=ratio(common,contrast)-ratio(within,contrast)
    pair_effect=ratio(cross,full)/h-ratio(within,full)/h
    conservative=np.clip(np.median(pair_effect)-.5*np.median(abs(pair_effect-np.median(pair_effect))),-.8,.8)
    return np.array((long,balance,conservative))


def blocks(move,ok,p,unit):
    path=move.cumsum(0)
    gaps=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t=np.arange(5,len(p),5)
    good=gaps[t+1]==gaps[t-5]
    error=path[t]-path[t-5]-p[t-5,:,0]*unit
    for lo in range(5*(N+1),len(p),120):
        hi=t.searchsorted(lo)
        sl=slice(hi-N,hi)
        c,s=spectra(error[sl],good[sl])
        state=np.stack([features(c,s,h) for h in (4,12)])
        yield lo,min(lo+120,len(p)),state


@lru_cache(maxsize=48)
def record(day,unit):
    p=read_p(day)
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    states=[v for a,b,m,ok in event.sessions(raw,grid)
            for _,_,v in blocks(m,ok,p[a:b],unit)]
    return np.median(states,axis=0)


def train(end):
    unit=float(np.median([units(d)[0] for d in range(4,end)]))
    baseline=np.median([record(d,unit) for d in range(4,end)],axis=0)
    return np.r_[unit,baseline.ravel()]


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    center=bank[1:].reshape(2,3)
    for a,b,m,ok in event.sessions(raw,grid):
        market=p[a:b,:,1:].mean(1,dtype=float)
        for lo,hi,state in blocks(m,ok,p[a:b],bank[0]):
            alpha=.28*np.tanh((state-center)*[3.,2.,2.])
            out[a+lo:a+hi]=market[lo:hi,None,None,:]*alpha.T
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
