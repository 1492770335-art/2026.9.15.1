"""Discount a near-hedged payoff-timing signal using independent tapers and disjoint stock panels."""
import sys
import numpy as np
import torch

import experiment_dense_near_neutral_timing_update_v1 as hedge
import experiment_dense_forecast_payoff_timing_v1 as timing
import experiment_dense_multitaper_common_surprises_v1 as spec
from experiment_dense_spectral_forecast_conditioning_v1 import REFERENCE,read_p,event,onset,rel,strip,spectrum


OUT=REFERENCE.parent/'dense_replicated_timing_hedge_v1'
ARMS=('taper_uncertainty','separate_stock_confirmation','replicated_near_hedge')
train=hedge.train


def per_taper(f,shared):
    n=f.shape[2]
    mat=np.einsum('kfsc,kfsd->kfcd',f.conj(),f,optimize=True)
    if shared:
        total=f.sum(2)
        mat=(total.conj()[...,None]*total[...,None,:]-mat)/(n-1)
    return mat*(spec.MULT/n)[None,:,None,None]


def arrow(common,local):
    return np.array([timing.prices(common,j)[0][1]-timing.prices(local,j)[0][1] for j in (1,2)])


def trust_stream(m,ok,p,unit,center):
    p=p.astype(float)
    path=m.cumsum(0)
    gap=np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    t=np.arange(5,len(p),5)
    good=gap[t+1]==gap[t-5]
    obs=np.concatenate((p[t]*unit,(path[t]-path[t-5])[...,None]),axis=2)
    panels=[]
    for ids in (spec.BASKETS,spec.BASKETS[:,::2],spec.BASKETS[:,1::2]):
        live=good[:,ids]
        x=obs[:,ids]
        cm=(x*live[...,None]).sum((1,2))/live.sum((1,2)).clip(1)[:,None]
        relative=((x-cm[:,None,None])*live[...,None]).reshape(len(t),-1,4)
        baskets=(x*live[...,None]).sum(2)/live.sum(2).clip(1)[...,None]
        panels.append((baskets,relative))
    for lo in range(5*(spec.N+1),len(p),120):
        end=t.searchsorted(lo)
        ix=slice(end-spec.N,end)
        all_views=[]
        for n,(groups,relvals) in enumerate(panels):
            a=per_taper(spectrum(groups[ix]),True)
            b=per_taper(spectrum(relvals[ix]),False)
            if n==0:
                z=arrow(a.mean(0),b.mean(0))-center
                jack=np.stack([arrow((a.sum(0)-a[k])/3,(b.sum(0)-b[k])/3) for k in range(4)])
                var=.75*((jack-jack.mean(0))**2).sum(0)
            else:
                all_views.append(arrow(a.mean(0),b.mean(0))-center)
        left,right=all_views
        taper=z**2/(z**2+var+.0001)
        match=(left*z>0)&(right*z>0)
        agreement=np.minimum(abs(left),abs(right))/(abs(z)+.04)
        stock=match*agreement.clip(0,1)
        yield lo,min(lo+120,len(p)),np.stack((taper,stock,np.sqrt(taper*stock)))


def signals(raw,grid,p,bank):
    params=bank[:11]
    prior=bank[11:].reshape(2,4,4)
    center=params[3:9].reshape(2,3)[:,1]
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        q=p[a:b]
        z=hedge.raw_signals(m,ok,q,params)
        avg=q.mean(1,dtype=float)
        gs=prior.copy()
        for lo,hi,trust in trust_stream(m,ok,q,params[:3],center):
            for j,h in enumerate((5,20)):
                t=np.arange(max(485,lo-120-h),lo-h,5)
                if len(t):
                    x=hedge.observations(m,ok,q,z,params,t,j)
                    if len(x): gs[j]=.82*gs[j]+.18*(x.T@x/len(x))
                controls=np.column_stack((z[lo:hi,j],np.zeros(hi-lo),avg[lo:hi,j+1]))
                delta=z[lo:hi,j,0]+controls@hedge.compensated(gs[j],0)
                out[a+lo:a+hi,:,:,j]=(delta[:,None]*trust[:,j])[:,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
