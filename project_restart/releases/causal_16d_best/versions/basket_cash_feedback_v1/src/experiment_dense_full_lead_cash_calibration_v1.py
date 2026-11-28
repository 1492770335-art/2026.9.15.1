"""Calibrate the full horizon using all elapsed five-tick leads of each newly observed payment."""
import sys
import numpy as np
import torch
import experiment_dense_live_transition_cash_readout_v1 as live

strip,rel,onset,REFERENCE,read_p=live.strip,live.rel,live.onset,live.REFERENCE,live.read_p
OUT=REFERENCE.parent/'dense_full_lead_cash_calibration_v1'
ARMS=('all_paid_leads','structural_cash_prior','independent_basket_support')
train=live.train
PART=np.random.default_rng(17624).permutation(50).reshape(2,25)


def evidence(last,x,price,gap):
    stats=np.zeros((len(last),2,2,2,2))
    ret=price[last]-price[last-5]
    for h,(start,stop) in enumerate(((5,20),(20,60))):
        for lag in range(5,stop+1,5):
            at=last-lag
            valid=gap[last+1]==gap[at]
            for j,ids in enumerate(PART):
                mask=valid[:,ids]
                den=mask.sum(1,keepdims=True).clip(1)
                raw=x[at[:,None],ids,h]
                z=(raw-(raw*mask).sum(1,keepdims=True)/den)*mask
                # z is centered on the same available names used to calculate the covariance.
                grp=int(lag>start)
                stats[:,j,h,grp,0]+=(z*ret[:,ids]).sum(1)/den[:,0]
                stats[:,j,h,grp,1]+=(z*z).sum(1)/den[:,0]/((stop-start if grp else start)/5)
    return stats


def slopes(first,second,mass,prior):
    cov=first[...,0]/max(mass,1)
    var=first[...,1]/max(mass,1)
    deviation=cov-prior*var
    # Estimate uncertainty for the departure from the prior including stochastic exposure size.
    v=second[...,0]+prior**2*second[...,2]-2*prior*second[...,1]
    noise=(v/max(1,mass)-deviation**2).clip(0)/max(1,mass/7)
    cred=deviation**2/(deviation**2+noise+1e-12)
    return (prior+deviation*cred/(var+.000006)).sum(-1).clip(-4,4)


def signals(raw,grid,p,bank):
    co=bank[:-2].reshape(96,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,x,price,gaps in live.sessions(raw,grid,p,co):
        first=np.zeros((2,2,2,2));second=np.zeros((2,2,2,3))
        mass=0.
        for i in range(0,len(x),20):
            t=np.arange(max(65,i-20),i,5)
            if len(t):
                stats=evidence(t,x,price,gaps)
                cov,var=stats[...,0],stats[...,1]
                decay=np.exp(-len(t)*5/520.)
                first=decay*first+stats.sum(0)
                second=decay*second+np.stack((cov*cov,cov*var,var*var),-1).sum(0)
                mass=decay*mass+len(t)
            pooled_a,pooled_b=first.mean(0),second.mean(0)
            unconstrained=slopes(pooled_a,pooled_b,mass,np.zeros(2))
            prior=np.array([0.,1.])
            shrunk=slopes(pooled_a,pooled_b,mass,prior)
            s=slopes(first,second,mass,prior)
            shared=np.where(s[0]*s[1]>0,np.sign(s[0])*np.minimum(abs(s[0]),abs(s[1])),0.)
            z=x[i:i+20]/bank[-2:]
            out[a+i:a+i+len(z)]=z[:,:,None]*np.stack((unconstrained,shrunk,shared))[None,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
