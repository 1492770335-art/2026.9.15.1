"""Calibrate a forecast of symmetric five-tick rewards from market-relative cash that has already settled."""
import sys
import numpy as np
import torch
import experiment_dense_frozen_transition_cash_span_v1 as frozen

strip,rel,onset,REFERENCE,read_p=frozen.strip,frozen.rel,frozen.onset,frozen.REFERENCE,frozen.read_p
OUT=REFERENCE.parent/'dense_live_transition_cash_readout_v1'
ARMS=('within_frame_cash_value','settled_return_value','settled_and_structural_agreement')


def train(end):
    g=np.mean([np.load(frozen.OUT/f'gram_{d:02d}.npy') for d in range(end)],axis=0)
    # Learn one future symmetric cash stream per horizon independently of the incumbent.
    co=frozen.ridge(g[3:frozen.W,3:frozen.W],g[3:frozen.W][:,[frozen.W+2,frozen.W+6]],.5)
    issued=np.mean([np.load(frozen.OUT/f'gram_{d:02d}.npy') for d in range(4,end)],axis=0)
    beta=issued[[1,2],[-2,-1]]/issued.diagonal()[1:3]
    _,units,_=strip.physical()
    return np.r_[co.ravel(),beta*units[2]/units[1:]]


def sessions(raw,grid,p,co):
    k,c,bad=strip.physical()
    target=frozen.hidden(raw,np.arange(len(p)))@co
    for a,b in rel.fields.parent.sessions(grid):
        q=strip.center(p[a:b].astype(float))
        s=strip.center(target[a:b])
        xt=q.transpose(0,2,1)
        cov=xt@q
        reg=.002*(np.eye(3)*np.diagonal(cov,axis1=1,axis2=2)[:,None,:])+.00001*np.eye(3)
        x=s-q@np.linalg.solve(cov+reg,xt@s)
        ok=~np.all(raw[a:b]==bad,axis=2)
        ret=np.log1p(k*(raw[a:b,:,0].astype(float)-.00136))*c[2]/k
        price=(ok*ret).cumsum(0)
        gaps=np.vstack((np.zeros((1,50),int),(~ok).cumsum(0)))
        yield a,b,x,price,gaps


def signals(raw,grid,p,bank):
    co=bank[:-2].reshape(96,2)
    units=bank[-2:]
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,x,price,gaps in sessions(raw,grid,p,co):
        avg=np.zeros((2,2));square=np.zeros((2,2));risk=np.full(2,.0001)
        mass=0.
        for i in range(0,len(x),20):
            # Each update uses just the five-tick cash blocks paid since the previous update.
            # All forecasts paired with one block predate the beginning of its return.
            last=np.arange(max(65,i-20),i,5)
            if len(last):
                obs=[]
                for h,(start,stop) in enumerate(((5,20),(20,60))):
                    z=[];v=[]
                    for lag in range(start+5,stop+1,5):
                        origin=last-lag
                        valid=gaps[last+1]==gaps[origin]
                        ret=price[last]-price[last-5]
                        ret=ret-(ret*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)
                        q=x[origin,:,h]*valid
                        z.append((q*ret).sum(1)/valid.sum(1).clip(1))
                        v.append((q*q).sum(1)/valid.sum(1).clip(1))
                    # Short leads provide a separate signal of whether the same issued value
                    # also improves the early part of the total-return target.
                    earlier=last-5
                    ok=gaps[last+1]==gaps[earlier]
                    ret=price[last]-price[last-5]
                    ret=ret-(ret*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1)
                    near=(x[earlier,:,h]*ret*ok).sum(1)/ok.sum(1).clip(1)
                    obs.append(np.stack((np.sum(z,axis=0),near*start/5,np.mean(v,axis=0)),axis=1))
                r=np.stack(obs,axis=1)
                decay=np.exp(-len(last)*5/520)
                avg=decay*avg+r[:,:,:2].sum(0)
                square=decay*square+(r[:,:,:2]**2).sum(0)
                risk=decay*risk+r[:,:,2].sum(0)
                mass=decay*mass+len(last)
            # Dependence between neighboring blocks reduces their effective sample count.
            m=avg/max(mass,1)
            noise=(square/max(mass,1)-m*m).clip(0)/max(mass/6,1)
            reliable=m*m/(m*m+noise+1e-12)
            slopes=((avg*reliable)/(risk[:,None]+.001)).clip(-4,4)
            settled=(slopes[:,0]+slopes[:,1]).clip(-4,4)
            gate=np.where(settled>0,np.minimum(1.,settled),0.)
            z=x[i:i+20]/units
            out[a+i:a+i+len(z)]=z[:,:,None]*np.stack((np.ones(2),settled,gate))[None,None]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
