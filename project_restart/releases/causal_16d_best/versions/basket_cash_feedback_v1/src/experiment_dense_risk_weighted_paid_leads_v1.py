"""Weight settled five-tick payoffs by risk at issuance and allow the inferred slope to vary across stocks."""
import sys
import numpy as np
import torch
import prepare_dense_full_lead_cash_reference_v1 as best

full=best.model
strip,rel,onset,REFERENCE,read_p=full.strip,full.rel,full.onset,best.OUT,best.read_p
OUT=REFERENCE.parent/'dense_risk_weighted_paid_leads_v1'
ARMS=('risk_balanced_response','risk_specific_response','supported_specific_response')
train=full.train


def exposures(raw):
    k,c,bad=strip.physical()
    valid=~np.all(raw==bad,axis=-1)
    r=np.log1p(k*(raw[...,0].astype(float)-.00136))*c[2]/k
    total=np.vstack((np.zeros(50),(r*r*valid).cumsum(0)))
    t=np.arange(len(raw))+1
    earlier=(t-180).clip(0)
    vol=np.sqrt((total[t]-total[earlier]+.03)/(t-earlier+10)[:,None])
    ratio=(vol/np.median(vol,axis=1,keepdims=True).clip(.00001)).clip(.3,3)
    high=ratio**3/(1+ratio**3)
    # Both the precision weight and the membership must be fixed at issuance.
    w=1/ratio**2
    return np.stack((w*(1-high),w*high),axis=-1),high


def evidence(last,x,price,gaps,w):
    z=np.zeros((len(last),2,2,2))
    cash=price[last]-price[last-5]
    for lag in range(5,61,5):
        at=last-lag
        mask=gaps[last+1]==gaps[at]
        weights=w[at]*mask[...,None]
        den=weights.sum(1).clip(1)
        v=x[at,:,1]
        means=(v[...,None]*weights).sum(1)/den
        v=v[...,None]-means[:,None]
        paid=(v*cash[...,None]*weights).sum(1)/den
        risk=(v*v*weights).sum(1)/den
        j=int(lag>20)
        z[:,:,j,0]+=paid
        z[:,:,j,1]+=risk/(8 if j else 4)
    return z


def signals(raw,grid,p,bank):
    co=bank[:-2].reshape(96,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,x,price,gaps in full.live.sessions(raw,grid,p,co):
        w,high=exposures(raw[a:b])
        first=np.zeros((2,2,2));second=np.zeros((2,2,3));mass=0.
        for i in range(0,len(x),20):
            t=np.arange(max(65,i-20),i,5)
            if len(t):
                s=evidence(t,x,price,gaps,w)
                f,v=s[...,0],s[...,1]
                decay=np.exp(-len(t)*5/520.)
                first=decay*first+s.sum(0)
                second=decay*second+np.stack((f*f,f*v,v*v),-1).sum(0)
                mass=decay*mass+len(t)
            mix=high[i:i+20]
            raw_prior=np.zeros(2)
            common=full.slopes(first.mean(0),second.mean(0),mass,raw_prior)
            group=full.slopes(first,second,mass,raw_prior)
            local=(1-mix)*group[0]+mix*group[1]
            supported=np.where(local*common>0,np.sign(local)*np.minimum(abs(local),abs(common)),0.)
            origin=x[i:i+20,:,1]/bank[-1]
            pred=strip.center(np.stack((origin*common,origin*local,origin*supported),axis=-1))
            # Receiver-dependent scaling can reintroduce exposure to today's three forecasts.
            q=strip.center(p[a+i:a+i+len(pred)].astype(float))
            qt=q.transpose(0,2,1)
            g=qt@q
            reg=.015*np.eye(3)*np.diagonal(g,axis1=1,axis2=2)[:,None,:]+.00001*np.eye(3)
            out[a+i:a+i+len(pred),:,:,1]=pred-q@np.linalg.solve(g+reg,qt@pred)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
