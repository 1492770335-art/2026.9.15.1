"""Estimate a live Y3 correction's marginal utility from payments to its already-issued forecasts."""
import sys
from pathlib import Path

import numpy as np
import torch
import experiment_dense_risk_weighted_paid_leads_v1 as risk

strip,rel,onset,REFERENCE,read_p=risk.strip,risk.rel,risk.onset,risk.REFERENCE,risk.read_p
OUT=REFERENCE.parent/'dense_paid_correction_utility_v1'
ARMS=('long_run_utility','recent_utility','early_and_distal_confirmation')
train=risk.train


def tapes(raw,grid,p,bank):
    k,c,bad=strip.physical()
    signal=risk.signals(raw,grid,p,bank)[:,:,2,1]
    for a,b in rel.fields.parent.sessions(grid):
        rx=raw[a:b]
        ok=~np.all(rx==bad,axis=2)
        inc=np.log1p(k*(rx[...,0].astype(float)-.00136))*c[2]/k
        path=(inc*ok).cumsum(0)
        yield a,b,signal[a:b],path,np.vstack((np.zeros((1,50),int),(~ok).cumsum(0)))


def payment(t,z,pos,gap,beta):
    # A newly paid five-tick block can score twelve earlier predictions. Group them
    # by the beginning and end of the forecast horizon before estimating uncertainty.
    change=pos[t]-pos[t-5]
    rows=[]
    for offsets in (range(5,21,5),range(25,61,5)):
        value=np.zeros(len(t))
        for lag in offsets:
            at=t-lag
            valid=gap[t+1]==gap[at]
            w=z[at]*valid
            mean=w.sum(1)/valid.sum(1).clip(1)
            w=(w-mean[:,None])*valid
            # A Taylor expansion of pooled Pearson supplies the quadratic risk term.
            value+=(w*change-.5*beta*w*w/12).sum(1)/valid.sum(1).clip(1)
        rows.append(value)
    return np.stack(rows,axis=-1)


def confidence(avg,ss,mass):
    mean=avg/max(mass,1)
    err=np.sqrt((ss/max(mass,1)-mean**2).clip(1e-12)/max(mass/4,1))
    return mean/err


def gates(z,path,gap,beta):
    weights=np.zeros((len(z),3))
    av=np.zeros((2,3));ss=np.zeros((2,3));m=np.zeros(2)
    for i in range(0,len(z),20):
        t=np.arange(max(60,i-20),i,5)
        if len(t):
            x=payment(t,z,path,gap,beta)
            x=np.column_stack((x,x.sum(1)))
            rates=np.exp(-5*len(t)/np.array([260.,950.]))
            av=av*rates[:,None]+x.sum(0)
            ss=ss*rates[:,None]+(x*x).sum(0)
            m=m*rates+len(t)
        fast,slow=[confidence(av[k],ss[k],m[k]) for k in range(2)]
        # A neutral prior assigns half strength before this day's performance is observed.
        weights[i:i+20]=(.5+.5*np.tanh(slow[2]),.5+.5*np.tanh(fast[2]),
                         .5+.5*np.tanh(.5*(slow[2]+min(fast[1],slow[1]))))
    return weights


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,z,path,gap in tapes(raw,grid,p,bank):
        out[a:b,:,:,1]=z[...,None]*gates(z,path,gap,bank[-1])[:,None,:]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


def audit():
    import json
    OUT.mkdir(exist_ok=True)
    records={};windows={}
    metric=rel.ref.market.metric
    gain=rel.ref.market.gain
    for d in range(6,16):
        if d in (6,8,12):
            bank=train(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
        p=read_p(d)
        terms=[];base=[];gate=[];phase=[]
        for a,b,z,path,gap in tapes(raw,grid,p,bank):
            terms.append(z)
            phase.append(np.minimum(np.arange(b-a)//600,3))
            gate.append(gates(z,path,gap,bank[-1]))
        z=np.concatenate(terms);g=np.concatenate(gate);part=np.concatenate(phase)
        for mask in (part==0,part==1,part==2,part==3,g[:,0]<=.5,g[:,0]>.5):
            base.append(metric(p[...,2],y[...,2],z*mask[:,None]))
        records[d+1]=base
    for d in (9,13):
        g=np.stack([records[i] for i in range(d,d+4)]).sum(0)
        windows[d]=[float(gain(v)) for v in g]
    report=dict(order=['first_600','next_600','next_600','session_tail','weak_past_utility','positive_past_utility'],windows=windows)
    (OUT/'timing.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
