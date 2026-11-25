"""Forecast relative occupancy of anonymized discrete states across the middle and final twenty ticks."""
import sys

import numpy as np
import torch

import experiment_y3_order_response_filter_v1 as filters
import experiment_y3_paired_gross_flow_v1 as gross
import y3_terminal_validation as trial

OUT=trial.clock.ART/'y3_forward_category_occupancy_v1'
ARMS=['direct final return','via middle occupancy','via final occupancy','mid-to-final update',
      'stock-reassigned update','matched minus reassigned','middle and final updates']
statistics=trial.statistics
SECONDARY='late-relative and final-twenty relative correlation'


def state(raw):
    cats=raw[:,:,83:].astype('int8')
    level=np.concatenate((cats==0,cats==2),axis=2).astype('float32')
    p,paired,_=gross.components(raw)
    base=np.concatenate((p,paired,level),axis=2)
    fast=filters.smooth(base,8)
    slow=filters.smooth(base,64)
    features=trial.clock.center(np.concatenate((base,fast,slow,fast-slow),axis=2))
    # Preserve raw state occupancy for training labels. Inference consumes only the prefix features.
    return features,level


def observe(s,y,d):
    features,level=s
    t=np.arange(d%7,len(y)-60,7)
    prefix=np.concatenate((np.zeros_like(level[:1]),level.cumsum(0)),axis=0)
    mid=trial.clock.center((prefix[t+41]-prefix[t+21])/20)
    end=trial.clock.center((prefix[t+61]-prefix[t+41])/20)
    now=trial.clock.center(level[t])
    c=trial.curve.settled.historical_geometry()[1]
    ret=trial.clock.center(np.stack((y[t+20,:,1],y[t+40,:,1]),axis=2))*(c[2,1]/c[1,1])
    flat=lambda z:z.reshape(-1,z.shape[-1])
    x=flat(features[t]);r=flat(ret.clip(-3,3))
    # Decompose future state levels and changes. Exchange receivers to check whether stock identity matters.
    future=flat(np.concatenate((mid,end,end-mid,np.roll(end-mid,17,axis=1),mid-now),axis=2))
    m,e=flat(mid),flat(end)
    return [v.astype(float)/len(x) for v in (x.T@x,x.T@future,x.T@r,m.T@m,e.T@e,m.T@r,e.T@r)]


def fit(hist):
    g,b,price,gm,ge,bm,be=[np.stack(v) for v in zip(*hist)]
    def solve(gram,cov,reg):
        mean=cov.mean(0)
        err=cov.std(0)/np.sqrt(len(hist))
        return trial.curve.solve(gram.mean(0),mean*abs(mean)/(abs(mean)+err+1e-5),reg)
    middle,final,update,wrong,early=np.hsplit(solve(g,b,.60),5)
    wm=solve(gm,bm,.7);we=solve(ge,be,.7)
    direct=solve(g,price,.8)[:,1]
    # Construct mediator and payoff maps on disjoint dates to avoid crediting correlated estimation noise.
    guarded=[];control=[]
    for k in (0,1):
        pred=trial.curve.solve(g[k::2].mean(0),b[k::2].mean(0),.60)
        _,_,change,placebo,_=np.hsplit(pred,5)
        price=trial.curve.solve(ge[1-k::2].mean(0),be[1-k::2].mean(0),.7)[:,1]
        guarded.append(change@price);control.append(placebo@price)
    best=np.mean(guarded,0);bad=np.mean(control,0)
    w=np.column_stack((direct,middle@wm[:,1],final@we[:,1],best,bad,best-bad,.5*(early@wm[:,0]+best)))
    return dict(w=w.astype('float32')),dict(transfer=float(np.linalg.norm(best)))


def changes(z,m):
    scores=.25*trial.clock.center(z[0]@m['w'])
    return list(np.moveaxis(scores,2,0))


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    def ok(a,b=None):
        x,h=f[:,:,:,a:a+1],d[:,:,a:a+1]
        if b is not None:x,h=x-f[:,:,:,b:b+1],h-d[:,:,b:b+1]
        return trial.scoring.select(x,h) is not None
    ids=[]
    if ok(2) and ok(2,0) and ok(2,1):ids.append(2)
    if ok(3,4):ids.extend(a for a in (3,5,6) if ok(a))
    return int(max(ids,key=lambda k:f[:,0,0,k].mean())) if ids else None


def correction():return trial.prior.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
