"""Separate predictable price changes from new price information before encoding the stock's history."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import y3_consensus_references as history

parent=history.incumbent
scale=parent.parent
path=scale.path
trial=parent.trial
OUT=trial.clock.ART/'y3_expected_price_paths_v1'
ARMS=('original path','innovations only','expected moves only','extra response to surprise',
      'extra response to expectation','unrelated stock expectation','two-path combination','half surprise',
      'shared surprise direction')


def teacher_features(raw,t):
    cols=[0,7,11,12,13,14,15,16,17,18,21,25,26,34,36,38,40,41,42,50,51,54,55,64,66,69,74,77,78]
    x=np.tanh(.8*np.arcsinh(raw[:,:,cols].astype('float32')))
    history=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    now=x[t]
    def average(width):
        end=t+1
        start=(end-width).clip(0)
        return ((history[end]-history[start])/(end-start)[:,None,None]).astype('float32')
    fast,slow=average(8),average(40)
    side=now[:,:,[17,19,21,27]]-now[:,:,[18,20,22,28]]
    data=np.concatenate((now,now-fast,fast-slow,side*np.abs(now[:,:,:1]),
                         raw[t,:,83:].astype('float32')-1),2)
    return data


@lru_cache(maxsize=1)
def teacher():
    grams,ys=[],[]
    for d in range(4):
        raw=trial.clock.book.rawday(d)
        t=np.arange(45+d%5,len(raw)-1,5)
        z=teacher_features(raw,t)
        y=trial.clock.center(raw[t+1,:,0].astype('float32')).clip(-5,5)
        z=trial.clock.center(z).reshape(len(t)*50,-1)
        grams.append(z.T@z/len(z));ys.append(z.T@y.ravel()/len(z))
    return trial.curve.solve(np.mean(grams,0),np.mean(ys,0),.35).astype('float32')


def predicted(raw,w):
    out=np.zeros(raw.shape[:2],dtype='float32')
    if len(raw)>1:
        z=teacher_features(raw,np.arange(len(raw)-1))
        out[1:]=trial.clock.center(z)@w
    return out


def make_price(series):
    value=np.tanh(.65*np.arcsinh(series.astype('float32')+.00136))
    return np.vstack((np.zeros_like(value[:1],dtype=float),value.cumsum(0,dtype=float)))


@torch.inference_mode()
def encode(raw,m,surprise_only=False):
    net=scale.MagnitudePath(44).cuda().eval()
    net.load_state_dict({k[5:]:torch.from_numpy(v) for k,v in m.items() if k.startswith('fine/')})
    phi=torch.from_numpy(scale.base.dense.PHI[:,[4,8]].copy()).cuda()
    stream=path.stream(raw)
    exp=predicted(raw,m['teacher'])
    actual=raw[:,:,0].astype('float32')-.00136
    surprise=actual-exp
    prior=np.roll(exp,17,axis=1)
    paths=(exp-surprise,) if surprise_only else (surprise,exp,exp-surprise,surprise-exp,actual-prior)
    series=[make_price(s) for s in paths]
    rows=[]
    for t in np.array_split(np.arange(len(raw)),(len(raw)+79)//80):
        x=torch.from_numpy(path.patches(stream,t)).cuda()
        def read(patch):
            a,va=net(patch,phi)
            b,vb=net(path.reverse(patch,True),phi)
            return .5*(a*va.exp()-b*vb.exp())
        original=read(x)
        effects=[]
        for prices in series:
            patch=x.clone()
            end=t[:,None]+1-8*np.arange(7,-1,-1)
            start=(end-8).clip(0);end=end.clip(0)
            avg=(prices[end]-prices[start])/(end-start).clip(1)[:,:,None]
            patch[:,:,:,0]=torch.from_numpy(avg.transpose(0,2,1).astype('float16')).cuda()
            effects.append(read(patch))
        if surprise_only:
            vals=original.new_zeros(len(x),50,14)
            vals[:,:,0:2]=original
            vals[:,:,6:8]=.5*(original-effects[0])
            rows.append((vals-vals.mean(1,keepdim=True)).cpu().numpy())
            continue
        inn,ex,rev_inn,rev_ex,wrong=effects
        unique=.5*(original-rev_inn)
        common=torch.where(inn*original>0,inn.sign()*torch.minimum(inn.abs(),original.abs()),0)
        vals=torch.cat((original,inn,ex,unique,.5*(original-rev_ex),wrong,common),2)
        rows.append((vals-vals.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(rows)


@lru_cache(maxsize=4)
def weights(end):
    if end==16:
        with np.load(scale.OUT/'model.npz') as f:
            weights={k:f[k] for k in f.files if k.startswith('fine/')}
    else:
        weights={k:v for k,v in scale.train(end).items() if k.startswith('fine/')}
    weights['teacher']=teacher()
    return weights


def historical_state(raw,d):
    return encode(raw,weights(4*(d//4)))


record_with_state=parent.record_with_state
changes_with_reference=parent.changes_with_reference


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        rows.append(record_with_state(historical_state(raw,d),path.early.load(d,raw),y,d))
    return rows


def fit(rows):
    gs,bs=[np.stack(v).astype(float) for v in zip(*rows)]
    g=gs.mean(0)
    def read(cols,n):
        basis=np.eye(len(g))[:,cols]
        basis[:n]=-trial.curve.solve(g[:n,:n],g[:n,cols],.045)
        daily=bs@basis
        mu=daily.mean(0)
        mu*=abs(mu)/(abs(mu)+daily.std(0)/np.sqrt(len(rows))+1e-7)
        return basis@trial.curve.solve(basis.T@g@basis,mu,.95)
    surprise=read([9,10],5)
    bank=[read([3,4],3),read([5,6],3),read([7,8],3),surprise,read([11,12],5),read([13,14],3),
          read([5,6,7,8],3),.5*surprise,read([15,16],3)]
    model=dict(w=(.4*np.column_stack(bank)).astype('float32'))
    if len(rows)==12:
        model.update(weights(16))
    return model


def select(f,d):
    good=[]
    for j in (1,2,3,4,6,7,8):
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and
                (d[:,0,j]>0).sum()>=5 and f[:,:,1,j].mean(0).min()>0 and
                (v-f[:,:,0,5]).mean(0).min()>0):
            good.append(j)
    return max(good,key=lambda j:f[:,0,0,j].mean()) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        m=dict(f)
    pick=int(m['pick'])
    required=[0,1,2,3,4,9,10]
    surprise_only=not np.any(np.delete(m['w'][:,pick],required))
    return lambda raw,p:changes_with_reference(encode(raw,m,surprise_only),m,p)[pick]


def layer():
    predict,update=parent.layer(),correction()
    def apply(raw,p):
        q=predict(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    reports=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        t=np.arange(45+d%7,len(raw)-1,7)
        pred=trial.clock.center(teacher_features(raw,t))@teacher()
        y=trial.clock.center(raw[t+1,:,0].astype('float32'))
        reports.append(dict(day=d+1,ic=float(np.corrcoef(pred.ravel(),y.ravel())[0,1]),
                            r2=float(1-np.mean((pred-y)**2)/np.mean(y**2))))
    (OUT/'teacher_quality.json').write_text(json.dumps(reports,indent=2))
    history.run(sys.modules[__name__])
