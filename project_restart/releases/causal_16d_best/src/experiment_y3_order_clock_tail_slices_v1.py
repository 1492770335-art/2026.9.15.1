"""Forecast small, moderate and tail subreturns separately from the two ordered event clocks."""
from functools import lru_cache
import sys
import numpy as np
import torch
import experiment_y3_independent_order_clocks_v1 as clocks
import joint_order_clock_references as history

parent,study,trial=history.incumbent,history.study,history.trial
OUT=trial.clock.ART/'y3_order_clock_tail_slices_v1'
ARMS=('raw clipped returns','risk-normalized sum','small moves','moderate moves','large moves','all size slices',
      'constant volatility','wrong-stock volatility','later large moves only','probability of a tail event',
      'tail direction and size','tail return without clipping')
CANDIDATES=(1,2,3,4,5,8,9,10,11)
CONTROLS=(0,6,7)


@lru_cache(maxsize=1)
def prior():
    return np.mean([np.mean((trial.clock.book.rawday(d)[:,:,0].astype(float)-.00136)**2,0) for d in range(4)],0)


def risk(raw,t,st):
    r=raw[:,:,0].astype(float)-.00136
    sq=np.concatenate((np.zeros((1,50)),(r*r).cumsum(0)))
    end=t+1
    begin=np.maximum(end-140,np.where(t>=2401,2401,0))
    relative=np.sqrt((sq[end]-sq[begin]+40*st)/((end-begin+40)[:,None]))
    return np.sqrt(.07+20*.272**2*relative**2).clip(.25,3.).astype('float32')


@lru_cache(maxsize=16)
def record(d):
    raw=trial.clock.book.rawday(d)
    t=np.arange(90+d%7,len(raw)-60,7)
    t=t[(t+60<2401)|(t>=2401)]
    x=clocks.design_bank(raw,t).reshape(len(t)*50,-1)
    ys=trial.clock.labels(d).reshape(-1,50,3)
    y=trial.clock.center(np.stack([ys[t+h,:,1] for h in (0,20,40)],2))
    v=y/risk(raw,t,prior())[:,:,None]
    small=v.clip(-.6,.6)
    mid=v.clip(-1.8,1.8)-small
    tail=v.clip(-6,6)-mid-small
    parts=np.stack((y.clip(-4,4),small,mid,tail,np.sign(v)*(abs(v)>1.8),v.clip(-12,12)),2)
    z=trial.clock.center(parts).reshape(len(x),-1)
    return (x.T@x/len(x)).astype(float),(x.T@z/len(x)).astype(float)


@lru_cache(maxsize=7)
def params(end):
    gs,bs=[np.stack(v) for v in zip(*[record(d) for d in range(end)])]
    g=gs.mean(0)
    b=bs.mean(0)
    b*=abs(b)/(abs(b)+bs.std(0)/np.sqrt(end)+1e-6)
    return np.linalg.solve(g+.9*np.diag(g.diagonal().clip(.00001)),b).astype('float32')


def signals(raw,t,w,st):
    x=(clocks.design_bank(raw,t)@w).reshape(len(t),50,6,3)
    scale=risk(raw,t,st)[:,:,None]
    original=x[:,:,0]
    s,m,l=[x[:,:,i]*scale for i in (1,2,3)]
    total=s+m+l
    propensity=x[:,:,4]*scale
    unbounded=x[:,:,5]*scale
    fixed=x[:,:,1:4].sum(2)*np.sqrt(.07+20*.272**2*st)[None,:,None]
    wrong=x[:,:,1:4].sum(2)*np.roll(scale,17,axis=1)
    return trial.clock.center(np.concatenate((original,total,s,m,l,fixed,wrong,propensity,unbounded),2).astype('float32'))


def historical_design(d):
    cache={}
    w=params(max(4,d-d%2))
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=signals(raw,t,w,prior())
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    s=[np.arange(3+i*3,6+i*3) for i in range(9)]
    groups=[*s[:5],np.concatenate(s[2:5]),s[5],s[6],s[4][1:],s[7],np.r_[s[7],s[4]],s[8]]
    return dict(**clocks.linear.gram(rows,[(cols,3,.35) for cols in groups]),bank=params(len(rows)+4),prior=prior())


changes=clocks.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st=dict(f)
    def apply(raw,p):
        x=np.concatenate((trial.clock.center(p),signals(raw,np.arange(len(raw)),st['bank'],st['prior'])),2)
        return changes(x,st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
