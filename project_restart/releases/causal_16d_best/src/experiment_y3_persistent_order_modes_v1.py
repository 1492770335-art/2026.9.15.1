"""Learn past order modes whose correspondence to future order and price paths repeats on disjoint dates."""
from functools import lru_cache
import json
import sys
import numpy as np
import torch
from scipy.linalg import solve_triangular

import y3_order_price_rollout_references as history

parent=history.incumbent
process=parent.process
study,trial,book=parent.study,parent.trial,parent.book
OUT=trial.clock.ART/'y3_persistent_order_modes_v1'
ARMS=('unrestricted direct interval regression','replicated price modes','replicated order modes','joint order and price modes',
      'modes common to both early and late orders','single-fold pooled modes','one split with reassigned future stocks',
      'persistent orders and stock-local fit','persistent response after local fit','modes of last twenty steps')
CANDIDATES=(1,2,3,4,7,8,9)
CONTROLS=(0,5,6)
WIDTH=500


@lru_cache(maxsize=1)
def calibration():
    init=[process.moments(d) for d in range(4)]
    return dict(**process.routing.queues.calibration(),mu=np.mean([x[0] for x in init],0),sc=np.mean([x[1] for x in init],0))


def inputs(raw,t,st):
    z=(process.state(raw,st)-st['mu'])/st['sc']
    x=process.lagged(z,t+1)
    return x,z


@lru_cache(maxsize=16)
def moments(day):
    raw=trial.clock.book.rawday(day)
    t=np.r_[np.arange(80+day%4,2260,4),np.arange(2460+day%4,len(raw)-60,4)]
    x,z=inputs(raw,t,calibration())
    cs=np.vstack((np.zeros(100),z.cumsum(0)))
    target=np.stack([(cs[t+hi+1]-cs[t+lo])/np.sqrt(hi-lo+1) for lo,hi in process.BANDS],1)
    # Preserve relative stock-specific dynamics at each future horizon.
    target=target.reshape(len(t),4,2,50)
    target-=target.mean(-1,keepdims=True)
    y=target.reshape(len(t),-1)
    x=x-x.mean(0)
    return x.T@x/len(x),x.T@y/len(x)


def projection(left,right,dim=16):
    cross=left@right.T
    # Positive eigenvalues identify directions that carry the same signed association on both splits.
    val,vec=np.linalg.eigh((cross+cross.T)/2)
    vectors=vec[:, -dim:][:,val[-dim:]>0]
    return vectors@vectors.T


@lru_cache(maxsize=7)
def params(end):
    daily=[moments(d) for d in range(end)]
    gram,cross=[np.stack(z) for z in zip(*daily)]
    g=gram.mean(0)
    mean_target=cross.mean(0).reshape(WIDTH,4,2,50)
    ch=np.linalg.cholesky(g+np.diag(.9*g.diagonal().clip(.02)+.001))
    # Whiten the lagged predictors before comparing cross-covariances across dates.
    values=[solve_triangular(ch,cross[j::2].mean(0),lower=True) for j in (0,1)]
    a,b=[v.reshape(WIDTH,4,2,50) for v in values]
    def packed(v):
        return v.reshape(WIDTH,-1)
    price=(a[:,:,1]+b[:,:,1])/2
    def for_target(left,right):
        return projection(packed(left),packed(right))@packed(price)
    direct=packed(price)
    order=for_target(a[:,:,0],b[:,:,0])
    price_only=for_target(a[:,:,1],b[:,:,1])
    joint=for_target(a,b)
    long_order=for_target(a[:,:2,0],b[:,2:,0])
    pooled=(a+b)/2
    simple=for_target(pooled,pooled)
    wrong=for_target(a,np.roll(b,17,axis=3))
    tail=for_target(a[:,3],b[:,3])
    def unwhiten(w):
        return solve_triangular(ch.T,w,lower=False).reshape(WIDTH,4,50)
    co=[unwhiten(v) for v in (direct,price_only,order,joint,long_order,simple,wrong,tail)]
    local=np.zeros_like(co[0])
    for stock in range(50):
        source=stock+50*np.arange(10)
        y=mean_target[source,:,1,stock]
        gram_local=g[np.ix_(source,source)]
        local[source,:,stock]=np.linalg.solve(gram_local+np.diag(.18*gram_local.diagonal().clip(.02)+.001),y)
    co.append(local)
    return dict(**calibration(),weights=np.stack(co).astype('float32'))


def forecast(raw,t,st):
    x,_=inputs(raw,t,st)
    w=st['weights']
    v=(x.astype('float32')@w.transpose(1,0,2,3).reshape(WIDTH,-1)).reshape(len(t),len(w),4,50)
    return v.transpose(1,0,3,2)*st['sc'][None,None,50:,None]


def design_with(raw,p,t,st):
    direct,price,order,joint,long_order,pooled,wrong,tail,local=forecast(raw,t,st)
    cols=(direct,price,order,joint,long_order,pooled,wrong,local+order,order-local,tail)
    return trial.clock.center(np.concatenate((p[t],*cols),2)).astype('float32')


def historical_design(d):
    st=params(max(4,d-d%2))
    return lambda raw,p,t:design_with(raw,p,t,st)


def fit(rows):
    groups=[(np.arange(3+4*i,7+4*i),3,.35) for i in range(len(ARMS))]
    return dict(**book.prev.gram(rows,groups),**params(len(rows)+4))


changes=book.changes


def correction():
    with np.load(OUT/'model.npz') as f:st=dict(f)
    def update(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),st),st)[int(st['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


def audit():
    daily=[]
    for d in range(8,16):
        st=params(d-d%2)
        raw=trial.clock.book.rawday(d)
        t=np.r_[np.arange(80,2260,4),np.arange(2460,len(raw)-60,4)]
        guesses=forecast(raw,t,st)
        ret=process.state(raw,st)[:,50:]
        cs=np.vstack((np.zeros(50),ret.cumsum(0)))
        day=[]
        for j,(lo,hi) in enumerate(process.BANDS):
            y=trial.clock.center(cs[t+hi+1]-cs[t+lo]).ravel()
            day.append([float(np.corrcoef(trial.clock.center(v[:,:,j]).ravel(),y)[0,1]) for v in guesses])
        daily.append(day)
    report=dict(days=[9,16],windows=process.BANDS,models=['direct','price','order','joint','early-late','pooled','reassigned','tail','local'],
                mean=np.mean(daily,0).tolist(),daily=daily)
    (OUT/'future_mode_skill.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('order mode skill',np.round(report['mean'],4),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
    audit()
