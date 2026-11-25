"""Estimate which receiving-stock portfolios preserve an early price response into later return windows."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch
from scipy.linalg import solve_triangular
import y3_persistent_order_references as history

model,study,trial=history.incumbent,history.study,history.trial
process,book=model.process,model.book
OUT=trial.clock.ART/'y3_receiver_price_portfolios_v1'
ARMS=('all destination stocks','near-response recipients','late-response recipients','both source and recipient',
      'near-to-late replicated recipients','early response at source only','permuted recipients','source period-order factors',
      'local plus near-to-late recipients','local plus uncompressed late forecast','local plus permuted recipients',
      'delayed response shared by order and price recipients')
CANDIDATES=(1,2,3,4,8,11)
CONTROLS=(0,5,6,9,10)


@lru_cache(maxsize=16)
def moments(d):
    st=model.calibration()
    raw=trial.clock.book.rawday(d)
    t=np.r_[np.arange(100+d%4,2260,4),np.arange(2470+d%4,len(raw)-60,4)]
    x,z=model.inputs(raw,t,st)
    k=float(st['k'])
    true_return=np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136))))/k
    z[:,50:]=true_return/st['sc'][50:]
    cs=np.vstack((np.zeros(100),z.cumsum(0)))
    target=np.stack([(cs[t+hi+1]-cs[t+lo])/np.sqrt(hi-lo+1) for lo,hi in process.BANDS],1)
    target=target.reshape(len(t),4,2,50)
    target-=target.mean(3,keepdims=True)
    x=x-x.mean(0)
    return x.T@x/len(x),x.T@target.reshape(len(t),-1)/len(x)


def projector(gram,n=8):
    eig,v=np.linalg.eigh((gram+gram.T)/2)
    keep=v[:,-n:][:,eig[-n:]>0]
    return keep@keep.T


@lru_cache(maxsize=7)
def params(end):
    grams,cross=[np.stack(v) for v in zip(*(moments(d) for d in range(end)))]
    g=grams.mean(0)
    lower=np.linalg.cholesky(g+np.diag(.9*g.diagonal().clip(.02)+.001))
    a,b=[solve_triangular(lower,cross[j::2].mean(0),lower=True).reshape(500,4,2,50) for j in (0,1)]
    # Receiver vectors have 50 coordinates regardless of the number of lagged source features.
    def stack(m,bands,field):
        return m[:,bands,field,:].reshape(-1,50)
    price=(a[:,:,1]+b[:,:,1])/2
    an,bn=stack(a,slice(0,2),1),stack(b,slice(0,2),1)
    af,bf=stack(a,slice(2,4),1),stack(b,slice(2,4),1)
    ao,bo=stack(a,slice(2,4),0),stack(b,slice(2,4),0)
    near=projector(an.T@bn)
    far=projector(af.T@bf)
    lasting=projector((an.T@bf+bn.T@af)/2)
    joint=projector((ao.T@bf+bo.T@af)/2)
    left=model.projection(a[:,:2,1].reshape(500,-1),b[:,:2,1].reshape(500,-1))
    both=np.einsum('ij,jhk->ihk',left,price)
    perm=np.random.default_rng(9422).permutation(50)
    wrong=lasting[np.ix_(perm,perm)]
    orders=model.projection(a[:,:,0].reshape(500,-1),b[:,:,0].reshape(500,-1))
    order_price=np.einsum('ij,jhk->ihk',orders,price)
    vals=(price,price@near,price@far,both@near,price@lasting,both,price@wrong,order_price,price@joint)
    weights=[solve_triangular(lower.T,v.reshape(500,-1),lower=False).reshape(500,4,50) for v in vals]
    local=np.zeros_like(price)
    target=cross.mean(0).reshape(500,4,2,50)
    for stock in range(50):
        ix=np.arange(10)*50+stock
        gg=g[np.ix_(ix,ix)]
        local[ix,:,stock]=np.linalg.solve(gg+np.diag(.18*gg.diagonal().clip(.02)+.001),target[ix,:,1,stock])
    return dict(**model.calibration(),weights=np.stack((*weights,local)).astype('float32'))


def design_with(raw,p,t,st):
    base,near,far,both,lasting,left,wrong,order,joint,local=model.forecast(raw,t,st)
    vals=(base,near,far,both,lasting,left,wrong,order,local+lasting,local+base,local+wrong,joint)
    return trial.clock.center(np.concatenate((p[t],*[v[:,:,2:] for v in vals]),2)).astype('float32')


def historical_design(d):
    st=params(max(4,d-d%2))
    return lambda raw,p,t:design_with(raw,p,t,st)


def fit(rows):
    return dict(**book.prev.gram(rows,[(np.arange(3+2*i,5+2*i),3,.35) for i in range(len(ARMS))]),**params(len(rows)+4))


changes=book.changes


def correction():
    with np.load(OUT/'model.npz') as f:st=dict(f)
    def update(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),st),st)[int(st['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


def audit():
    out=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d);st=params(d-d%2)
        t=np.r_[np.arange(100,2260,5),np.arange(2470,len(raw)-60,5)]
        vals=model.forecast(raw,t,st)
        k=float(st['k'])
        cs=np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136)))).cumsum(0)/k
        windows=[]
        for j,(lo,hi) in enumerate(process.BANDS):
            y=trial.clock.center(cs[t+hi]-cs[t+lo-1]).ravel()
            windows.append([float(np.corrcoef(trial.clock.center(v[:,:,j]).ravel(),y)[0,1]) for v in vals])
        out.append(windows)
    info=dict(days=[9,16],windows=process.BANDS,methods=[*ARMS[:8],ARMS[-1],'local'],daily=out,mean=np.mean(out,0).tolist())
    (OUT/'interval_skill.json').write_text(json.dumps(info,indent=2))
    print('receiver skill',np.round(info['mean'],4),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
    audit()
