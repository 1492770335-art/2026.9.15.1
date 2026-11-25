"""Predict pairwise victory probability and conditional winning margin for stocks with similar baseline Y3."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps/tree_models'))
import lightgbm as lgb
import experiment_y3_paired_gross_flow_v1 as book
import experiment_y3_order_response_filter_v1 as filt
import experiment_y3_score_neighborhood_heads_v1 as local
import y3_disjoint_validation as trial
import y3_neighborhood_validation as check

OUT=trial.clock.ART/'y3_pair_win_margin_v1'
ARMS=['probability only','probability times typical margin','conditional signed margin','direct pair payoff','asymmetric margin only','half signed margin']
# A connected, regular graph in each 10-stock bin gives every member four comparisons.
EDGES=np.array([(i,(i+k)%10) for i in range(10) for k in (1,3)])
LEFT,RIGHT=EDGES.T


def state(raw):
    p,l,m=book.components(raw)
    price=np.arcsinh(raw[:,:,:1].astype('float32'))
    src=np.concatenate((p,l,m,price),axis=2)
    fast=filt.smooth(src,8);slow=filt.smooth(src,64)
    move=fast-slow
    broad=np.arcsinh(raw[:,:,[4,13,18,25,66,69,74]].astype('float32'))
    return np.concatenate((fast,move,broad-filt.smooth(broad,64)),axis=2).astype('float16')


def comparison(z,p):
    rank=check.order(p[:,:,2])
    pos=rank.reshape(len(z),5,10)
    i,j=pos[:,:,LEFT].reshape(len(z),-1),pos[:,:,RIGHT].reshape(len(z),-1)
    return i,j,*pair_features(z,p,i,j)


def pair_features(z,p,i,j):
    t=np.arange(len(z))[:,None]
    x=z[t,i].astype('float32');q=z[t,j].astype('float32')
    # The neighborhood is defined by Y3. Shorter forecasts help compare the two stocks' horizon profiles.
    near=p[:,:,:2]
    a=np.concatenate((x-q,near[t,i]-near[t,j]),2)
    b=np.concatenate((.5*(x[:,:,:8]+q[:,:,:8]),abs(near[t,i]-near[t,j])),2)
    return a,b


def record(z,p,y,d):
    t=np.arange(d%23,len(z)-60,23)
    i,j,a,b=comparison(z[t],p[t])
    rows=np.arange(len(t))[:,None]
    v=y[t,:,2]
    # Learn the remaining pair payoff after the existing score difference. Keep margin units across folds.
    pred=p[t,:,2].astype(float).ravel()
    ref=v.ravel().astype(float)
    k=np.cov(pred,ref,bias=True)[0,1]/max(pred.var(),1e-8)
    residual=v-k*p[t,:,2]
    outcome=residual[rows,i]-residual[rows,j]
    return a.reshape(-1,a.shape[-1]).astype('float16'),b.reshape(-1,b.shape[-1]).astype('float16'),outcome.ravel()


def merge_records(records):
    return [np.concatenate(items) for items in zip(*records)]


def fit(history):
    a,b,realized=[np.concatenate(x) for x in zip(*history)]
    m=len(realized)
    # Mirror a deterministic half of the examples; both orientations represent the same statistical problem.
    orient=np.random.default_rng(5729).choice(np.array([-1,1],dtype='float32'),size=m)
    x=np.column_stack((a*orient[:,None],b)).astype('float32')
    y=realized*orient
    def train(target,ids,objective):
        trainset=lgb.Dataset(x[ids],label=target[ids],params={'max_bin':31})
        model=lgb.train(dict(objective=objective,num_leaves=7,min_data_in_leaf=650,
            learning_rate=.05,lambda_l2=140,max_bin=31,feature_fraction=.75,
            verbosity=-1,num_threads=2,force_col_wise=True,seed=5201),trainset,num_boost_round=35)
        return np.array(model.model_to_string())
    return dict(sign=train((y>0).astype(float),np.arange(m),'binary'),
                size=train(abs(y).clip(0,3),np.flatnonzero(y>0),'regression'),
                direct=train(y.clip(-3,3),np.arange(m),'regression'),scale=np.float32(np.mean(abs(y).clip(0,3))))


@lru_cache(maxsize=12)
def booster(text):
    return lgb.Booster(model_str=text)


def changes_with_reference(z,m,p):
    models=[booster(m[k].item()) for k in ('sign','size','direct')]
    out=[]
    for t in range(0,len(z),128):
        pp=p[t:t+128];i,j,a,b=comparison(z[t:t+128],pp)
        shape=a.shape[:2]
        def forecasts(sign):
            x=np.concatenate((sign*a,b),axis=2)
            x=x.reshape(-1,x.shape[-1])
            return [tree.predict(x,num_threads=2).reshape(shape) for tree in models]
        plus,minus=forecasts(1),forecasts(-1)
        prob=.5*(plus[0]+1-minus[0])
        tilt=2*prob-1
        ordinal=m['scale']*tilt
        common=.5*(plus[1]+minus[1])*tilt
        asym=.5*(plus[1]-minus[1])
        direct=.5*(plus[2]-minus[2])
        v=np.stack((ordinal,common,common+asym,direct,asym,.5*(common+asym)),2)
        score=np.zeros((len(i),50,v.shape[2]))
        rows=np.arange(len(i))[:,None]
        np.add.at(score,(rows,i),v);np.add.at(score,(rows,j),-v)
        out.append(.20*local.residual((score/4).astype('float32'),pp[:,:,2],pp[:,:,2],5))
    return list(np.moveaxis(np.concatenate(out),2,0))


def select(f,d,pairs):
    eligible=[i for i in range(len(ARMS)) if check.qualifies(f,d,pairs,i)]
    return int(max(eligible,key=lambda i:f[:,0,0,i].mean())) if eligible else None


def correction():return trial.previous.prior.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    check.run(sys.modules[__name__])
