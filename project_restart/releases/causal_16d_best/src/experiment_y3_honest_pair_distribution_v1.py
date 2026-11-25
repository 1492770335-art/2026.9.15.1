"""Cross-fit pairwise return-distribution trees, sharing splits across nested win thresholds."""
from functools import lru_cache
import pickle
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import torch

import experiment_y3_pair_win_margin_v1 as pairs
import experiment_y3_score_neighborhood_heads_v1 as local
import y3_early_score_neighborhoods as early
import y3_disjoint_validation as trial
import y3_neighborhood_validation as check

OUT=trial.clock.ART/'y3_honest_pair_distribution_v1'
ARMS=['win probability','bounded payoff integral','large payoff probability','full integral','signed leaf mean','half full integral','replicated integral']
state,record,merge_records=pairs.state,pairs.record,pairs.merge_records
CUT=np.array([0.,.3,.85,1.65])


def warm_history():
    history=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        history.append(record(state(raw),early.load(d,raw),trial.clock.labels(d).reshape(-1,50,3),d))
    return history


def samples(history):
    a,b,y=[np.concatenate(records) for records in zip(*history)]
    sign=np.random.default_rng(6093).choice(np.array([-1,1],dtype='float32'),len(y))
    y=y*sign
    return np.column_stack((a*sign[:,None],b)).astype('float32'),y


def fit(history):
    models=[];levels=[]
    for i in [0,1]:
        x,y=samples(history[i::2])
        target=(y[:,None]>CUT).astype('float32')
        forest=ExtraTreesRegressor(n_estimators=32,max_depth=6,min_samples_leaf=300,max_features=.7,
                                   n_jobs=2,random_state=701+i)
        forest.fit(x,target)
        x,y=samples(history[1-i::2])
        truth=np.column_stack((y[:,None]>CUT,y.clip(-3.5,3.5)))
        prior=truth.mean(0)
        posterior=[]
        for tree in forest.estimators_:
            leaf=tree.apply(x);n=tree.tree_.node_count
            counts=np.bincount(leaf,minlength=n)
            sums=np.column_stack([np.bincount(leaf,weights=col,minlength=n) for col in truth.T])
            posterior.append(((sums+180*prior)/(counts[:,None]+180)).astype('float32'))
        models.append((forest,posterior))
        levels.append(np.mean(abs(y).clip(0,3)))
    return dict(forest=np.frombuffer(pickle.dumps(models,protocol=4),dtype='uint8'),amplitude=np.float32(np.mean(levels)))


@lru_cache(maxsize=2)
def load_models(data):
    return pickle.loads(data)


def changes_with_reference(z,m,p):
    models=load_models(m['forest'].tobytes());parts=[]
    for t in range(0,len(z),128):
        pp=p[t:t+128];i,j,a,b=pairs.comparison(z[t:t+128],pp)
        def forecast(sign):
            x=np.concatenate((sign*a,b),2).reshape(-1,a.shape[2]+b.shape[2])
            return np.stack([np.mean([post[tree.apply(x)] for tree,post in zip(f.estimators_,probs)],axis=0)
                             for f,probs in models])
        diff=forecast(1)-forecast(-1)
        v=diff.mean(0)
        # Trapezoidal integral of the positive minus negative survival function over payout magnitude.
        quadrature=np.array([.15,.425,.675,.7],dtype='float32')
        integral=v[:,:4]@quadrature
        low=v[:,0]*m['amplitude']
        tail=v[:,2:4]@quadrature[2:]
        separately=diff[:,:,:4]@quadrature
        replicate=integral*np.maximum(0,np.sign(separately[0]*separately[1]))
        bank=np.column_stack((low,integral-tail,tail,integral,.5*v[:,4],.5*integral,replicate)).reshape(len(i),i.shape[1],-1)
        out=np.zeros((len(i),50,bank.shape[2]),dtype='float32');rows=np.arange(len(i))[:,None]
        np.add.at(out,(rows,i),bank);np.add.at(out,(rows,j),-bank)
        parts.append(.2*local.residual(out/4,pp[:,:,2],pp[:,:,2],5))
    return list(np.moveaxis(np.concatenate(parts),2,0))


def select(f,d,pairs):
    valid=[]
    for i in range(len(ARMS)):
        x=f[:,:,0,i]
        if ((x>0).sum(0).min()>=3 and x.mean(0).min()>0 and x[-1].min()>0
            and (d[:,0,i]>0).sum()>=5 and sum(r['gain'][i] for r in pairs)>0):
            valid.append(i)
    return int(max(valid,key=lambda i:f[:,0,0,i].mean())) if valid else None


def correction():return trial.previous.prior.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    check.run(sys.modules[__name__])
