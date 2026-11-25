"""Decompose future book covariance into information arriving by tick ten and the remaining uncertainty."""
from functools import lru_cache
import sys

import numpy as np
import torch

import y3_interval_book_training as screen

parent,trial,stack,book,history = screen.parent,screen.trial,screen.stack,screen.book,screen.history
OUT = trial.clock.ART/'y3_interval_covariance_bridge_v1'
SCREEN_STRIDE = 7
ARMS = ('unchanged','total conditional covariance','midpoint-resolved covariance','unresolved covariance',
        'different payoff for each interval','resolved late risk','resolved and unresolved late risk',
        'mixed midpoint forecast errors','reassigned midpoint observations','midpoint versus residual covariance',
        'posterior mean correction','near unresolved and later resolved covariance')
GROUPS = ((),(0,),(1,),(2,),(3,4,5),(4,5),(1,2),(6,),(7,),(8,),(9,),(10,4,5))
CANDIDATES = (2,3,4,5,6,7,9,10,11)
CONTROLS = (1,8)


@lru_cache(maxsize=16)
def sample(day):
    raw = trial.clock.book.rawday(day)
    y = trial.clock.labels(day).reshape(-1,50,3)
    t = np.arange(97+day%13,len(y)-60,13)
    t = t[(t+60<2401)|(t>=2498)]
    x,f,_ = book.observations(raw,t,True)
    mid = book.observations(raw,t+10)[0][:,:,1:49]
    ratio = book.model.observed.historical_geometry()[1][:,1]
    ret = trial.clock.center(np.stack([y[t+h,:,1]*ratio[2]/ratio[1] for h in (0,20,40)],axis=-1))
    # The midpoint view exists only for fitting the teacher. The deployed encoder sees the origin alone.
    x = x.reshape(len(t)*50,-1)
    m = mid.reshape(len(x),-1).copy()
    f = f.reshape(len(x),80)
    joint = np.concatenate((x,m),axis=1).astype(float)
    return (x.astype('float16'),m.astype('float16'),f[:,:16].copy(),joint.T@joint/len(x),
            joint.T@f[:,:16]/len(x),f.astype(float).T@f/len(x),f.astype(float).T@ret.reshape(len(x),3).clip(-5,5)/len(x))


def covariance_terms(x,m,z,prior,posterior,pay):
    mu = x@prior
    step = x@posterior[:len(prior)]+m@posterior[len(prior):]
    news = step-mu
    error = z-step
    def price(a,b):
        return np.einsum('ni,ijk,nj->nk',a[:,:8],pay,b[:,8:],optimize=True)
    resolved = price(news,news)
    residual = price(error,error)
    mixed = price(news,error)+price(error,news)
    np.testing.assert_allclose(resolved+residual+mixed,price(z-mu,z-mu),atol=4.e-6,rtol=2.e-6)
    return resolved,residual,mixed,price(mu,news)+price(news,mu)


def params(end):
    rows = [sample(d) for d in range(end)]
    def mean(part,k):
        return np.mean([a[k] for a in part],axis=0)
    def fit(part):
        gram,c = mean(part,3),mean(part,4)
        n = rows[0][0].shape[1]
        return book.ridge(gram[:n,:n],c[:n]),book.ridge(gram,c)
    pay = book.ridge(mean(rows,5),mean(rows,6))[16:].reshape(8,8,3)
    # All terms from a given date share a teacher estimated using different dates.
    halves = [fit(rows[1::2]),fit(rows[::2])]
    scores = []
    for day,(x,m,z,*_) in enumerate(rows):
        x,m = x.astype(float),m.astype(float)
        prior,posterior = halves[day%2]
        resolved,residual,mixed,rev = covariance_terms(x,m,z,prior,posterior,pay)
        sham = covariance_terms(x,np.roll(m.reshape(-1,50,m.shape[1]),17,axis=1).reshape(m.shape),z,prior,posterior,pay)[0]
        scores.append(x.T@np.column_stack((resolved,residual,mixed,rev,sham))/len(x))
    n = rows[0][0].shape[1]
    read = book.ridge(mean(rows,3)[:n,:n],np.mean(scores,axis=0))
    return dict(read=read.astype('float32'))


def advice(raw,t,st):
    x = book.observations(raw,t)[0]
    resolved,resid,mixed,revision,wrong = np.split(x@st['read'],5,axis=2)
    tail = lambda a:a[:,:,1:].sum(2)
    parts = (tail(resolved+resid+mixed),tail(resolved),tail(resid),*np.moveaxis(resolved,2,0),
             tail(mixed),tail(wrong),tail(resolved-resid),tail(revision),resid[:,:,0])
    return trial.clock.center(np.stack(parts,axis=2))


def correction():
    return screen.make_correction(sys.modules[__name__])


def replay_with():
    return screen.replay_for(sys.modules[__name__])


def layer():
    return history.study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    mod = sys.modules[__name__]
    if sys.argv[1:]==['replay'] or screen.train(mod):
        book.model.replay.replay(mod)
