"""Causal order-flow and price states for the fifty-stock distributed-lag model."""
from functools import lru_cache

import numpy as np

import y3_peer_depth_routing as routing
import y3_peer_terminal_news_references as history

trial,study,book=history.trial,history.study,routing.book
LAGS=((1,1),(2,5),(6,19),(20,20),(21,40))
BANDS=((1,5),(6,20),(21,40),(41,60))


def state(raw,st):
    side=(raw[:,:,54]>st['floors'][1]).astype(float)-(raw[:,:,55]>st['floors'][2])
    k=float(st['k'])
    ret=np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136))))/k
    return np.column_stack((side,np.arcsinh(ret)))


def lagged(z,t):
    cs=np.vstack((np.zeros(z.shape[1]),z.cumsum(0)))
    return np.column_stack([(cs[(t-lo+1).clip(0)]-cs[(t-hi).clip(0)])/(hi-lo+1) for lo,hi in LAGS])


@lru_cache(maxsize=16)
def moments(d):
    st=routing.queues.calibration()
    z=state(trial.clock.book.rawday(d),st)
    mean=z.mean(0)
    scale=z.std(0).clip(.03)
    z=(z-mean)/scale
    t=np.r_[np.arange(80+d%2,2300,2),np.arange(2460+d%2,len(z),2)]
    x=lagged(z,t)
    return mean,scale,x.T@x/len(x),x.T@z[t]/len(x)


@lru_cache(maxsize=7)
def fitted(end):
    mean,scale,g,b=[np.mean(v,0) for v in zip(*(moments(d) for d in range(end)))]
    penalty=.65*g.diagonal().clip(.025)+.0003
    a=np.linalg.solve(g+np.diag(penalty),b)
    return dict(**routing.queues.calibration(),ar=a.astype("float32"),mu=mean,sc=scale)
