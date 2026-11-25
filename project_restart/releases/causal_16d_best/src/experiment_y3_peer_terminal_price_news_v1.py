"""Transmit changes to peers' implied future prices at a common deadline when their spot quotes move."""
import sys
import numpy as np
import torch

import experiment_y3_same_deadline_event_curve_v1 as deadline
import y3_peer_depth_routing as routing
import y3_peer_depth_references as history

trial,study,book=history.trial,history.study,routing.book
OUT=trial.clock.ART/'y3_peer_terminal_price_news_v1'
ARMS=('realized peer move','peer near terminal revision','peer middle terminal revision',
      'peer far terminal revision','peer term-structure change','forecast and quote agree',
      'mismatched endpoint','old forecast update with current price','sender changed no quoted price',
      'news across all maturities')
CANDIDATES=(1,2,3,4,5,9)
CONTROLS=(0,6,7,8)


def params(end):
    return dict(**routing.params(end),coeff=trial.curve.settled.historical_geometry()[1])


def design_with(raw,p,t,st):
    k=float(st['k'])
    coeff=st['coeff']
    mid,half,_,_,valid,_,_,_=routing.queues.tape.prices(raw,st)
    prev=np.r_[0,np.arange(len(raw)-1)]
    active=(mid!=mid[prev])|(half!=half[prev])
    ok=valid&valid[prev]
    ok[0]=False
    spot=np.log1p(k*(raw[:,:,0].astype(float)-.00136))*coeff[2,1]/k
    levels=np.log1p(k*((p-coeff[:,0])/coeff[:,1]).clip(-.95/k,None))*coeff[2,1]/k
    now=np.arange(len(p))[:,None]
    def view(h):
        return deadline.curve(levels,now,h)
    def delta(h):
        return spot+view(h)-view(h+1)[prev]
    early,middle,far=[delta(h) for h in (5,20,40)]
    old=(np.arange(len(p))-10).clip(0)
    vals=(spot,early,middle,far,far-early,far*(far*spot>0),
          spot+view(40)-view(40)[prev],spot+(far-spot)[old],far)
    graph=routing.balanced(st['graphs'][1])
    cols=[]
    for i,s in enumerate(vals):
        v=np.where(ok&((~active) if i==8 else active),np.arcsinh(s),0)
        cs=np.vstack((np.zeros(50),v.cumsum(0)))
        cols.extend([routing.recent(cs,t,w)@graph for w in (5,20,60)])
    return trial.clock.center(np.concatenate((p[t],np.stack(cols,2)),2)).astype('float32')


def historical_design(day):
    st=params(max(4,day-day%2))
    return lambda raw,p,t:design_with(raw,p,t,st)


def fit(rows):
    groups=[np.arange(3+3*i,6+3*i) for i in range(9)]
    groups.append(np.r_[groups[1],groups[2],groups[3]])
    return dict(**book.prev.gram(rows,[(g,3,.35) for g in groups]),**params(len(rows)+4))


changes=book.changes


def correction():
    with np.load(OUT/'model.npz') as f:st=dict(f)
    def update(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),st),st)[int(st['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
