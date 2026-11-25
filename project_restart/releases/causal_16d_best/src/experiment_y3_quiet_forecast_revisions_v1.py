"""Split forecast revisions into revisions at a stock's price changes and revisions between them."""
import sys
import numpy as np
import torch

import y3_monotone_references as parent

screen,trial=parent.screen,parent.trial
OUT=trial.clock.ART/'y3_quiet_forecast_revisions_v1'
ARMS=('fixed-clock revision','event-clock revision','price-change revision','quiet revision','event and quiet',
      'reassigned event times','same-length fixed window','unresolved quiet revision','half combined')
CANDIDATES=(1,2,3,4,7,8)
CONTROLS=(0,5,6)


def boundaries(events,count):
    time=np.arange(len(events))
    result=np.empty(events.shape,dtype=np.int32)
    for stock in range(events.shape[1]):
        positions=time[events[:,stock]]
        idx=np.searchsorted(positions,time,side='right')-count
        result[:,stock]=np.where(idx>=0,positions[idx.clip(0)] if len(positions) else 0,0)
    return result


def features(raw,p):
    coef=trial.curve.settled.historical_geometry()[1][:,1]
    near=trial.clock.center(p[:,:,1])*coef[2]/coef[1]
    tail=trial.clock.center(p[:,:,2])-near
    signals=np.stack((near,tail),axis=2)
    change=np.diff(signals,axis=0,prepend=signals[:1])
    event=raw[:,:,0]!=np.float16(.00136)
    wrong=np.roll(event,17,axis=1)
    r=raw[:,:,0].astype('float32')-.00136
    def prefix(z):return np.concatenate((np.zeros_like(z[:1],dtype=float),z.cumsum(0,dtype=float)))
    total=prefix(change);at=prefix(change*event[:,:,None]);other=prefix(change*wrong[:,:,None]);price=prefix(r)
    end=np.arange(len(p))[:,None]+1;stock=np.arange(50)[None,:]
    groups=[[] for _ in range(7)]
    for count,width in ((3,9),(12,36)):
        start=boundaries(event,count)
        false=boundaries(wrong,count)
        all_moves=total[end,stock]-total[start,stock]
        fixed=total[end,stock]-total[(end-width).clip(0),stock]
        hit=at[end,stock]-at[start,stock]
        quiet=all_moves-hit
        placebo=(total[end,stock]-total[false,stock])-(other[end,stock]-other[false,stock])
        paid=price[end,stock]-price[start,stock]
        typical=(end-np.median(end-start,axis=1,keepdims=True).astype(int)).clip(0)
        terms=(fixed,all_moves,hit,quiet,placebo,total[end,stock]-total[typical,stock],quiet*np.tanh(paid[:,:,None]*signals*2))
        for group,v in zip(groups,terms):group.append(v)
    return trial.clock.center(np.concatenate([np.concatenate(v,2) for v in groups],2)).astype('float32')


def fit(rows):
    fixed,whole,hit,quiet,wrong,typical,unresolved=[np.arange(3+4*j,7+4*j) for j in range(7)]
    joint=np.r_[hit,quiet]
    return screen.projected_readouts(rows,[(fixed,3,.4),(whole,7,.4),(hit,7,.4),(quiet,7,.4),
                (joint,7,.4),(wrong,7,.4),(typical,3,.4),(unresolved,7,.4),(joint,7,.2)])


def correction():return screen.correction(sys.modules[__name__])
def layer():return screen.layer(sys.modules[__name__],parent.incumbent)

if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__],parent.references,parent.outer)
