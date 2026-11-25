"""Forecast last-interval categorical occupancy and price it on separate trading dates."""
import sys

import numpy as np
import torch

import experiment_y3_forward_category_occupancy_v1 as occupancy
import y3_terminal_validation as trial

OUT=trial.clock.ART/'y3_crossfitted_future_states_v1'
ARMS=['pooled final states','independent-date final states','future states priced for middle',
      'middle states priced for end','reassigned receiver','independent vs reassigned',
      'independent horizon-specific transfer']
statistics=trial.statistics
SECONDARY=occupancy.SECONDARY
state=occupancy.state


def observe(s,y,d):
    rows=occupancy.observe(s,y,d)
    x,level=s
    t=np.arange(d%7,len(y)-60,7)
    prefix=np.concatenate((np.zeros_like(level[:1]),level.cumsum(0)),axis=0)
    final=trial.clock.center((prefix[t+61]-prefix[t+41])/20)
    wrong=np.roll(final,17,axis=1)
    design=x[t].reshape(-1,x.shape[2])
    return rows,(design.T@wrong.reshape(len(design),-1)/len(design)).astype(float)


def shrink(grams,moments,r):
    mean=moments.mean(0);error=moments.std(0)/np.sqrt(len(moments))
    return trial.curve.solve(grams.mean(0),mean*abs(mean)/(abs(mean)+error+1e-5),r)


def fit(hist):
    rows,placebo=zip(*hist)
    g,b,_,_,ge,_,be=[np.stack(v) for v in zip(*rows)]
    c=ge.shape[-1]
    pooled=shrink(g,b[:,:,c:2*c],.60)@shrink(ge,be,.70)[:,1]
    estimates=[]
    for k in (0,1):
        s=slice(k,None,2);o=slice(1-k,None,2)
        response=shrink(ge[o],be[o],.70)
        # Both transfers see equal sample counts and use the same stock features and regularization.
        maps=shrink(g[s],np.concatenate((b[s,:,:2*c],np.asarray(placebo)[s]),axis=2),.60)
        mid,end,wrong=np.hsplit(maps,3)
        estimates.append(np.column_stack((end@response[:,1],end@response[:,0],mid@response[:,1],wrong@response[:,1])))
    final,wrong_payoff,wrong_time,shuffled=np.mean(estimates,0).T
    w=np.column_stack((pooled,final,wrong_payoff,wrong_time,shuffled,final-shuffled,final-wrong_time))
    return dict(w=w.astype('float32')),dict(norms=np.linalg.norm(w,axis=0).tolist())


changes=occupancy.changes


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    # For identity and horizon controls use the target interval; full-score promotion uses both historical references.
    def target_beats(a,b):
        diff=f[:,:,2,a]-f[:,:,2,b]
        return diff.mean(0).min()>0 and (diff>0).sum(0).min()>=3
    if not target_beats(1,4) or not target_beats(1,3):return None
    ids=[i for i in (1,5,6) if trial.scoring.select(f[:,:,:,i:i+1],d[:,:,i:i+1]) is not None]
    return int(max(ids,key=lambda i:f[:,0,0,i].mean())) if ids else None


def correction():return trial.prior.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
