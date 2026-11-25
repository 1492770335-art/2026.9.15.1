"""Separate predictable terminal occupancy carried by categorical history from new continuous pressure information."""
import sys

import numpy as np
import torch

import experiment_y3_crossfitted_future_states_v1 as cross
import y3_terminal_validation as trial

OUT=trial.clock.ART/'y3_state_transition_sources_v1'
ARMS=['full mediator','categorical history','numeric history','increment beyond categories',
      'increment beyond pressure','reassigned category history','reassigned numeric history','equal contributions']
statistics=trial.statistics
SECONDARY=cross.SECONDARY
state,observe,changes=cross.state,cross.observe,cross.changes


def fit(hist):
    old,shuffled=zip(*hist)
    gram,forward,_,_,end,_,pay=[np.stack(v) for v in zip(*old)]
    n=end.shape[-1];b=forward[:,:,n:2*n]
    width=b.shape[1]
    # At each timescale, 8 pressure and 4 paired-level channels precede the discrete-state indicators.
    index=np.arange(width).reshape(4,-1)
    numeric=index[:,:12].ravel(); categories=index[:,12:].ravel()
    weights=[]
    for cols in (index.ravel(),categories,numeric):
        g=gram[:,cols][:,:,cols]
        w=cross.shrink(g,b[:,cols],.6)
        other=cross.shrink(g,np.stack(shuffled)[:,cols],.6)
        price=cross.shrink(end,pay,.7)[:,1]
        tmp=np.zeros((width,2))
        tmp[cols,0]=w@price;tmp[cols,1]=other@price
        weights.append(tmp)
    full,cat,num=weights
    w=np.column_stack((full[:,0],cat[:,0],num[:,0],full[:,0]-cat[:,0],full[:,0]-num[:,0],
                       cat[:,1],num[:,1],.5*(cat[:,0]+num[:,0])))
    return dict(w=w.astype('float32')),dict(dates=len(hist))


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    def ok(a,b=None):
        x,y=f[:,:,:,a:a+1],d[:,:,a:a+1]
        if b is not None:x,y=x-f[:,:,:,b:b+1],y-d[:,:,b:b+1]
        return trial.scoring.select(x,y) is not None
    ids=[a for a,b in ((1,5),(2,6),(3,6),(4,5),(7,5)) if ok(a) and ok(a,b)]
    return int(max(ids,key=lambda i:f[:,0,0,i].mean())) if ids else None


def correction():return trial.prior.correction(sys.modules[__name__])
def layer():return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
