"""Hold stock-local forecasts fixed while varying the cross-stock destination structure."""
import sys
import numpy as np
import torch
import experiment_y3_receiver_price_portfolios_v1 as model

OUT=model.OUT/'matched_controls'
ARMS=('local','local plus direct','local plus early recipients','local plus late recipients','local plus lasting recipients',
      'local plus shuffled recipients','local plus lasting diagonal','local plus lasting cross-stock','local plus common cross-stock weights')
CONTROLS=()
CANDIDATES=()
changes=model.changes


def historical_design(day):
    st=model.params(max(4,day-day%2))
    weights=st['weights']
    local=weights[9]
    lasting=weights[4].reshape(10,50,4,50)
    diagonal=np.eye(50)[None,:,None,:]
    own=lasting*diagonal
    peers=lasting*(1-diagonal)
    common=peers.sum(1,keepdims=True)*(1-diagonal)/49
    raw=[np.zeros_like(local),weights[0],weights[1],weights[2],weights[4],weights[6],
         own.reshape(500,4,50),peers.reshape(500,4,50),common.reshape(500,4,50)]
    state=dict(st,weights=np.stack([local+w for w in raw]).astype('float32'))
    def design(raw,p,t):
        vals=model.model.forecast(raw,t,state)
        return model.trial.clock.center(np.concatenate((p[t],*[v[:,:,2:] for v in vals]),2)).astype('float32')
    return design


def fit(rows):
    return model.book.prev.gram(rows,[(np.arange(3+2*i,5+2*i),3,.35) for i in range(len(ARMS))])


if __name__=='__main__':
    torch.set_num_threads(2)
    model.study.score_run(sys.modules[__name__],reference=model.history)
