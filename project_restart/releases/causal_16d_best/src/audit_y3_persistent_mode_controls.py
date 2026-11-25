"""Match the stock-local component in every comparison of the projected future-order forecasts."""
import sys

import numpy as np
import torch

import experiment_y3_persistent_order_modes_v1 as model

OUT=model.OUT/'matched_controls'
ARMS=('local alone','local plus direct','local plus future orders','local plus future prices',
      'local plus jointly derived modes','local plus pooled modes','local plus wrong-stock modes',
      'local plus future orders shifted across recipients','replicated order modes alone')
CANDIDATES=()
CONTROLS=()
changes=model.changes


def historical_design(day):
    st=model.params(max(4,day-day%2))
    def design(raw,p,t):
        direct,price,order,joint,_,pooled,wrong,_,local=model.forecast(raw,t,st)
        cols=(local,local+direct,local+order,local+price,local+joint,local+pooled,local+wrong,
              local+np.roll(order,17,axis=1),order)
        return model.trial.clock.center(np.concatenate((p[t],*cols),2)).astype('float32')
    return design


def fit(rows):
    return model.book.prev.gram(rows,[(np.arange(3+4*i,7+4*i),3,.35) for i in range(len(ARMS))])


if __name__=='__main__':
    torch.set_num_threads(2)
    model.study.score_run(sys.modules[__name__],reference=model.history)
