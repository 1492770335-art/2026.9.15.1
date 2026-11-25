"""Use the same four return windows and rolling correction fit to compare order-price propagation mechanisms."""
import sys

import numpy as np
import torch

import experiment_y3_joint_order_price_rollout_v1 as model

OUT=model.OUT/'matched_controls'
ARMS=('direct','full','suppress future orders','suppress price feedback','own stock only','shuffled order destinations',
      'full minus direct','full minus own stock')
CANDIDATES=()
CONTROLS=()
changes=model.changes


def historical_design(day):
    st=model.params(max(4,day-day%2))
    def design(raw,p,t):
        forecasts=model.estimates(raw,t,st).transpose(1,0,3,2)
        direct,full,_,_,own,_=forecasts
        x=np.concatenate((p[t],*forecasts,full-direct,full-own),axis=2)
        return model.trial.clock.center(x).astype('float32')
    return design


def fit(rows):
    groups=[(np.arange(3+4*j,7+4*j),3,.35) for j in range(len(ARMS))]
    return model.book.prev.gram(rows,groups)


if __name__=='__main__':
    torch.set_num_threads(2)
    model.study.score_run(sys.modules[__name__],reference=model.history)
