"""Determine which source stocks supply the incremental signal after holding the local forecast fixed."""
import sys
import numpy as np
import torch

import experiment_y3_persistent_order_modes_v1 as model

OUT=model.OUT/'source_controls'
ARMS=('local alone','local plus replicated modes','local plus diagonal modes','local plus off-diagonal modes',
      'local plus common peer loading','local plus above-common peer loading','local plus permuted pair loading',
      'local plus full modes with date-independent reassigned individual peers')
CANDIDATES=()
CONTROLS=()
changes=model.changes


def historical_design(d):
    st=model.params(max(4,d-d%2))
    # Order coefficients map lag, variable and source stock to horizon and destination stock.
    w=st['weights'][2].astype(float).reshape(10,50,4,50)
    diag=np.eye(50)[None,:,None,:]
    own=w*diag
    peer=w*(1-diag)
    common=peer.sum(1,keepdims=True)*(1-diag)/49
    diff=peer-common
    perm=np.random.default_rng(4031).permutation(50)
    shuffled=diff[:,perm,:,:][:,:,:,perm]
    local=st['weights'][8].reshape(10,50,4,50)
    bank=np.stack((local,local+w,local+own,local+peer,local+common,local+diff,local+shuffled,local+own+common+shuffled))
    state=dict(st,weights=bank.reshape(8,500,4,50).astype('float32'))
    def design(raw,p,t):
        values=model.forecast(raw,t,state)
        return model.trial.clock.center(np.concatenate((p[t],*values),2)).astype('float32')
    return design


def fit(rows):
    return model.book.prev.gram(rows,[(np.arange(3+4*i,7+4*i),3,.35) for i in range(len(ARMS))])


if __name__=='__main__':
    torch.set_num_threads(2)
    model.study.score_run(sys.modules[__name__],reference=model.history)
