"""Compare three matched terminal-price news features with same-width price, forecast, time and receiver controls."""
import json
import sys
import numpy as np
import torch

import experiment_y3_peer_terminal_price_news_v1 as model

routing,study,trial,history=model.routing,model.study,model.trial,model.history
OUT=model.OUT/'matched_controls'
ARMS=('aligned news','unchanged forecast horizon','only spot repricing','only forecast revision',
      'recompute past revision at current price','permuted receiver and sender','quiet quote revisions',
      'current forecast without the preceding forecast')
CANDIDATES=()
CONTROLS=()


def design_with(raw,p,t,st):
    k=float(st['k'])
    cf=st['coeff']
    mid,half,_,_,valid,_,_,_=routing.queues.tape.prices(raw,st)
    prev=np.r_[0,np.arange(len(raw)-1)]
    active=(mid!=mid[prev])|(half!=half[prev])
    ok=valid&valid[prev]
    ok[0]=False
    spot=np.log1p(k*(raw[:,:,0].astype(float)-.00136))*cf[2,1]/k
    levels=np.log1p(k*((p-cf[:,0])/cf[:,1]).clip(-.95/k,None))*cf[2,1]/k
    now=np.arange(len(p))[:,None]
    g=routing.balanced(st['graphs'][1])
    perm=np.random.default_rng(38951).permutation(50)
    other=g[np.ix_(perm,perm)]
    old=(np.arange(len(raw))-10).clip(0)
    cols=[[] for _ in ARMS]
    for h in (5,20,40):
        present=model.deadline.curve(levels,now,h)
        rev=present-model.deadline.curve(levels,now,h+1)[prev]
        news=spot+rev
        vals=(news,spot+present-present[prev],spot,rev,spot+rev[old],news,news,present)
        for j,v in enumerate(vals):
            cs=np.vstack((np.zeros(50),np.where(ok&((~active) if j==6 else active),np.arcsinh(v),0).cumsum(0)))
            cols[j].extend([routing.recent(cs,t,w)@(other if j==5 else g) for w in (5,20,60)])
    return trial.clock.center(np.concatenate((p[t],*(np.stack(v,2) for v in cols)),2)).astype('float32')


def historical_design(day):
    st=model.params(max(4,day-day%2))
    return lambda raw,p,t:design_with(raw,p,t,st)


def fit(rows):
    return model.book.prev.gram(rows,[(np.arange(3+9*i,12+9*i),3,.35) for i in range(len(ARMS))])


changes=model.changes


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
    got=np.asarray(json.loads((OUT/'results.json').read_text())['folds'])[:,:,0,0]
    prior=np.asarray(json.loads((model.OUT/'results.json').read_text())['folds'])[:,:,0,9]
    np.testing.assert_allclose(got,prior,rtol=0,atol=4e-8)
