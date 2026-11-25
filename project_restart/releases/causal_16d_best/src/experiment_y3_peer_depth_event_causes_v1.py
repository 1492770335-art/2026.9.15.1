"""Route liquidity messages according to whether a displayed price moved or its stationary queue changed."""
import sys
import numpy as np
import torch

import y3_peer_depth_routing as source

trial,study,history=source.trial,source.study,source.history
OUT=trial.clock.ART/'y3_peer_depth_event_causes_v1'
ARMS=('undivided flow','stationary quotes','moving quotes','improving price levels',
      'price worsening','both price directions and stable quotes','stationary passive volume',
      'stationary traded volume','permuted passive volume','quote direction alone',
      'passive volume and displaced queues')
CANDIDATES=(1,2,3,4,5,6,10)
CONTROLS=(0,7,8,9)


def design_with(raw,p,t,state):
    mid,half,_,_,valid,_,buy,sell=source.queues.tape.prices(raw,state)
    bid,ask=source.queues.size(raw,state)
    prev=np.r_[0,np.arange(len(raw)-1)]
    ok=valid&valid[prev]
    ok[0]=False
    groups=np.zeros((len(raw),50,7),dtype=float)
    for sign,level,size,vol in ((1,mid-half,bid,sell),(-1,mid+half,ask,buy)):
        change=sign*(level-level[prev])
        same=change==0
        net=(size-size[prev])*same
        improve=(change>0)*size
        lost=-(change<0).astype(float)*size[prev]
        for i,v in enumerate((net,improve,lost,net+same*vol,-(same*vol),np.sign(change),net+improve+lost)):
            groups[:,:,i]+=sign*v
    matrix=source.balanced(state['graphs'][1])
    perm=np.random.default_rng(870223).permutation(50)
    def routing(v,graph=matrix,scaled=True):
        v=np.where(ok,np.arcsinh(v/state['volume_scale']) if scaled else v,0.)
        cs=np.vstack((np.zeros(50),v.cumsum(0)))
        return np.stack([source.recent(cs,t,w)@graph for w in (5,20,60)],2)
    stable,up,down,passive,trade,direction,total=np.moveaxis(groups,2,0)
    parts=(routing(total),routing(stable),routing(up+down),routing(up),routing(down),routing(passive),
           routing(trade),routing(passive,matrix[np.ix_(perm,perm)]),routing(direction,scaled=False))
    return trial.clock.center(np.concatenate((p[t],*parts),2)).astype('float32')


def historical_design(day):
    state=source.params(max(4,day-day%2))
    return lambda raw,p,t:design_with(raw,p,t,state)


def fit(rows):
    total,steady,moved,up,down,passive,trade,wrong,direction=[np.arange(3+3*i,6+3*i) for i in range(9)]
    sets=(total,steady,moved,up,down,np.r_[steady,up,down],passive,trade,wrong,direction,np.r_[passive,up,down])
    return dict(**source.book.prev.gram(rows,[(g,3,.35) for g in sets]),**source.params(len(rows)+4))


changes=source.book.changes


def correction():
    with np.load(OUT/'model.npz') as f:state=dict(f)
    def update(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),state),state)[int(state['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
