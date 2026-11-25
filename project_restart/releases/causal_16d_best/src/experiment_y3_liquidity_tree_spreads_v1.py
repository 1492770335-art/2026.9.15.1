"""Forecast sibling-portfolio spreads from pressure and activity within a training-derived stock hierarchy."""
from functools import lru_cache
import sys
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import joint_factor_stage_references as history
import experiment_y3_factor_order_response_v1 as factors
import experiment_y3_continuous_factor_order_v1 as spectrum

trial,study,parent=history.trial,history.study,history.incumbent
OUT=trial.clock.ART/'y3_liquidity_tree_spreads_v1'
ARMS=('linear projection of local orders','relative pressure with shared activity','common pressure with relative activity',
      'between portfolios beyond individual order','joint sibling responses','wrong leaf identities',
      'smallest sibling portfolios','broad cluster return','cross-side pressure and activity',
      'separate within-side heterogeneity','later sibling spread prediction')
CANDIDATES=(1,2,3,4,6,7,8,9,10)
CONTROLS=(0,5)
SPANS=(4,20,64)
NVIEW=6
WIDTH=36


@lru_cache(maxsize=7)
def tree(end):
    corr=np.mean([spectrum.cov(d) for d in range(end)],0)
    dist=np.sqrt(np.maximum(0,1-(corr+corr.T)*.5))
    np.fill_diagonal(dist,0)
    merges=linkage(squareform(dist,checks=False),method='average')
    groups=[np.eye(50)[i] for i in range(50)]
    left,right=[],[]
    sizes=[]
    for i,j,_,_ in merges:
        a,b=groups[int(i)],groups[int(j)]
        left.append(a/a.sum());right.append(b/b.sum())
        sizes.append((a.sum(),b.sum()))
        groups.append(a+b)
    a,b=np.asarray(sizes).T
    scale=np.sqrt(a*b/(a+b))
    left,right=np.stack(left),np.stack(right)
    h=(left-right)*scale[:,None]
    np.testing.assert_allclose(h@h.T,np.eye(49),atol=1e-9)
    return dict(left=left.astype('float32'),right=right.astype('float32'),nodes=h.astype('float32'),
                scale=scale.astype('float32'),levels=np.digitize(a+b,[5,13]))


def bank(raw,t,st,wrong=False,weights=None):
    left,right,haar=(st[k] for k in ('left','right','nodes'))
    if wrong:
        left,right,haar=[np.roll(a,17,axis=1) for a in (left,right,haar)]
    scale=st['scale'][None,:,None]
    v=factors.clocks.signature.impact.arrival.signals(raw)
    p=np.tanh(v[:,:,[0,2,3]])
    activity=np.tanh(np.stack((abs(v[:,:,0]),abs(v[:,:,3:5]).mean(2)),2))
    parts=[]
    for lo,hi in ((0,min(2401,len(raw))),(2401,len(raw))):
        ix=t[(t>=lo)&(t<hi)]-lo
        if not len(ix):
            continue
        z=np.concatenate((p[lo:hi],activity[lo:hi]),2)
        z-=factors.clocks.filters.smooth(z,128)
        l,r=left@z,right@z
        def product(p,a):
            return (p[:,:,:,None]*a[:,:,None,:]).reshape(len(p),p.shape[1],6)
        signal=np.zeros((len(ix),49,NVIEW,WIDTH if weights is None else 3),dtype='float32')
        for k,span in enumerate(SPANS):
            old=factors.clocks.filters.smooth(z,span)
            lag=np.concatenate((np.zeros_like(old[:1]),old[:-1]))
            a,b=left@lag,right@lag
            def pair(pa,ac,pnow,apast):
                return product(pa[:,:,:3],ac[:,:,3:])-product(pnow[:,:,:3],apast[:,:,3:])
            original=haar@pair(lag,z,z,lag)
            relative=.5*scale*pair(a-b,l+r,l-r,a+b)
            common=.5*scale*pair(a+b,l-r,l+r,a-b)
            within=scale*(pair(a,l,l,a)-pair(b,r,r,b))
            cross=scale*(pair(a,r,l,b)-pair(b,l,r,a))
            block=np.stack((original,relative,common,within-original,cross,original-within),2)
            for j,age in enumerate((8,48)):
                pos=6*(2*k+j)
                if weights is None:
                    signal[:,:,:,pos:pos+6]=factors.clocks.filters.smooth(block,age)[ix]
                else:
                    projected=np.einsum('tnvc,nvch->tnvh',block,weights[:,:,pos:pos+6],optimize=True)
                    signal+=factors.clocks.filters.smooth(projected,age)[ix]
        parts.append(signal)
    return np.concatenate(parts)


@lru_cache(maxsize=16)
def moments(d):
    raw=trial.clock.book.rawday(d)
    t=np.arange(75+d%11,len(raw)-60,11)
    t=t[(t+60<2401)|(t>=2401)]
    st=tree(max(4,d-d%2))
    y=trial.clock.labels(d).reshape(-1,50,3)
    y=trial.clock.center(np.stack([y[t+h,:,1] for h in (0,20,40)],2)).clip(-4,4)
    gs,bs=[],[]
    for wrong in (False,True):
        z=bank(raw,t,st,wrong)
        h=np.roll(st['nodes'],17,1) if wrong else st['nodes']
        target=h@y
        gram,rhs=[],[]
        for k in range(3):
            use=st['levels']==k
            x=z[:,use].reshape(-1,NVIEW,WIDTH).transpose(1,0,2).copy()
            v=target[:,use].reshape(-1,3)
            gram.append(x.transpose(0,2,1)@x/len(v))
            rhs.append(x.transpose(0,2,1)@v/len(v))
        gs.append(gram);bs.append(rhs)
    return np.array(gs),np.array(bs)


@lru_cache(maxsize=7)
def state(end):
    gs,bs=[np.asarray(v,dtype=float) for v in zip(*(moments(d) for d in range(end)))]
    g,b=gs.mean(0),bs.mean(0)
    b*=abs(b)/(abs(b)+bs.std(0)/np.sqrt(end)+1e-7)
    pooled=g.mean(1,keepdims=True)
    penalty=np.eye(WIDTH)*g.diagonal(axis1=-2,axis2=-1)[...,None,:].clip(.000002)
    return dict(heads=np.linalg.solve(g+.45*pooled+.95*penalty,b).astype('float32'),**tree(end))


def features(raw,t,st):
    def reconstruct(z,h):
        return np.einsum('tnh,ns->tsh',z,h,optimize=True)
    w=st['heads'][0,st['levels']]
    raw_scores=bank(raw,t,st,weights=w)
    h=st['nodes']
    y=[reconstruct(raw_scores[:,:,i],h) for i in range(NVIEW)]
    wrong=bank(raw,t,st,True,st['heads'][1,st['levels']])
    fake=reconstruct(wrong[:,:,1],np.roll(h,17,1))
    ends=[reconstruct(raw_scores[:,:,1]*(st['levels']==k)[None,:,None],h) for k in (0,2)]
    return np.concatenate((*y,fake,*ends),2)


def historical_design(d):
    st=state(max(4,d-d%2));cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=features(raw,t,st)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    s=[np.arange(3+3*i,6+3*i) for i in range(9)]
    groups=[s[0],s[1],s[2],s[3],np.r_[s[1],s[2]],s[6],s[7],s[8],s[4],np.r_[s[1],s[5]],s[1][1:]]
    return dict(**factors.clocks.linear.gram(rows,[(g,3,.43) for g in groups]),**state(len(rows)+4))


changes=factors.changes


def correction():
    with np.load(OUT/'model.npz') as f: st=dict(f)
    def apply(raw,p):
        z=features(raw,np.arange(len(raw)),st)
        return changes(np.concatenate((trial.clock.center(p),z),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
