"""Adjust a stock's delayed cash-return forecast by its observed sensitivity to other stocks' portfolios."""
from functools import lru_cache
import json
import sys
import numpy as np
import torch
import y3_receiver_portfolio_references as history

parent=history.incumbent
model,study,trial,book=parent.model,parent.study,parent.trial,parent.book
OUT=trial.clock.ART/'y3_realized_recipient_exposures_v1'
ARMS=('fixed exposure','recent return exposure','recent order exposure','cross-activity agreement','relative variances alone',
      'permuted recipient update','older return exposure','local plus current exposure','local plus fixed exposure',
      'local plus wrong exposure','order-induced return exposure')
CANDIDATES=(1,2,3,7,10)
CONTROLS=(0,4,5,6,8,9)
STEP=8


def observations(raw,st):
    direction=(raw[:,:,54]>st['floors'][1]).astype(float)-(raw[:,:,55]>st['floors'][2])
    k=float(st['k'])
    ret=np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136))))/k
    blocks=len(raw)//STEP
    def pooled(x,scale):
        return x[:blocks*STEP].reshape(blocks,STEP,50).sum(1)/(scale*np.sqrt(STEP))
    return pooled(ret,st['sc'][50:]),pooled(direction,st['sc'][:50])


@lru_cache(maxsize=16)
def covariances(d):
    raw=trial.clock.book.rawday(d)
    r,o=observations(raw,model.calibration())
    r-=r.mean(0)
    o-=o.mean(0)
    return np.stack((r.T@r,o.T@o,o.T@r))/len(r)


def leave_one_out_basis(basis):
    # For each receiver j, its own weight is zero and every peer portfolio sums to zero.
    w=np.broadcast_to(basis,(50,*basis.shape)).copy()
    mask=1-np.eye(50)
    w*=mask[:,:,None]
    return (w-w.sum(1,keepdims=True)/49)*mask[:,:,None]


def loading(cov,cross,w):
    # The target is the receiving stock relative to the mean of the other names.
    local_target=(cross.T-cross.mean(1)[None,:])*(50/49)
    gram=w.transpose(0,2,1)@(cov@w)
    b=(w.transpose(0,2,1)@local_target[:,:,None])[:,:,0]
    regularizer=.3*gram.diagonal(axis1=1,axis2=2).mean(1)+.025
    return np.linalg.solve(gram+regularizer[:,None,None]*np.eye(w.shape[2]),b[:,:,None])[:,:,0]


@lru_cache(maxsize=7)
def params(end):
    st=parent.params(end)
    matrix=st['weights'][4].reshape(-1,50).astype(float)
    _,v=np.linalg.eigh(matrix.T@matrix)
    basis=leave_one_out_basis(v[:,-8:])
    prior=np.mean([covariances(d) for d in range(end)],0)
    coef=np.stack([loading(prior[a],prior[b],basis) for a,b in ((0,0),(1,1),(1,2))])
    return dict(**{k:v for k,v in st.items() if k!='weights'},prior=prior,basis=basis,coef=coef,
                weights=st['weights'][[4,9],:,2:,:])


def update(r,o,start,st,offset=0):
    right=max(0,start//STEP-offset)
    left=max(0,right-120)
    cov=st['prior']
    if right<=left+1:return np.zeros_like(st['coef']),np.zeros_like(st['coef'][0])
    a=r[left:right]-r[left:right].mean(0)
    b=o[left:right]-o[left:right].mean(0)
    n=len(a)
    gram=(40*cov+np.stack((a.T@a,b.T@b,b.T@a)))/(40+n)
    ratio=np.sqrt((gram[0].diagonal()/cov[0].diagonal().clip(.001)).clip(.15,6))
    diag=cov[0]*ratio[None,:]*ratio[:,None]
    fit=np.stack([loading(gram[i],gram[j],st['basis']) for i,j in ((0,0),(1,1),(1,2))])
    return fit-st['coef'],loading(diag,diag,st['basis'])-st['coef'][0]


def components(raw,t,st):
    x,_=model.inputs(raw,t,st)
    ws=st['weights'].transpose(1,0,2,3).reshape(500,-1)
    vals=(x.astype('float32')@ws).reshape(len(t),2,2,50)
    base,local=vals[:,0],vals[:,1]
    r,o=observations(raw,st)
    # Each receiver observes its distinct leave-one-out portfolio of the current predicted moves.
    scores=np.einsum('ths,rsk->thrk',base,st['basis'],optimize=True)
    out=np.zeros((len(ARMS),len(t),2,50),np.float32)
    for start in np.unique(t//80)*80:
        take=(t>=start)&(t<start+80)
        change,diag=update(r,o,start,st)
        stale,_=update(r,o,start,st,90)
        def pay(beta):return np.einsum('thrk,rk->thr',scores[take],beta)
        a,b,c=[pay(beta) for beta in change]
        wrong=pay(np.roll(change[0],17,axis=0))
        v=base[take];loc=local[take]
        terms=(v,v+a,v+b,v+(a+b)/2,v+pay(diag),v+wrong,v+pay(stale[0]),loc+v+a,
               loc+v,loc+v+wrong,v+c)
        for dest,u in zip(out,terms):dest[take]=u*st['sc'][50:]
    return out.transpose(0,1,3,2)


def design_with(raw,p,t,st):
    return trial.clock.center(np.concatenate((p[t],*components(raw,t,st)),2)).astype('float32')


def historical_design(d):
    st=params(max(4,d-d%2))
    return lambda raw,p,t:design_with(raw,p,t,st)


def fit(rows):
    return dict(**book.prev.gram(rows,[(np.arange(3+2*i,5+2*i),3,.35) for i in range(len(ARMS))]),**params(len(rows)+4))


changes=book.changes


def correction():
    with np.load(OUT/'model.npz') as f:st=dict(f)
    def apply(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),st),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


def audit():
    results=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d);st=params(d-d%2)
        r,o=observations(raw,st)
        out=[]
        for t in range(960,len(raw)-320,320):
            delta,vol=update(r,o,t,st)
            actual=r[t//8:t//8+40]
            y=(actual-actual.mean(1,keepdims=True))*50/49
            portfolio=np.einsum('ts,rsk->trk',actual,st['basis'],optimize=True)
            vals=[]
            for adj in (0,delta[0],vol,np.roll(delta[0],17,axis=0),delta[1]):
                estimate=np.einsum('trk,rk->tr',portfolio,st['coef'][0]+adj)
                vals.append(1-np.sum((y-estimate)**2)/np.sum(y*y))
            out.append(vals)
        results.append(np.mean(out,0))
    report=dict(days=[9,16],target='next 320 frames contemporaneous relative return given the realized other-stock returns',
                controls=['historical','updated returns','diagonal only','permuted update','updated orders'],
                daily=np.array(results).tolist(),mean=np.mean(results,0).tolist())
    (OUT/'next_block_exposure.json').write_text(json.dumps(report,indent=2))
    print('realized exposure R2',np.round(report['mean'],5),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
    audit()
