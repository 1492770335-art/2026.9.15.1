"""Correct Y3 from the two earliest settled return intervals of each stock."""
import argparse
import json

import numpy as np
import torch

import experiment_compounded_return_audit_v1 as tape
import experiment_y3_temporal_pair_alignment_v1 as pairs
from y3_portfolio_walkforward import replay

parent = pairs.parent
OUT = parent.ART / 'y3_early_settled_stock_slopes_v1'
BLOCK = 240
H = (5, 20, 60)
ARMS = ('incumbent', 'first five', 'next fifteen', 'first twenty', 'agreed part of short and medium', 'medium confirmed by far')



def fit(gram,prior_size):
    # Shrink each stock toward a pooled response, with a zero prior for the shared return error.
    gram=gram.astype(float)
    pooled=gram.mean(0)
    scale=pooled.diagonal()[:4]/max(pooled[0,0],1.)
    diag=np.diag(scale.clip(.015))
    common=np.linalg.solve(pooled[:4,:4]+prior_size*.6*diag,pooled[:4,4])
    beta=np.linalg.solve(gram[:,:4,:4]+prior_size*diag,(gram[:,:4,4]+prior_size*common@diag)[:,:,None])[:,:,0]
    return beta,common


def geometry():
    with np.load(tape.OUT/'model.npz') as f:return float(f['k']),f['coeff'],f['row']


def collect(raw,p,geo):
    k,co,template=geo
    scale=co[2,1]/co[:,1]
    curve=(p-co[:,0])*scale
    legs=np.diff(curve,axis=2,prepend=np.zeros_like(curve[:,:,:1]))
    legs-=legs.mean(1,keepdims=True)
    x=np.concatenate((np.ones_like(legs[:,:,:1]),legs),2)
    groups=[]
    for a,b in parent.base.segments(len(p)):
        r=raw[a:b];n=len(r)
        logs=tape.compound(r,k)
        miss=np.vstack((np.zeros((1,50),int),(~tape.present(r,template)).cumsum(0)))
        v=np.zeros(((n+BLOCK-1)//BLOCK,3,50,5,5))
        for j,h in enumerate(H):
            t=np.arange(0,n-h,5)
            realized=co[2,1]*tape.span(logs,t,h,k)
            if j:realized-=co[2,1]*tape.span(logs,t,H[j-1],k)
            ok=miss[t+h+1]==miss[t]
            err=realized-legs[a+t,:,j]
            err-=(err*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1)
            feature=x[a+t].copy()
            feature[:,:,j+2:]=0
            z=np.concatenate((feature,err.clip(-4,4)[:,:,None]),2)
            for block in range(len(v)):
                ix=(t+h)//BLOCK==block
                v[block,j]=np.einsum('tsi,tsj->sij',z[ix]*ok[ix,:,None],z[ix],optimize=True)
        groups.append(v)
    return legs,groups


def interval_changes(p,x,groups,prior):
    out=np.empty((len(p),50,7),np.float32)
    for (a,b),grams in zip(parent.base.segments(len(p)),groups):
        for j,t in enumerate(range(a,b,BLOCK)):
            sl=slice(t,min(b,t+BLOCK))
            # Observations are bucketed by actual settlement time of that leg.
            # At this block boundary every earlier bucket can already be used.
            stats=.35*prior+grams[:j].sum(0)
            beta=[]
            for v in stats:
                stock,common=fit(v,360.)
                beta.append(stock[:,1:]-common[1:])
            estimates=[(x[sl]*coef).sum(2) for coef in beta]
            near,mid,far=estimates
            wrong=(x[sl]*np.roll(sum(beta),17,axis=0)).sum(2)
            vals=np.stack((near+mid+far,near,mid,far,near+mid,near+far,wrong),2)
            out[sl]=.30*(vals-vals.mean(1,keepdims=True))
    return out


def signals(raw,p,prior,geo):
    x,blocks=collect(raw,p,geo)
    z=interval_changes(p,x,blocks,prior)
    short,medium,far=z[:,:,1],z[:,:,2],z[:,:,3]
    def agree(a,b):return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
    a=np.stack((short,medium,short+medium,agree(short,medium),agree(medium,far)),2)
    a-=a.mean(1,keepdims=True)
    return a,blocks


def gram(p,y,v):
    a=np.concatenate((np.ones_like(y[:,:,None]),p[:,:,None],y[:,:,None],v),2)
    a=a.reshape(-1,a.shape[-1]).astype(float)
    return a.T@a


def gains(stats):
    m=sum(stats)
    c=m[1:,1:]/m[0,0]-np.outer(m[0,1:]/m[0,0],m[0,1:]/m[0,0])
    r=c[0,1]/np.sqrt(c[0,0]*c[1,1])
    return np.r_[0,(c[0,1]+c[1,2:])/np.sqrt(c[1,1]*(c[0,0]+2*c[0,2:]+c.diagonal()[2:]))-r]


def train():
    geo=geometry();hist=[];days=[]
    ratio=geo[1][2,1]/geo[1][1,1]
    for d,raw,ps,y,_ in pairs.panels():
        p=ps[-1] if d>=8 else np.mean(ps[:2],0)
        if d>=8:
            prior=np.mean(hist,0)
            v,blocks=signals(raw,p,prior,geo)
            if d==8:
                n=2451
                np.testing.assert_allclose(v[:n],signals(raw[:n],p[:n],prior,geo)[0],rtol=0,atol=3.e-6)
            days.append((gram(p[:,:,1],y[:,:,1],v/ratio),gram(p[:,:,2],y[:,:,2],v)))
        else:_,blocks=collect(raw,p,geo)
        hist.append(sum(b.sum(0) for b in blocks));print('day',d+1,flush=True)
    def combined(group):return np.array([gains(v) for v in zip(*group)])
    h=np.array([combined(g) for g in (days[:4],days[4:])])
    daily=np.array([combined([v]) for v in days])
    composite=.3*h[:,0]+.5*h[:,1]
    j=max(range(1,len(ARMS)),key=lambda k:composite[0,k] if h[0,:,k].min()>0 else -1)
    keep=h[:,:,j].min()>0 and ((.3*daily[:,0,j]+.5*daily[:,1,j])>0).sum()>=5
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=j if keep else 0,choice=j,
                    halves=h.tolist(),daily=daily.tolist()),indent=2))
    print('selected',j if keep else 0,'Y2, Y3 gains',h[:,:,j],flush=True)
    if keep:np.savez(OUT/'model.npz',prior=np.mean(hist,0),choice=j)


def correction():
    geo=geometry()
    prior=pairs.ensemble.correction()
    f=torch.load(pairs.OUT/'model.pt',weights_only=True,map_location='cpu')
    net=pairs.Spread();net.load_state_dict(f['state']);net.eval()
    with np.load(OUT/'model.npz') as saved:g,j=saved['prior'],int(saved['choice'])
    def update(raw,p):
        base=prior(raw,p)
        q=p.copy();q[:,:,2]+=base
        lag=pairs.shifts(pairs.prefix(raw),q,net)[0][:,:,f['pick']-1]
        short=signals(raw,q,g,geo)[0][:,:,j-1]
        return base+lag+short
    return update

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train','verify'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=='train':
        train()
    else:
        replay(OUT,correction)
