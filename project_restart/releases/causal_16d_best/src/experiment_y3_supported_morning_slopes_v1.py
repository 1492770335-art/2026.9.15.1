"""Estimate replicated early stock exposures and transfer morning information to the afternoon."""
import argparse
import json

import numpy as np
import torch

import experiment_y3_early_settled_stock_slopes_v1 as early_slopes
from y3_portfolio_walkforward import replay, scores

pairs = early_slopes.pairs
parent = early_slopes.parent
OUT = parent.ART / 'y3_supported_morning_slopes_v1'
ARMS = ('early slope correction','reliable exposures','morning transfer','both increments')



def beta(g):
    out=[]
    for v in g[:2]:
        per,common=early_slopes.fit(v,360.)
        out.append(per[:,1:3]-common[1:3])
    return sum(out)


def morning_parameters(hist):
    gg=np.stack(hist)
    tot=gg.sum(1)
    prior=tot.mean(0)
    rows=[]
    for i,(morning,afternoon) in enumerate(gg):
        previous=(tot.sum(0)-tot[i])/(len(tot)-1)
        baseline=beta(.35*previous)
        a=beta(.35*previous+morning)-baseline
        b=beta(.35*previous+afternoon)-baseline
        a=a-a.mean(0);b=b-b.mean(0)
        rows.append((a.T@a,a.T@b))
    def rotation(part):
        aa,ab=[np.mean(v,0) for v in zip(*part)]
        w=np.linalg.solve(aa+1.5*np.diag(aa.diagonal().clip(.00004)),ab)
        u,d,v=np.linalg.svd(w,full_matrices=False)
        return (u*d.clip(0,.60))@v
    w=rotation(rows)
    a,b=[rotation(rows[i::2]) for i in (0,1)]
    stable=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
    return prior,w,stable


def morning_changes(p,x,sections,state):
    prior,w,stable=state
    early=early_slopes.interval_changes(p,x,sections,prior)[:,:,4]
    out=np.zeros((len(p),50,5),np.float32)
    sessions=parent.base.segments(len(p))
    if len(sessions)<2:return early,out
    a,b=sessions[1]
    base=beta(.35*prior)
    morning=beta(.35*prior+sections[0].sum(0))-base
    morning-=morning.mean(0)
    for j,t in enumerate(range(a,b,early_slopes.BLOCK)):
        sl=slice(t,min(b,t+early_slopes.BLOCK))
        pm=beta(.35*prior+sections[1][:j].sum(0))-base
        mapped=morning@w
        confirmed=np.where(mapped*pm>0,np.sign(mapped)*np.minimum(abs(mapped),abs(pm)),0.)
        coefs=(.22*morning,mapped,morning@stable,confirmed,np.roll(mapped,17,axis=0))
        val=np.stack([(x[sl,:,:2]*coef).sum(2) for coef in coefs],2)
        out[sl]=(.30/(1+j/4))*(val-val.mean(1,keepdims=True))
    return early,out


def betas(prior,live):
    slopes=[]
    for g in (.35*prior+live)[:2]:
        per,pooled=early_slopes.fit(g,360.)
        slopes.append(per[:,1:]-pooled[1:])
    return sum(slopes)


def confidence_parameters(hist):
    return np.stack([np.mean(h,0) for h in (hist,hist[::2],hist[1::2])])


def confidence_changes(p,x,blocks,priors):
    original=np.empty((len(p),50),np.float32)
    delta=np.empty((len(p),50,6),np.float32)
    for (a,b),gr in zip(parent.base.segments(len(p)),blocks):
        for j,t in enumerate(range(a,b,early_slopes.BLOCK)):
            sl=slice(t,min(b,t+early_slopes.BLOCK))
            past=gr[:j]
            def value(prior,live):
                return .30*(x[sl]*betas(prior,live)).sum(2)
            base=value(priors[0],past.sum(0))
            base-=base.mean(1,keepdims=True)
            original[sl]=base
            # Shift both the historical dates and the within-day sample separately, preserving effective sample size.
            samples=np.stack([[value(h,2*past[k::2].sum(0)) for k in (0,1)] for h in priors[1:]])
            samples-=samples.mean(3,keepdims=True)
            hv=.5*(samples[0].mean(0)-samples[1].mean(0))
            tv=.5*(samples[:,0].mean(0)-samples[:,1].mean(0))
            def shrink(v):return (base**2+.000004)/(base**2+v+.000004)
            hwt=shrink(hv**2);twt=shrink(tv**2)
            w=shrink(hv**2+tv**2)
            trusted=w*base
            denom=np.sqrt(np.mean(trusted**2,1,keepdims=True)+.00000001)
            ratio=np.clip(np.sqrt(np.mean(base**2,1,keepdims=True))/denom,1.,2.)
            low,high=samples.min((0,1)),samples.max((0,1))
            consistent=np.where(low>0,low,np.where(high<0,high,0.))
            values=np.stack((hwt*base,twt*base,trusted,trusted*ratio,consistent,np.roll(w,17,axis=1)*base),2)
            delta[sl]=values-values.mean(1,keepdims=True)-base[:,:,None]
    return original,delta


def fit(hist):
    carry=morning_parameters(hist)
    risk=confidence_parameters([r.sum(0) for r in hist])
    return carry,risk


def changes(p,x,blocks,models):
    carry,risk=models
    early,w=confidence_changes(p,x,blocks,risk)
    early2,m=morning_changes(p,x,blocks,carry)
    np.testing.assert_allclose(early,early2,rtol=0,atol=3e-6)
    cw=w[:,:,3];cm=m[:,:,1]
    return early,np.stack((cw,cm,cw+cm),2)


def train():
    hist,days=[],[]
    geo=early_slopes.geometry()
    for d,raw,ps,y,_ in pairs.panels():
        p=ps[-1] if d>=8 else np.mean(ps[:2],0)
        x,blocks=early_slopes.collect(raw,p,geo)
        if d>=8:
            models=fit(hist)
            early,z=changes(p,x,blocks,models)
            if d==8:
                n=2723;sx,sb=early_slopes.collect(raw[:n],p[:n],geo)
                np.testing.assert_allclose(z[:n],changes(p[:n],sx,sb,models)[1],rtol=0,atol=4e-6)
            q=p.copy();q[:,:,2]+=early
            days.append([parent.base.records(q,y,z)])
        hist.append(np.stack([b.sum(0) for b in blocks]));print('day',d+1,flush=True)
    h=np.stack([scores(v,len(ARMS))[0] for v in (days[:4],days[4:])])
    ds=np.stack([scores([v],len(ARMS))[0] for v in days])
    keep=(h[:,0,3:4]>h[:,0,1:3]).all() and ((ds[:,0,3:4]>ds[:,0,1:3]).sum(0)>=5).all()
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=3 if keep else 0,halves=h.tolist(),daily=ds.tolist()),indent=2))
    print('combined passed',keep,'vs early',h[:,0,:],flush=True)
    if keep:
        (prior,w,stable),risk=fit(hist)
        np.savez(OUT/'model.npz',prior=prior,w=w,stable=stable,risk=risk)


def correction():
    import torch
    prior=pairs.ensemble.correction();geo=early_slopes.geometry()
    f=torch.load(pairs.OUT/'model.pt',weights_only=True,map_location='cpu')
    net=pairs.Spread();net.load_state_dict(f['state']);net.eval()
    with np.load(OUT/'model.npz') as saved:models=((saved['prior'],saved['w'],saved['stable']),saved['risk'])
    def update(raw,p):
        base=prior(raw,p);q=p.copy();q[:,:,2]+=base
        lag=pairs.shifts(pairs.prefix(raw),q,net)[0][:,:,f['pick']-1]
        x,blocks=early_slopes.collect(raw,q,geo)
        early,z=changes(q,x,blocks,models)
        return base+lag+early+z[:,:,2]
    return update

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('train','verify'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=='train':
        train()
    else:
        replay(OUT,correction)
