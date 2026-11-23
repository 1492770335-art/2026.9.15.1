"""Infer relative returns by fitting pairwise boosted trees over a connected stock graph."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps/tree_models'))
import lightgbm as lgb
from experiment_causal_cross_horizon_ridge_v1 import ART,CACHE,labels,corr,measure

OUT=ART/'pairwise_book_tree_v1'
FIELDS=[0,7,10,11,13,14,16,23,26,29,34,36,38,39,40,41,42,50,51,54,55,60,64,72,74,77,80,94]


def rawday(d):
    return np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')


def graph():
    cov=[]
    for d in range(8):
        r=rawday(d)[:,:,0].astype('float32')
        rr=r[:len(r)//24*24].reshape(-1,24,50).sum(1)
        rr-=rr.mean(1,keepdims=True)
        cov.append(np.corrcoef(rr.T))
    _,axes=np.linalg.eigh(np.mean(cov,0))
    ids=np.argsort(axes[:,-1])
    i=np.tile(ids,3);j=np.concatenate([np.roll(ids,-gap) for gap in (1,5,18)])
    incidence=np.eye(50)[i]-np.eye(50)[j]
    inverse=np.linalg.solve(incidence.T@incidence+np.ones((50,50))/50+.2*np.eye(50),incidence.T)
    return i,j,inverse.astype('float32')


def state(raw):
    z=np.arcsinh(raw[:,:,FIELDS].astype('float32'))
    pairs=np.stack([z[:,:,15]-z[:,:,16],z[:,:,17]-z[:,:,18],z[:,:,19]-z[:,:,20]],axis=2)
    z=np.concatenate([z,pairs],axis=2)
    sub=[0,2,3,5,7,8,10,12,21,23,28,29,30]
    v=z[:,:,sub]
    c=np.concatenate([np.zeros_like(v[:1],dtype=float),v.cumsum(0,dtype=float)])
    t=np.arange(1,len(v)+1)
    blocks=[z]
    for h in (6,30,128):
        left=(t-h).clip(0)
        blocks.append((v-(c[t]-c[left])/(t-left)[:,None,None]).astype('float32'))
    r=raw[:,:,0].astype('float32')-.00136
    cs=np.vstack([np.zeros(50),np.cumsum(r*r,axis=0,dtype=float)])
    left=(t-96).clip(0)
    vol=np.sqrt((8+cs[t]-cs[left])/(8+t-left)[:,None]).clip(.5,4).astype('float32')
    return np.concatenate(blocks,axis=2),vol


def matrix(s,v,t,i,j):
    a=s[t,i];b=s[t,j]
    scale=.4+.3*(v[t,i]+v[t,j])
    # Symmetric modifiers can change the conditional size of the signed book differences.
    avg=(a[:,:31]+b[:,:31])*.5
    meta=np.stack([np.log(scale),np.log(v[t,i]/v[t,j]),np.exp(-t/320)],axis=1)
    return np.concatenate([a-b,avg,meta],axis=1).astype('float32'),scale


def fit(end,graph_info):
    left,right,_=graph_info;xs=[];ys=[]
    rng=np.random.default_rng(481)
    for d in range(end):
        raw=rawday(d);s,v=state(raw);y=labels(d).reshape(-1,50,3)
        ids=np.r_[np.arange(d%4,600,4),np.arange(600+(d%16),len(s)-61,16)]
        t=np.repeat(ids,50)
        edge=np.tile(np.arange(50),len(ids))+50*rng.integers(3,size=len(t))
        reverse=rng.integers(2,size=len(t))==1
        i=np.where(reverse,left[edge],right[edge]);j=np.where(reverse,right[edge],left[edge])
        x,scale=matrix(s,v,t,i,j)
        xs.append(x);ys.append(((y[t,i]-y[t,j])/scale[:,None]).clip(-5,5))
    x=np.concatenate(xs);del xs
    y=np.concatenate(ys);del ys
    models=[]
    for k in range(3):
        ds=lgb.Dataset(x,label=y[:,k],params={'max_bin':47})
        model=lgb.train(dict(objective='huber',alpha=1.6,learning_rate=.055,num_leaves=23,
                  min_data_in_leaf=250,lambda_l2=22,feature_fraction=.78,bagging_fraction=.82,
                  bagging_freq=3,max_bin=47,num_threads=2,seed=182+k,force_col_wise=True,verbosity=-1),
                  ds,num_boost_round=138)
        models.append(model);print('trained',end,k+1,flush=True)
        if end==16:model.save_model(str(OUT/f'tree_{k}.txt'))
        del ds
    return models


def predict(raw,models,graph_info):
    left,right,inverse=graph_info
    s,v=state(raw);out=np.empty((len(s),50,3),np.float32)
    for lo in range(0,len(s),96):
        ids=np.arange(lo,min(lo+96,len(s)));t=np.repeat(ids,len(left))
        i=np.tile(left,len(ids));j=np.tile(right,len(ids))
        x,scale=matrix(s,v,t,i,j)
        # Every graph edge is visited in both directions before reconstructing returns.
        flip,_=matrix(s,v,t,j,i)
        gaps=[]
        for model in models:
            gap=(model.predict(x,num_threads=2)-model.predict(flip,num_threads=2))*.5
            gaps.append((gap*scale).reshape(len(ids),len(left)))
        out[ids]=inverse@np.stack(gaps,axis=2)
    return out


def innovation(p,n):
    x=p.astype(float)-p.mean(1,keepdims=True)
    gram=x.transpose(0,2,1)@x/50
    g=gram+np.eye(3)*(.0006+.035*gram.diagonal(axis1=1,axis2=2))[:,None,:]
    return (n-x@np.linalg.solve(g,x.transpose(0,2,1)@n/50)).astype('float32')


def weights(hist):
    g,b=(np.mean(v,0) for v in zip(*hist))
    reg=1.4*np.diag(g.diagonal().clip(.0002,None))
    daily=np.stack([np.linalg.solve(a+reg,c) for a,c in hist])
    return (.68*(.5*np.linalg.solve(g+reg,b)+.5*np.median(daily,0))*np.abs(np.sign(daily).mean(0))).clip(-.17,.24)


def main():
    OUT.mkdir(exist_ok=True)
    hist=[];report={};valid=np.ones(3,bool);geometry=graph()
    np.savez(OUT/'graph.npz',left=geometry[0],right=geometry[1],inverse=geometry[2])
    for end,stop in ((12,14),(14,16),(16,20)):
        models=fit(end,geometry)
        if end>=14:
            coef=weights(hist)*valid
            if end==16:np.save(OUT/'coef.npy',coef)
        rows=[];daygain=[]
        for d in range(end,stop):
            raw=rawday(d);n=predict(raw,models,geometry)
            if d==end:np.testing.assert_allclose(n[:111],predict(raw[:111],models,geometry),atol=3e-6)
            np.save(OUT/f'side_{d}.npy',n)
            base='dynamic_factor_return_net_v1' if d>=14 else 'intraday_relative_calibration_v1'
            p=np.load(ART/base/f'pred_{d}.npy');z=innovation(p,n)
            if end>=14:
                q=(p+z@coef).astype('float32')
                np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d)
            if end<16:
                a=p.reshape(-1,3).astype(float);a-=a.mean(0);b=y.astype(float);b-=b.mean(0)
                slope=((a*b).sum(0)/(a*a).sum(0)).clip(.4,2.5)
                X=z.reshape(-1,3).astype(float)
                hist.append((X.T@X/len(X),X.T@(b/slope-a)/len(X)))
            if end>=14:
                row=(p.reshape(-1,3),q.reshape(-1,3),y);rows.append(row)
                daygain.append(corr(row[1],y)-corr(row[0],y))
        if end>=14:
            report[str(stop)]=measure(rows);print(stop,report[str(stop)],flush=True)
            if end==14:
                valid=np.min(daygain,axis=0)>0
                report['eligible']=valid.tolist()
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            if not valid.any():break

if __name__=='__main__':main()

