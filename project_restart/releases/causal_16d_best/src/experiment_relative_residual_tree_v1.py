"""Learn nonlinear corrections to stock-relative forecasts using earlier out-of-fold days."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps/tree_models'))
import lightgbm as lgb
import experiment_endpoint_innovation_v1 as innov
from experiment_causal_return_basket_v1 import moving_sum
prior=innov.prior
OUT=prior.ART/'relative_residual_tree_v1'
BOOK=[5,7,8,10,11,13,14,16,20,22,23,24,25,26,29,31,34,36,37,38,39,40,41,42,46,50,51,54,55,60,64,67,72,75,77,79,82,84,87,90,92,94,97,99]

def previous(d):
    if d<12:return innov.forecast(d)
    return np.load(innov.OUT/f'pred_{d}.npy').astype(float)

def matrix(raw,p,t):
    r=raw[:,:,0].astype(float)-.00136; seq=np.arange(len(r));s=np.arange(50)
    price=r.cumsum(0); last=np.maximum.accumulate(np.where(abs(r)>.018,seq[:,None],0),axis=0)
    age=seq[:,None]-last;state=np.arcsinh(raw[t][:,:,BOOK].astype(np.float32))
    x=[state,p[t].astype(np.float32)]
    scale=np.sqrt((8+moving_sum(r*r,96))/(8+np.minimum(seq+1,96))[:,None]).clip(.45,3.)
    scale=(.4+.6*scale).astype(np.float32)
    def add(v):
        if v.ndim==2:v=v[:,:,None]
        x.append(np.broadcast_to(v,(len(t),50,v.shape[-1])).astype(np.float32))
    add(np.log(scale[t]));add(np.log1p(age[t]));add(np.log1p(t[:,None]))
    add(price.mean(1)[t,None]-price.mean(1)[last[t]])
    add(p[t]-p[t].mean(1,keepdims=True));add(p[t]/scale[t,:,None])
    add(state[:,:,[4,8,14,21,22,26,29]]-state[:,:,[4,8,14,21,22,26,29]].mean(1,keepdims=True))
    for h in (3,16,60,240):
        i=np.maximum(0,t-h);u=(price[t]-price[i])/np.sqrt(h)
        add(u);add(u-u.mean(1,keepdims=True));add((p[t]-p[i])/scale[t,:,None])
        add(np.arcsinh(raw[i][:,:,[11,16,26,38,41,42,64,94]].astype(np.float32)))
    return np.concatenate(x,axis=-1).reshape(len(t)*50,-1),scale[t]

def train(end):
    xs=[];ys=[]
    for d in range(8,end):
        raw=innov.grid(d);p=previous(d);y=prior.labels(d).reshape(p.shape).astype(float)
        t=np.arange((d*3)%5,len(p),5);x,s=matrix(raw,p,t)
        residual=y[t]-p[t];residual-=residual.mean(1,keepdims=True)
        xs.append(x);ys.append((residual/s[:,:,None]).reshape(-1,3).astype(np.float32))
    x=np.concatenate(xs);y=np.concatenate(ys);del xs,ys
    models=[]
    for j in range(3):
        data=lgb.Dataset(x,label=y[:,j].clip(-4,4),params={'max_bin':63})
        model=lgb.train(dict(objective='huber',alpha=1.3,learning_rate=.04,num_leaves=15,
            min_data_in_leaf=420,lambda_l2=22,feature_fraction=.72,bagging_fraction=.85,
            bagging_freq=3,max_bin=63,verbosity=-1,num_threads=4,seed=910+j,force_col_wise=True),data,num_boost_round=150)
        if end==16:model.save_model(str(OUT/f'target_{j}.txt'))
        models.append(model)
    return models

def sides(raw,p,models,t):
    x,s=matrix(raw,p,t)
    q=np.stack([m.predict(x,num_threads=4) for m in models],axis=-1).reshape(len(t),50,3)*s[:,:,None]
    return q-q.mean(1,keepdims=True)

def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        models=train(end);rows=[]
        for d in range(end,stop):
            p=previous(d);raw=innov.grid(d);t=np.arange(len(p));r=np.empty_like(p)
            for k in range(5):r[k::5]=sides(raw,p,models,t[k::5])
            if d==end:
                np.testing.assert_allclose(r[:62],sides(raw[:62],p[:62],models,t[:62]),atol=1e-7)
            rows.append((p,prior.labels(d).reshape(p.shape),np.zeros((len(p),3)),r))
        if end<16:early.extend(rows);w=innov.strength(early)[1]
        entries=[]
        for d,(p,y,_,r) in zip(range(end,stop),rows):
            q=p+w*r;np.save(OUT/f'pred_{d}.npy',q.astype(np.float32));entries.append(tuple(z.reshape(-1,3) for z in (p,q,y)))
        report[stop]={'weight':w.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)
if __name__=='__main__':main()
