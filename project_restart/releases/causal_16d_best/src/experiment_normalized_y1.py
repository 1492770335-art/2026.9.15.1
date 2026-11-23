"""Normalized MarketGRU y1 diversity, with a frozen half/half raw-normalized blend."""
import argparse
import gc
import json
import numpy as np
import torch
from torch.nn import functional as F

import experiment_y2_normalized_outer as outer
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import reproduce_restart_k16 as repro

ROOT=outer.ROOT
OUT=ROOT/'artifacts/project_restart_generalization/normalized_y1'
OOF=ROOT/'artifacts/project_restart_generalization/y2_oof_corrector'
ROUTING=ROOT/'artifacts/project_restart_generalization/normalized_market_routing'

def train(src,market,X,y,tid,seed,path):
    torch.manual_seed(seed);model=market.GRU300(128).to('cuda:0')
    if path.exists():
        model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));return model
    Y=torch.from_numpy(y[:,1:2]).to('cuda:0')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    batches=(len(X)+1023)//1024
    for epoch in range(2):
        model.train();total=0.0;order=np.random.default_rng(seed+epoch).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            inp=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            loss=F.mse_loss(model(inp),Y[I]);opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=float(loss.detach())*len(I)
            if batch==1 or batch%1000==0 or batch==batches:
                print(json.dumps({'stage':'train','model':path.name,'epoch':epoch+1,'batch':batch,
                                  'batches':batches,'loss':total/min(batch*1024,len(X))}),flush=True)
    torch.save(model.state_dict(),path);return model

def measure(old,new,own,y):
    b=[norm.corr(old,y[:,1]),norm.corr(.5*old+.5*own,y[:,2])]
    mixed=.5*(old+new)
    c=[norm.corr(mixed,y[:,1]),norm.corr(.5*mixed+.5*own,y[:,2])]
    return {'base_y1_y2':b,'candidate_y1_y2':c,
            'score_delta':.3*(c[0]-b[0])+.5*(c[1]-b[1])}

def pilot(src,market):
    results={'folds':{}}
    for fold,spec in common.FOLDS.items():
        X,y,tid=norm.load_days(spec['train'],normalized=True)
        X=repro.LazyMarketArray(X.astype(np.float32))
        model=train(src,market,X,y,tid,42,OUT/f'normmarket_{fold}_s42.pth')
        del X,y,tid;gc.collect()
        X,y,tid=norm.load_days(spec['validation'],normalized=True)
        X=repro.LazyMarketArray(X.astype(np.float32))
        p=norm.predict(src,model,X,tid)
        np.save(OUT/f'prediction_{fold}.npy',p)
        old=np.concatenate([np.load(OOF/f'day_{d:02d}_oof.npy')[:,0] for d in spec['validation']])
        n128=np.concatenate([np.load(norm.OUT/f'normalized_{fold}_day_{d:02d}_y2_128.npy') for d in spec['validation']])
        own=.5*(n128+np.load(ROUTING/f'norm64_{fold}.npy'))
        record=measure(old,p,own,y);record['daily']={};start=0
        for day in spec['validation']:
            n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50;sl=slice(start,start+n)
            record['daily'][str(day)]=measure(old[sl],p[sl],own[sl],y[sl]);start+=n
        results['folds'][fold]=record
        common.save_json(OUT/'pilot_partial.json',results)
        print(json.dumps({'stage':'fold_complete','fold':fold,**record}),flush=True)
        del X,y,tid,model,p,old,n128,own;gc.collect();torch.cuda.empty_cache()
    positive=sum(d['score_delta']>0 for r in results['folds'].values() for d in r['daily'].values())
    results['daily_positive']=positive
    results['passed']=all(r['score_delta']>0 for r in results['folds'].values()) and positive>=6
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete','passed':results['passed']}),flush=True)

def full(src,market):
    if not json.loads((OUT/'pilot_results.json').read_text())['passed']:raise RuntimeError('Pilot did not pass')
    X,y,tid=norm.load_days(list(range(16)),normalized=True)
    X=repro.LazyMarketArray(X.astype(np.float32))
    for seed in (42,43,44):
        model=train(src,market,X,y,tid,seed,OUT/f'normmarket_outer_s{seed}.pth')
        del model;gc.collect();torch.cuda.empty_cache()
    del X,y,tid;gc.collect()
    raw,y,tid=src.load_block([16,17,18,19]);mu,sd=norm.global_stats();parts=[];start=0
    for day in (16,17,18,19):
        n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50
        parts.append(outer.transform(raw[start:start+n],mu,sd));start+=n
    X=repro.LazyMarketArray(np.concatenate(parts).astype(np.float32));del raw,parts
    predictions=[]
    for seed in (42,43,44):
        model=market.GRU300(128).to('cuda:0')
        model.load_state_dict(torch.load(OUT/f'normmarket_outer_s{seed}.pth',map_location='cuda:0',weights_only=True))
        predictions.append(norm.predict(src,model,X,tid));del model
    p=np.stack(predictions).mean(0);np.save(OUT/'outer_prediction.npy',p)
    z=np.load(outer.OUT/'validation_components.npz')
    if not np.array_equal(z['target'],y):raise RuntimeError('Alignment mismatch')
    mixed=.5*(z['y1']+p);own=.5*(z['norm128']+z['norm64'])
    base=np.stack([z['y0'],z['y1'],.5*z['y1']+.5*own],1)
    candidate=np.stack([z['y0'],mixed,.5*mixed+.5*own],1)
    result={'baseline':outer.metrics(base,y),'candidate':outer.metrics(candidate,y),'daily':{}}
    start=0
    for day,n in zip(range(16,20),z['day_rows']):
        sl=slice(start,start+n)
        result['daily'][str(day)]={'baseline':outer.metrics(base[sl],y[sl]),'candidate':outer.metrics(candidate[sl],y[sl])}
        start+=n
    common.save_json(OUT/'outer_results.json',result)
    print(json.dumps({'stage':'outer_complete',**result}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','outer'],required=True)
    args=parser.parse_args()
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data SHA mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'model':'MarketGRU128 K16 on causal normalized f0..82 and current mean/std of all normalized features',
          'epochs':2,'lr':1e-4,'weight_decay':1e-4,'batch':1024,'pilot_seed':42,'outer_seeds':[42,43,44],
          'blend':'newy1=0.5*rawy1+0.5*normy1; y2=0.5*newy1+0.25*norm128+0.25*norm64',
          'gate':'positive total score contribution delta on both early folds and >=6/8 days',
          'outer_train':list(range(16)),'outer_validation':[16,17,18,19],
          'note':'Freeze architecture and blend before evaluation; only early-passing candidate checks late dates'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);src,market=outer.prepare()
    if args.stage=='pilot':pilot(src,market)
    else:full(src,market)

if __name__=='__main__':main()
