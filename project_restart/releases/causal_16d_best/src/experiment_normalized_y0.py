"""Causal normalized MarketGRU y0, frozen half raw/half normalized blend."""
import argparse
import gc
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
import experiment_y2_normalized_outer as outer
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_normalized_y1 as y1run
import reproduce_restart_k16 as repro

OUT=outer.ROOT/'artifacts/project_restart_generalization/normalized_y0'

def train(src,market,X,y,tid,seed,epochs,path):
    torch.manual_seed(seed)
    model=market.GRU300(128).to('cuda:0')
    if path.exists():
        model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True))
        return model
    Y=torch.from_numpy(y[:,0:1]).to('cuda:0')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    batches=(len(X)+1023)//1024
    started=time.perf_counter();losses=[];first_epoch=0
    resume=path.with_suffix('.resume.pth')
    if resume.exists():
        saved=torch.load(resume,map_location='cuda:0',weights_only=True)
        if saved['seed']!=seed or saved['epochs']!=epochs:raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        first_epoch=saved['next_epoch'];losses=saved['losses']
        print(json.dumps({'stage':'resumed','model':path.name,'next_epoch':first_epoch+1}),flush=True)
    for epoch in range(first_epoch,epochs):
        model.train();total=0.0
        order=np.random.default_rng(seed+epoch).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            inp=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            loss=F.mse_loss(model(inp),Y[I])
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=float(loss.detach())*len(I)
            if batch==1 or batch%500==0 or batch==batches:
                print(json.dumps({'stage':'training','model':path.name,'epoch':epoch+1,
                    'epochs':epochs,'batch':batch,'batches':batches,
                    'loss':total/min(batch*1024,len(X)),
                    'elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
        losses.append(total/len(X))
        temporary=resume.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                    'next_epoch':epoch+1,'losses':losses,'seed':seed,'epochs':epochs},temporary)
        temporary.replace(resume)
    temporary=path.with_suffix('.pending.pth')
    torch.save(model.state_dict(),temporary);temporary.replace(path)
    common.save_json(path.with_suffix('.json'),{'losses':losses,'seconds':time.perf_counter()-started,
        'seed':seed,'epochs':epochs,'sha256':common.sha256(path)})
    return model

def measure(raw,new,y):
    b=norm.corr(raw,y[:,0]);n=norm.corr(new,y[:,0]);c=norm.corr(.5*(raw+new),y[:,0])
    return {'raw_corr0':b,'normalized_corr0':n,'candidate_corr0':c,
            'delta_corr0':c-b,'score_delta':.2*(c-b)}

def pilot(src,market):
    results={'folds':{}}
    for fold,spec in common.FOLDS.items():
        X,y,tid=norm.load_days(spec['train'],normalized=True)
        X=repro.LazyMarketArray(X.astype(np.float32))
        model=train(src,market,X,y,tid,42,6 if fold=='A' else 4,OUT/f'normmarket_{fold}_s42_y0.pth')
        del X,y,tid;gc.collect()
        X,y,tid=norm.load_days(spec['validation'],normalized=True)
        X=repro.LazyMarketArray(X.astype(np.float32))
        p=norm.predict(src,model,X,tid);np.save(OUT/f'normalized_prediction_{fold}.npy',p)
        del X,model;gc.collect();torch.cuda.empty_cache()
        raw,truth,rawtid=src.load_block(spec['validation'])
        if not np.array_equal(truth,y) or not np.array_equal(rawtid,tid):raise RuntimeError('Alignment mismatch')
        model=market.GRU300(128).to('cuda:0')
        checkpoint=outer.REPO/f'project_restart/experiments/EXP-0019/ckpt_mkt_{fold}/Model_0.pth'
        model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True))
        raw=repro.LazyMarketArray(raw)
        old=norm.predict(src,model,raw,tid);np.save(OUT/f'raw_prediction_{fold}.npy',old)
        result=measure(old,p,y);result['daily']={};start=0
        for day in spec['validation']:
            n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50
            sl=slice(start,start+n)
            result['daily'][str(day)]=measure(old[sl],p[sl],y[sl]);start+=n
        if start!=len(p):raise RuntimeError('Day alignment mismatch')
        results['folds'][fold]=result
        common.save_json(OUT/'pilot_partial.json',results)
        print(json.dumps({'stage':'fold_complete','fold':fold,**result}),flush=True)
        del raw,y,tid,truth,rawtid,model,p,old;gc.collect();torch.cuda.empty_cache()
    positive=sum(r['delta_corr0']>0 for f in results['folds'].values() for r in f['daily'].values())
    results['daily_positive']=positive
    results['passed']=all(f['delta_corr0']>0 for f in results['folds'].values()) and positive>=6
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete','passed':results['passed'],'daily_positive':positive}),flush=True)

def full(src,market):
    if not json.loads((OUT/'pilot_results.json').read_text())['passed']:raise RuntimeError('Pilot did not pass')
    X,y,tid=norm.load_days(list(range(16)),normalized=True)
    X=repro.LazyMarketArray(X.astype(np.float32))
    for seed in (42,43,44):
        model=train(src,market,X,y,tid,seed,4,OUT/f'normmarket_outer_s{seed}_y0.pth')
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
        model.load_state_dict(torch.load(OUT/f'normmarket_outer_s{seed}_y0.pth',map_location='cuda:0',weights_only=True))
        predictions.append(norm.predict(src,model,X,tid));del model
    p=np.stack(predictions).mean(0);np.save(OUT/'outer_prediction.npy',p)
    z=np.load(outer.OUT/'validation_components.npz')
    if not np.array_equal(y,z['target']):raise RuntimeError('Alignment mismatch')
    p1=.5*(z['y1']+np.load(y1run.OUT/'outer_prediction.npy'))
    p2=.5*p1+.25*(z['norm128']+z['norm64'])
    base=np.stack([z['y0'],p1,p2],1);candidate=base.copy();candidate[:,0]=.5*(z['y0']+p)
    result={'baseline':outer.metrics(base,y),'candidate':outer.metrics(candidate,y),
            'normalized_only_corr0':norm.corr(p,y[:,0]),'daily':{}}
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
    plan={'folds':common.FOLDS,'model':'MarketGRU128 K16, causal EWMA64 normalized f0..82',
        'target':0,'epochs':{'A':6,'B':4,'outer':4},'epochs_source':'Frozen EXP-0019 early-fold recipe, frozen full16 baseline recipe',
        'lr':1e-4,'weight_decay':1e-4,'batch':1024,'pilot_seed':42,'outer_seeds':[42,43,44],
        'blend':'0.5 raw y0 + 0.5 normalized y0; y1 and y2 from current 0.3214 package',
        'gate':'positive pooled corr0 delta in both early folds and at least 6/8 daily gains',
        'outer_train':list(range(16)),'outer_validation':[16,17,18,19],
        'data_sha256':common.EXPECTED_SHA,'note':'Late dates repeatedly exposed; evaluate only after early-fold gate'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);src,market=outer.prepare()
    if args.stage=='pilot':pilot(src,market)
    else:full(src,market)

if __name__=='__main__':main()
