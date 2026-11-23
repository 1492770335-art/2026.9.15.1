"""Matched MSE -> 2*Huber(delta=2) experiment for normalized y2 GRU128."""
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
import experiment_normalized_y0 as y0run
import experiment_normalized_y1 as y1run
import experiment_norm_refinement as refine

OUT=outer.ROOT/'artifacts/project_restart_generalization/normalized_y2_huber'

def train(src,X,y,tid,seed,path):
    torch.manual_seed(seed);model=src.GRU(128).to('cuda:0')
    if path.exists():
        model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));return model
    Y=torch.from_numpy(y[:,2:3]).to('cuda:0')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    first=0;losses=[];resume=path.with_suffix('.resume.pth')
    if resume.exists():
        s=torch.load(resume,map_location='cuda:0',weights_only=True)
        if s['seed']!=seed:raise RuntimeError('Resume seed mismatch')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
        first=s['next_epoch'];losses=s['losses']
    batches=(len(X)+1023)//1024;started=time.perf_counter()
    for epoch in range(first,2):
        model.train();total=0.;mse_total=0.
        order=np.random.default_rng(seed+epoch).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            x=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            p=model(x);loss=2*F.huber_loss(p,Y[I],delta=2.0)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=float(loss.detach())*len(I)
            mse_total+=float(F.mse_loss(p.detach(),Y[I]))*len(I)
            if batch==1 or batch%500==0 or batch==batches:
                print(json.dumps({'stage':'training','model':path.name,'epoch':epoch+1,'epochs':2,
                    'batch':batch,'batches':batches,'huber_loss':total/min(batch*1024,len(X)),
                    'mse':mse_total/min(batch*1024,len(X)),
                    'seconds':round(time.perf_counter()-started,1)}),flush=True)
        losses.append({'huber':total/len(X),'mse':mse_total/len(X)})
        tmp=resume.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'next_epoch':epoch+1,
                    'seed':seed,'losses':losses},tmp);tmp.replace(resume)
    tmp=path.with_suffix('.pending.pth');torch.save(model.state_dict(),tmp);tmp.replace(path)
    common.save_json(path.with_suffix('.json'),{'losses':losses,'seconds':time.perf_counter()-started,
        'seed':seed,'sha256':common.sha256(path)})
    return model

def early_components(fold,days):
    raw1=np.concatenate([np.load(refine.OOF/f'day_{d:02d}_oof.npy')[:,0] for d in days])
    p1=.5*(raw1+np.load(y1run.OUT/f'prediction_{fold}.npy'))
    p128=np.concatenate([np.load(norm.OUT/f'normalized_{fold}_day_{d:02d}_y2_128.npy') for d in days])
    p64=np.load(refine.ROUTING/f'norm64_{fold}.npy')
    return p1,p128,p64

def pilot(src):
    results={'folds':{}}
    for fold,spec in common.FOLDS.items():
        X,y,tid=norm.load_days(spec['train'],normalized=True)
        model=train(src,X,y,tid,42,OUT/f'huber_{fold}_s42.pth')
        del X,y,tid;gc.collect()
        X,y,tid=norm.load_days(spec['validation'],normalized=True)
        p=norm.predict(src,model,X,tid);np.save(OUT/f'prediction_{fold}.npy',p)
        p1,p128,p64=early_components(fold,spec['validation'])
        base=.5*p1+.25*(p128+p64);candidate=.5*p1+.25*(p+p64)
        record=refine.metrics(candidate,base,y[:,2],spec['validation'])
        record['standalone']={'mse':norm.corr(p128,y[:,2]),'huber':norm.corr(p,y[:,2])}
        results['folds'][fold]=record;common.save_json(OUT/'pilot_partial.json',results)
        print(json.dumps({'stage':'fold_complete','fold':fold,**record}),flush=True)
        del X,y,tid,model,p,p1,p128,p64,base,candidate;gc.collect();torch.cuda.empty_cache()
    count=sum(d['delta']>0 for r in results['folds'].values() for d in r['daily'].values())
    results['daily_positive']=count
    results['passed']=all(r['delta']>0 for r in results['folds'].values()) and count>=6
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete','passed':results['passed'],'daily_positive':count}),flush=True)

def full(src):
    if not json.loads((OUT/'pilot_results.json').read_text())['passed']:raise RuntimeError('Pilot did not pass')
    X,y,tid=norm.load_days(list(range(16)),normalized=True)
    for seed in (42,43,44):
        model=train(src,X,y,tid,seed,OUT/f'huber_outer_s{seed}.pth')
        del model;gc.collect();torch.cuda.empty_cache()
    del X,y,tid;gc.collect()
    raw,y,tid=src.load_block([16,17,18,19]);mu,sd=norm.global_stats();parts=[];start=0
    for day in range(16,20):
        n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50
        parts.append(outer.transform(raw[start:start+n],mu,sd));start+=n
    X=np.concatenate(parts);del raw,parts
    predictions=[]
    for seed in (42,43,44):
        model=src.GRU(128).to('cuda:0')
        model.load_state_dict(torch.load(OUT/f'huber_outer_s{seed}.pth',map_location='cuda:0',weights_only=True))
        predictions.append(norm.predict(src,model,X,tid));del model
    p=np.stack(predictions).mean(0);np.save(OUT/'outer_prediction.npy',p)
    z=np.load(outer.OUT/'validation_components.npz')
    if not np.array_equal(y,z['target']):raise RuntimeError('Alignment mismatch')
    p0=.5*(z['y0']+np.load(y0run.OUT/'outer_prediction.npy'))
    p1=.5*(z['y1']+np.load(y1run.OUT/'outer_prediction.npy'))
    base=np.stack([p0,p1,.5*p1+.25*(z['norm128']+z['norm64'])],1)
    candidate=base.copy();candidate[:,2]=.5*p1+.25*(p+z['norm64'])
    result={'baseline':outer.metrics(base,y),'candidate':outer.metrics(candidate,y),'daily':{}}
    start=0
    for day,n in zip(range(16,20),z['day_rows']):
        sl=slice(start,start+n)
        result['daily'][str(day)]={'baseline':outer.metrics(base[sl],y[sl]),'candidate':outer.metrics(candidate[sl],y[sl])};start+=n
    common.save_json(OUT/'outer_results.json',result)
    print(json.dumps({'stage':'outer_complete',**result}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','outer'],required=True)
    args=parser.parse_args()
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data SHA mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'change':'normalized y2 GRU128 from scratch using 2*Huber(delta=2)',
        'window':16,'epochs':2,'lr':1e-4,'weight_decay':1e-4,'batch':1024,'clip':1.,
        'pilot_seed':42,'outer_seeds':[42,43,44],'order':'seed+epoch, same as MSE control',
        'blend':'y2=0.5*mixed_y1+0.25*huber128+0.25*normalized_mse64; y0/y1 unchanged',
        'gate':'positive pooled y2 gain in both earlier folds and at least 6/8 daily positive',
        'outer_train':list(range(16)),'outer_validation':[16,17,18,19],
        'data_sha256':common.EXPECTED_SHA,'baseline':'normalized_y0/commit_mixed_y0_y1, score 0.3222',
        'note':'No parameter search on late dates; earlier-passing candidate only. Late dates repeatedly exposed.'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);src,_=outer.prepare()
    if args.stage=='pilot':pilot(src)
    else:full(src)

if __name__=='__main__':main()
