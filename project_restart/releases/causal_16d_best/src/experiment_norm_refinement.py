"""Earlier-fold selection of level-preserving inputs or averaged fine tuning."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import experiment_y2_normalized_outer as outer
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common

ROOT=outer.ROOT
OUT=ROOT/'artifacts/project_restart_generalization/norm_refinement'
OOF=ROOT/'artifacts/project_restart_generalization/y2_oof_corrector'
ROUTING=ROOT/'artifacts/project_restart_generalization/normalized_market_routing'

def inputs(days,mode):
    X,y,tid=norm.load_days(days,normalized=True)
    if mode=='anchor':
        mu,sd=norm.global_stats();start=0
        for day in days:
            raw=np.load(norm.CACHE/f'day_{day:02d}_x.npy',mmap_mode='r').reshape(-1,100)
            for offset in range(0,len(raw),12800):
                stop=min(offset+12800,len(raw));sl=slice(start+offset,start+stop)
                global_z=np.clip((raw[offset:stop,:83].astype(np.float32)-mu)/sd,-6,6)
                X[sl,:83]=.5*(X[sl,:83].astype(np.float32)+global_z)
            start+=len(raw)
    return X,y,tid

def train(src,X,y,tid,mode,seed,source,checkpoint):
    if checkpoint.exists():
        model=src.GRU(128).to('cuda:0')
        model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True))
        return model
    torch.manual_seed(seed)
    model=src.GRU(128).to('cuda:0')
    if mode=='swa':
        model.load_state_dict(torch.load(source,map_location='cuda:0',weights_only=True))
    Y=torch.from_numpy(y[:,2:3]).to('cuda:0')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5 if mode=='swa' else 1e-4,weight_decay=1e-4)
    epochs=1 if mode=='swa' else 2
    batches=(len(X)+1023)//1024
    snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()}] if mode=='swa' else []
    save_at={round(batches*q/4) for q in (1,2,3,4)}
    for epoch in range(epochs):
        model.train();total=0.0
        order=np.random.default_rng(seed+epoch if mode=='anchor' else seed+9000).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            x=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            loss=F.mse_loss(model(x),Y[I]);opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=float(loss.detach())*len(I)
            if mode=='swa' and batch in save_at:
                snapshots.append({k:v.detach().clone() for k,v in model.state_dict().items()})
            if batch==1 or batch%1000==0 or batch==batches:
                print(json.dumps({'stage':'training','mode':mode,'seed':seed,'model':checkpoint.name,
                                  'epoch':epoch+1,'batch':batch,'batches':batches,
                                  'loss':total/min(batch*1024,len(X))}),flush=True)
    if mode=='swa':
        torch.save(model.state_dict(),checkpoint.with_name(checkpoint.stem+'_last.pth'))
        averaged={k:torch.stack([s[k] for s in snapshots]).mean(0) for k in snapshots[0]}
        model.load_state_dict(averaged)
    torch.save(model.state_dict(),checkpoint)
    del Y,opt,snapshots
    return model

def metrics(pred,base,y,days):
    result={'base':norm.corr(base,y),'candidate':norm.corr(pred,y),'daily':{}}
    result['delta']=result['candidate']-result['base'];start=0
    for day in days:
        n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50
        b=norm.corr(base[start:start+n],y[start:start+n]);c=norm.corr(pred[start:start+n],y[start:start+n])
        result['daily'][str(day)]={'base':b,'candidate':c,'delta':c-b};start+=n
    return result

def pilot(src):
    results={'modes':{}}
    for mode in ('anchor','swa'):
        results['modes'][mode]={}
        for fold,spec in common.FOLDS.items():
            X,y,tid=inputs(spec['train'],mode)
            source=norm.OUT/f'normalized_{fold}_s42_y2_128.pth'
            ckpt=OUT/f'{mode}_{fold}_s42.pth'
            model=train(src,X,y,tid,mode,42,source,ckpt)
            del X,y,tid;gc.collect()
            Xv,yv,tv=inputs(spec['validation'],mode)
            p=norm.predict(src,model,Xv,tv)
            np.save(OUT/f'{mode}_{fold}_prediction.npy',p)
            p1=np.concatenate([np.load(OOF/f'day_{d:02d}_oof.npy')[:,0] for d in spec['validation']])
            p128=np.concatenate([np.load(norm.OUT/f'normalized_{fold}_day_{d:02d}_y2_128.npy')
                                 for d in spec['validation']])
            p64=np.load(ROUTING/f'norm64_{fold}.npy')
            base=.5*p1+.25*(p128+p64);candidate=.5*p1+.25*(p+p64)
            result=metrics(candidate,base,yv[:,2],spec['validation'])
            result['checkpoint_sha256']=common.sha256(ckpt)
            if mode=='swa':
                model.load_state_dict(torch.load(ckpt.with_name(ckpt.stem+'_last.pth'),map_location='cuda:0',weights_only=True))
                final=norm.predict(src,model,Xv,tv)
                result['unaveraged_continuation']=metrics(.5*p1+.25*(final+p64),base,yv[:,2],spec['validation'])
            results['modes'][mode][fold]=result
            common.save_json(OUT/'pilot_partial.json',results)
            print(json.dumps({'stage':'fold_complete','mode':mode,'fold':fold,**result}),flush=True)
            del Xv,yv,tv,model,p,p1,p128,p64,base,candidate
            gc.collect();torch.cuda.empty_cache()
    eligible=[]
    for mode,folds in results['modes'].items():
        count=sum(d['delta']>0 for r in folds.values() for d in r['daily'].values())
        gains=[r['delta'] for r in folds.values()]
        if min(gains)>0 and count>=6:eligible.append((min(gains),mode))
    results['selected']=max(eligible)[1] if eligible else None
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete','selected':results['selected']}),flush=True)

def full(src):
    pilot_result=json.loads((OUT/'pilot_results.json').read_text())
    mode=pilot_result['selected']
    if mode is None:raise RuntimeError('No candidate passed early screening')
    X,y,tid=inputs(list(range(16)),mode)
    for seed in (42,43,44):
        ckpt=OUT/f'{mode}_outer_s{seed}.pth'
        model=train(src,X,y,tid,mode,seed,outer.OUT/f'norm_h128_s{seed}_y2.pth',ckpt)
        del model;gc.collect();torch.cuda.empty_cache()
    del X,y,tid;gc.collect()
    # Use the same causal transformer on late dates. Labels are read only now.
    raw,y,tid=src.load_block([16,17,18,19]);mu,sd=norm.global_stats();parts=[];start=0
    for day in (16,17,18,19):
        n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50
        part=outer.transform(raw[start:start+n],mu,sd)
        if mode=='anchor':
            g=np.clip((raw[start:start+n,:83].astype(np.float16).astype(np.float32)-mu)/sd,-6,6)
            part[:,:83]=.5*(part[:,:83].astype(np.float32)+g)
        parts.append(part);start+=n
    X=np.concatenate(parts);del raw,parts
    preds=[]
    for seed in (42,43,44):
        model=src.GRU(128).to('cuda:0')
        model.load_state_dict(torch.load(OUT/f'{mode}_outer_s{seed}.pth',map_location='cuda:0',weights_only=True))
        preds.append(norm.predict(src,model,X,tid));del model
    pred=np.stack(preds).mean(0)
    np.save(OUT/'outer_member_prediction.npy',pred)
    z=np.load(outer.OUT/'validation_components.npz')
    if not np.array_equal(z['target'],y):raise RuntimeError('Late label alignment mismatch')
    base=.5*z['y1']+.25*(z['norm128']+z['norm64'])
    candidate=.5*z['y1']+.25*(pred+z['norm64'])
    result={'selected':mode,'candidate':outer.metrics(np.stack([z['y0'],z['y1'],candidate],1),y),
            'baseline':outer.metrics(np.stack([z['y0'],z['y1'],base],1),y),
            'daily_y2':metrics(candidate,base,y[:,2],[16,17,18,19]),
            'note':'Late validation repeatedly exposed; model selected only using earlier folds in this experiment'}
    common.save_json(OUT/'outer_results.json',result)
    print(json.dumps({'stage':'outer_complete',**result}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','outer'],required=True)
    args=parser.parse_args()
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data SHA mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'variants':['anchor','swa'],'seed':42,'window':16,
          'anchor':'replace f0..82 by half causal normalized + half global standardized raw (global days0-3); GRU128 E2 lr1e-4',
          'swa':'normalized GRU128 E2 warm start; one extra epoch lr1e-5; average starting checkpoint and four quarter-epoch checkpoints',
          'selection':'both early pooled gains>0 and at least6/8 positive daily gains; maximize minimum fold gain',
          'outer_seeds':[42,43,44],'outer_train':list(range(16)),'outer_eval':[16,17,18,19],
          'blend':'change only normalized128 member in current 0.5*y1+0.25*norm128+0.25*norm64',
          'note':'Only early-selected candidate checked on late validation; original models retained'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);src,_=outer.prepare()
    if args.stage=='pilot':pilot(src)
    else:full(src)

if __name__=='__main__':main()
