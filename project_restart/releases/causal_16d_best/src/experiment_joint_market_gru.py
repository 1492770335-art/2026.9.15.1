"""Frozen paired multitask architecture experiment using earlier chronological folds."""
import argparse
import gc
import json
import time
import numpy as np
import torch
from joint_market_gru import JointMarketGRU
import experiment_y2_gru_capacity as common
import experiment_y2_generalization as norm
import experiment_y2_normalized_outer as outer
import experiment_normalized_y1 as y1run
import experiment_normalized_y0 as y0run
import experiment_normalized_y2_huber as components
import experiment_mixed_market_routing as routing
import reproduce_restart_k16 as repro

OUT=outer.ROOT/'artifacts/project_restart_generalization/joint_market_gru'
VARIANTS=('independent','shared','shared_private')

def preflight():
    torch.manual_seed(17);x=torch.randn(5,16,300);target=torch.randn(5,2)
    records={}
    for variant in VARIANTS:
        torch.manual_seed(42);model=JointMarketGRU(variant)
        p=model(x)
        if p.shape!=(5,2) or not torch.isfinite(p).all():raise RuntimeError('Invalid model output')
        gradients=[]
        for j in range(2):
            g=torch.autograd.grad((p[:,j]-target[:,j]).square().mean(),
                model.shared.weight_ih_l0,retain_graph=True,allow_unused=True)[0]
            gradients.append(0. if g is None else float(g.norm()))
        if gradients[0]<=0 or (variant!='independent' and gradients[1]<=0):raise RuntimeError('Missing shared gradient')
        if variant=='independent' and gradients[1]!=0:raise RuntimeError('Control shares gradients')
        records[variant]={'parameters':sum(p.numel() for p in model.parameters()),'shared_gradient_norms':gradients}
    return records

def predict(src,model,X,tid,indices=None):
    model.eval();outputs=[]
    if indices is None:indices=np.arange(len(X))
    with torch.no_grad():
        for start in range(0,len(indices),4096):
            I=indices[start:start+4096]
            x=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            outputs.append(model(x).cpu().numpy())
    return np.concatenate(outputs)

def train(src,X,y,tid,variant,seed,path):
    torch.manual_seed(seed);model=JointMarketGRU(variant).to('cuda:0')
    if path.exists():
        model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));return model
    Y=torch.from_numpy(y[:,1:3]).to('cuda:0')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    weights=torch.tensor([.375,.625],device='cuda:0')
    first=0;losses=[];resume=path.with_suffix('.resume.pth')
    if resume.exists():
        saved=torch.load(resume,map_location='cuda:0',weights_only=True)
        if saved['variant']!=variant or saved['seed']!=seed:raise RuntimeError('Resume mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        first=saved['next_epoch'];losses=saved['losses']
    batches=(len(X)+1023)//1024;started=time.perf_counter()
    for epoch in range(first,2):
        model.train();total=np.zeros(2)
        order=np.random.default_rng(seed+epoch).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            x=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            mse=(model(x)-Y[I]).square().mean(0);loss=(mse*weights).sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=mse.detach().cpu().numpy()*len(I)
            if batch==1 or batch%500==0 or batch==batches:
                print(json.dumps({'stage':'training','model':path.name,'epoch':epoch+1,'epochs':2,
                    'batch':batch,'batches':batches,'mse_y1_y2':(total/min(batch*1024,len(X))).tolist(),
                    'seconds':round(time.perf_counter()-started,1)}),flush=True)
        losses.append((total/len(X)).tolist())
        tmp=resume.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'next_epoch':epoch+1,
                    'variant':variant,'seed':seed,'losses':losses},tmp);tmp.replace(resume)
    tmp=path.with_suffix('.pending.pth');torch.save(model.state_dict(),tmp);tmp.replace(path)
    common.save_json(path.with_suffix('.json'),{'variant':variant,'seed':seed,'losses':losses,
        'seconds':time.perf_counter()-started,'sha256':common.sha256(path)})
    return model

def two_metrics(pred,truth):
    cs=[norm.corr(pred[:,j],truth[:,j+1]) for j in range(2)]
    return {'corr_y1_y2':cs,'score_contribution':float(np.dot(cs,[.3,.5]))}

def compare(base,candidate,y,days):
    result={'baseline':two_metrics(base,y),'candidate':two_metrics(candidate,y),'daily':{}}
    result['score_delta']=result['candidate']['score_contribution']-result['baseline']['score_contribution']
    result['y2_delta']=result['candidate']['corr_y1_y2'][1]-result['baseline']['corr_y1_y2'][1]
    start=0
    for day in days:
        n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50;sl=slice(start,start+n)
        b=two_metrics(base[sl],y[sl]);c=two_metrics(candidate[sl],y[sl])
        result['daily'][str(day)]={'baseline':b,'candidate':c,'score_delta':c['score_contribution']-b['score_contribution']}
        start+=n
    if start!=len(y):raise RuntimeError('Day alignment mismatch')
    return result

def pilot(src):
    results={'variants':{v:{} for v in VARIANTS},'preflight':preflight()}
    common.save_json(OUT/'preflight.json',results['preflight'])
    for fold,spec in common.FOLDS.items():
        print(json.dumps({'stage':'load_training','fold':fold}),flush=True)
        X,y,tid=norm.load_days(spec['train'],normalized=True);X=repro.LazyMarketArray(X.astype(np.float32))
        probe=np.flatnonzero(tid%16==0);training={}
        for variant in VARIANTS:
            model=train(src,X,y,tid,variant,42,OUT/f'{variant}_{fold}_s42.pth')
            p=predict(src,model,X,tid,probe)
            training[variant]={'sampled_rows':len(probe),'head_metrics':two_metrics(p,y[probe])}
            del model,p;gc.collect();torch.cuda.empty_cache()
        del X,y,tid,probe;gc.collect()
        X,y,tid=norm.load_days(spec['validation'],normalized=True);X=repro.LazyMarketArray(X.astype(np.float32))
        old1,p128,p64=components.early_components(fold,spec['validation'])
        raw1=np.concatenate([np.load(y1run.OOF/f'day_{d:02d}_oof.npy')[:,0] for d in spec['validation']])
        base=np.stack([old1,routing.routed(old1,.5*(p128+p64))],1)
        for variant in VARIANTS:
            model=JointMarketGRU(variant).to('cuda:0')
            model.load_state_dict(torch.load(OUT/f'{variant}_{fold}_s42.pth',map_location='cuda:0',weights_only=True))
            p=predict(src,model,X,tid);np.save(OUT/f'{variant}_{fold}_prediction.npy',p)
            new1=.5*(raw1+p[:,0]);new2=routing.routed(new1,.5*(p[:,1]+p64))
            record=compare(base,np.stack([new1,new2],1),y,spec['validation'])
            record['training_probe']=training[variant];record['validation_heads']=two_metrics(p,y)
            record['head_generalization_gap']=(np.array(training[variant]['head_metrics']['corr_y1_y2'])-
                                               np.array(record['validation_heads']['corr_y1_y2'])).tolist()
            results['variants'][variant][fold]=record
            common.save_json(OUT/'pilot_partial.json',results)
            print(json.dumps({'stage':'fold_complete','fold':fold,'variant':variant,**record}),flush=True)
            del model,p,new1,new2;gc.collect();torch.cuda.empty_cache()
        del X,y,tid,old1,p128,p64,raw1,base;gc.collect()
    eligible=[];results['eligibility']={}
    for variant,folds in results['variants'].items():
        gains=[r['score_delta'] for r in folds.values()]
        positive=sum(d['score_delta']>0 for r in folds.values() for d in r['daily'].values())
        passed=min(gains)>0 and all(r['y2_delta']>0 for r in folds.values()) and positive>=6
        results['eligibility'][variant]={'passed':passed,'daily_positive':positive,'minimum_score_gain':min(gains)}
        if passed:eligible.append((min(gains),variant))
    results['selected']=max(eligible)[1] if eligible else None
    for variant in ('shared','shared_private'):
        results['eligibility'][variant]['score_gain_vs_independent']={f:results['variants'][variant][f]['score_delta']-
            results['variants']['independent'][f]['score_delta'] for f in common.FOLDS}
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete','selected':results['selected'],'eligibility':results['eligibility']}),flush=True)

def full(src):
    variant=json.loads((OUT/'pilot_results.json').read_text())['selected']
    if variant is None:raise RuntimeError('No early candidate passed')
    X,y,tid=norm.load_days(list(range(16)),normalized=True);X=repro.LazyMarketArray(X.astype(np.float32))
    for seed in (42,43,44):
        model=train(src,X,y,tid,variant,seed,OUT/f'{variant}_outer_s{seed}.pth')
        del model;gc.collect();torch.cuda.empty_cache()
    del X,y,tid;gc.collect()
    raw,y,tid=src.load_block([16,17,18,19]);mu,sd=norm.global_stats();parts=[];start=0
    for day in range(16,20):
        n=len(np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r'))*50
        parts.append(outer.transform(raw[start:start+n],mu,sd));start+=n
    X=repro.LazyMarketArray(np.concatenate(parts).astype(np.float32));del raw,parts
    predictions=[]
    for seed in (42,43,44):
        model=JointMarketGRU(variant).to('cuda:0')
        model.load_state_dict(torch.load(OUT/f'{variant}_outer_s{seed}.pth',map_location='cuda:0',weights_only=True))
        predictions.append(predict(src,model,X,tid));del model
    p=np.stack(predictions).mean(0);np.save(OUT/'outer_prediction.npy',p)
    z=np.load(outer.OUT/'validation_components.npz')
    if not np.array_equal(y,z['target']):raise RuntimeError('Alignment mismatch')
    p0=.5*(z['y0']+np.load(y0run.OUT/'outer_prediction.npy'))
    old1=.5*(z['y1']+np.load(y1run.OUT/'outer_prediction.npy'))
    base=np.stack([p0,old1,routing.routed(old1,.5*(z['norm128']+z['norm64']))],1)
    new1=.5*(z['y1']+p[:,0]);candidate=np.stack([p0,new1,routing.routed(new1,.5*(p[:,1]+z['norm64']))],1)
    result={'selected':variant,'baseline':outer.metrics(base,y),'candidate':outer.metrics(candidate,y),'daily':{}}
    start=0
    for day,n in zip(range(16,20),z['day_rows']):
        sl=slice(start,start+n)
        result['daily'][str(day)]={'baseline':outer.metrics(base[sl],y[sl]),'candidate':outer.metrics(candidate[sl],y[sl])};start+=n
    common.save_json(OUT/'outer_results.json',result);print(json.dumps({'stage':'outer_complete',**result}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','outer'],required=True)
    args=parser.parse_args()
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data SHA mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'variants':list(VARIANTS),'input':'Causal normalized 100 + contemporaneous stock mean100/std100',
          'shared_hidden':128,'private_hidden':64,'window':16,'epochs':2,'lr':1e-4,'weight_decay':1e-4,
          'batch':1024,'grad_clip':1.,'loss':'0.375*MSE(y1)+0.625*MSE(y2)',
          'initialization':'from scratch; shared and independent use matched backbone and head initialization',
          'seed':42,'order':'seed+epoch','probe':'all stocks every16ticks within training dates, diagnostic only',
          'deployment':'replace normalized y1 and normalized y2-128 branches with paired model heads; raw y1/norm64/y0 and market routing .75 fixed',
          'gate':'positive combined score and y2 gains in both early folds; >=6/8 daily combined gains; choose largest minimum-fold score gain',
          'outer_seeds':[42,43,44],'outer_train':list(range(16)),'outer_validation':[16,17,18,19],
          'data_sha256':common.EXPECTED_SHA,'baseline':'mixed_market_routing/commit_market_routed 0.3233',
          'note':'Early screening only; late dates repeatedly exposed and only used after selection. No claim of reaching 0.39 from training metrics.'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);src,_=outer.prepare()
    if args.stage=='pilot':pilot(src)
    else:full(src)

if __name__=='__main__':main()
