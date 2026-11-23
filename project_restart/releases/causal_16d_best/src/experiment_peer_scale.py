"""Causal target-scale normalization vs inverse-variance loss alone.

Same train-only graph, PeerGRU400 architecture, seeds, row order and epochs as
the 0.3263 model. Scales are computed solely from observable f000 history.
"""
import argparse
import gc
import json
import time
import numpy as np
import torch
import experiment_target_peer_gru as previous
import experiment_dynamic_peer_gru as peer
import experiment_joint_market_gru as joint
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_y2_normalized_outer as outer
import experiment_normalized_y2_huber as components
import experiment_mixed_market_routing as routing
import experiment_norm_refinement as refine

OUT=outer.ROOT/'artifacts/project_restart_generalization/peer_scale'
VARIANTS=('scaled_target','weighted_only')
ALPHA=2./65.


def reference():
    total=0.;count=0
    for day in range(4):
        x=np.load(norm.CACHE/f'day_{day:02d}_x.npy',mmap_mode='r')[::16,:,0].astype(np.float64)
        total+=float(np.square(x).sum());count+=x.size
    return float(np.sqrt(total/count))


def causal_scale(x,ref):
    if x.ndim!=2 or x.shape[1]!=50 or ref<=0:raise ValueError('Invalid scale input')
    state=np.full(50,ref*ref,dtype=np.float64);scale=np.empty(x.shape,dtype=np.float32)
    for t,row in enumerate(x):
        state+=ALPHA*(np.square(row.astype(np.float64))-state)
        scale[t]=np.clip(np.sqrt(state)/ref,.5,2.)
    return scale.reshape(-1)


def load_scale(days,ref):
    return np.concatenate([causal_scale(np.load(norm.CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')[:,:,0],ref) for d in days])


def preflight():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,50)).astype(np.float16)
    original=causal_scale(x,1.).reshape(-1,50);changed=x.copy();changed[60:]*=4
    np.testing.assert_array_equal(original[:60],causal_scale(changed,1.).reshape(-1,50)[:60])
    np.testing.assert_array_equal(original[:20],causal_scale(x[:20],1.).reshape(-1,50))
    np.testing.assert_array_equal(causal_scale(np.ones((20,50)),1.),np.ones(1000,dtype=np.float32))
    torch.manual_seed(5);p=torch.randn(100,2,requires_grad=True);y=torch.randn_like(p)
    s=torch.from_numpy(original[:2].reshape(-1,1));w=s.square().reciprocal();w/=w.mean()
    loss=((p*s-y).square()*w).mean()
    equal=(((p-y/s).square())/s.square().reciprocal().mean()).mean()
    torch.testing.assert_close(loss,equal)
    loss.backward()
    if not torch.isfinite(p.grad).all():raise RuntimeError('Invalid gradients')
    return {'future_independence':True,'prefix_independence':True,'constant_input_scale_is_one':True,
            'scaled_loss_algebra_checked':True,'gradients_finite':True}


def train(src,X,y,tid,scale,variant,seed,path):
    torch.manual_seed(seed);model=peer.PeerGRU().to('cuda:0')
    if path.exists():
        record=json.loads(path.with_suffix('.json').read_text())
        if common.sha256(path)!=record['sha256']:raise RuntimeError('Checkpoint fingerprint mismatch')
        model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));return model
    if len(scale)!=len(X):raise ValueError('Scale alignment mismatch')
    Y=torch.from_numpy(y[:,1:3]).to('cuda:0');S=torch.from_numpy(scale).to('cuda:0')
    weight_mean=float(np.mean(np.square(scale.astype(np.float64))**-1))
    W=S.square().reciprocal()/weight_mean
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    tasks=torch.tensor([.375,.625],device='cuda:0');first=0;losses=[]
    resume=path.with_suffix('.resume.pth')
    if resume.exists():
        saved=torch.load(resume,map_location='cuda:0',weights_only=True)
        if saved['variant']!=variant or saved['seed']!=seed or abs(saved['weight_mean']-weight_mean)>1e-10:raise RuntimeError('Resume mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        first=saved['next_epoch'];losses=saved['losses']
    batches=(len(X)+1023)//1024;started=time.perf_counter()
    for epoch in range(first,2):
        model.train();total=np.zeros(2);weighted=0.
        order=np.random.default_rng(seed+epoch).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            inp=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            prediction=model(inp)
            if variant=='scaled_target':prediction=prediction*S[I,None]
            elif variant!='weighted_only':raise ValueError(variant)
            error=(prediction-Y[I]).square();loss=((error*tasks).sum(-1)*W[I]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=error.detach().sum(0).cpu().numpy();weighted+=float(loss.detach())*len(I)
            if batch==1 or batch%250==0 or batch==batches:
                print(json.dumps({'stage':'training','model':path.name,'epoch':epoch+1,'batch':batch,'batches':batches,
                    'unweighted_mse':(total/min(batch*1024,len(X))).tolist(),'weighted_loss':weighted/min(batch*1024,len(X)),
                    'seconds':round(time.perf_counter()-started,1)}),flush=True)
        losses.append({'mse':(total/len(X)).tolist(),'weighted_loss':weighted/len(X)})
        tmp=resume.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
            'next_epoch':epoch+1,'variant':variant,'seed':seed,'weight_mean':weight_mean,'losses':losses},tmp);tmp.replace(resume)
    tmp=path.with_suffix('.pending.pth');torch.save(model.state_dict(),tmp);tmp.replace(path)
    common.save_json(path.with_suffix('.json'),{'variant':variant,'seed':seed,'losses':losses,'weight_mean':weight_mean,
        'seconds':time.perf_counter()-started,'sha256':common.sha256(path)})
    return model


def evaluate_prediction(pred,baseline,y,days,scale):
    result=refine.metrics(pred,baseline,y[:,2],days);result['total_score_delta']=.5*result['delta']
    # Fixed observable scale groups, used for diagnosis, never for choosing weights.
    group=np.searchsorted([.65,1.,1.5],scale);result['scale_groups']={}
    for k in range(4):
        take=group==k
        if take.sum()<3:continue
        result['scale_groups'][str(k)]={'rows':int(take.sum()),'base':norm.corr(baseline[take],y[take,2]),
            'candidate':norm.corr(pred[take],y[take,2])}
    return result


def pilot(src):
    ref=reference();results={'reference_f0_rms_days0_3':ref,'variants':{v:{} for v in VARIANTS}}
    for fold,spec in common.FOLDS.items():
        graph=np.load(previous.OUT/f'graph_{fold}.npz')['target_graph']
        raw,y,tid=norm.load_days(spec['train'],normalized=True);raw=raw.astype(np.float32)
        X=peer.PeerArray(raw,previous.context(raw,graph));scale=load_scale(spec['train'],ref)
        print(json.dumps({'stage':'training_data','fold':fold,'rows':len(X),'scale_quantiles':np.quantile(scale,[0,.25,.5,.75,1]).tolist()}),flush=True)
        for variant in VARIANTS:
            model=train(src,X,y,tid,scale,variant,42,OUT/f'{variant}_{fold}_s42.pth')
            del model;gc.collect();torch.cuda.empty_cache()
        del raw,X,y,tid,scale;gc.collect()
        raw,y,tid=norm.load_days(spec['validation'],normalized=True);raw=raw.astype(np.float32)
        X=peer.PeerArray(raw,previous.context(raw,graph));scale=load_scale(spec['validation'],ref)
        p1,_,p64=components.early_components(fold,spec['validation'])
        old=np.load(previous.OUT/f'target_graph_{fold}_prediction.npy')[:,1]
        baseline=routing.routed(p1,.5*(old+p64))
        for variant in VARIANTS:
            model=peer.PeerGRU().to('cuda:0');model.load_state_dict(torch.load(OUT/f'{variant}_{fold}_s42.pth',map_location='cuda:0',weights_only=True))
            p=joint.predict(src,model,X,tid)
            if variant=='scaled_target':p*=scale[:,None]
            np.save(OUT/f'{variant}_{fold}_prediction.npy',p)
            candidate=routing.routed(p1,.5*(p[:,1]+p64));rec=evaluate_prediction(candidate,baseline,y,spec['validation'],scale)
            rec['direct_y2_corr']=norm.corr(p[:,1],y[:,2]);rec['baseline_direct_y2_corr']=norm.corr(old,y[:,2])
            results['variants'][variant][fold]=rec;common.save_json(OUT/'pilot_partial.json',results)
            print(json.dumps({'stage':'fold_complete','fold':fold,'variant':variant,**rec}),flush=True)
            del model,p,candidate;gc.collect();torch.cuda.empty_cache()
        del raw,X,y,tid,scale,p1,p64,old,baseline;gc.collect()
        if fold=='A' and all(results['variants'][v]['A']['delta']<.001 for v in VARIANTS):
            results['selected']=None;results['stopped_after']='A';results['reason']='Neither method meets the necessary first-fold gain'
            common.save_json(OUT/'pilot_results.json',results);print(json.dumps({'stage':'pilot_stopped_early','selected':None}),flush=True);return
    results['eligibility']={};eligible=[]
    for variant,folds in results['variants'].items():
        count=sum(d['delta']>0 for r in folds.values() for d in r['daily'].values())
        minimum=min(r['delta'] for r in folds.values());passed=minimum>=.001 and count>=6
        if variant=='scaled_target':
            gains={f:folds[f]['candidate']-results['variants']['weighted_only'][f]['candidate'] for f in common.FOLDS}
            passed=passed and min(gains.values())>0
        else:gains=None
        results['eligibility'][variant]={'passed':passed,'minimum_y2_gain':minimum,'daily_positive':count,'gain_vs_weighted_only':gains}
        if passed:eligible.append((minimum,variant))
    results['selected']=max(eligible)[1] if eligible else None
    common.save_json(OUT/'pilot_results.json',results);print(json.dumps({'stage':'pilot_complete','selected':results['selected'],'eligibility':results['eligibility']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','pilot'],required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='preflight':print(json.dumps(preflight()),flush=True);return
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data SHA mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'model':'unchanged PeerGRU400 hidden128; y1 auxiliary head, y2 deployed',
        'input_and_graph':'same causal-normalized features and frozen training-only graph as 0.3263 model',
        'scale':'s=clip(sqrt(EWMA64(f000^2))/reference,.5,2); update includes current observed f000; per-stock daily reset',
        'reference':'RMS raw f000 sampled every16 ticks on training days0-3; float16 input cache semantics',
        'scaled_target':'p=s*GRU(x); loss=mean(task-weighted (p-y)^2 / s^2) / training_mean(1/s^2)',
        'weighted_only':'p=GRU(x); identical inverse-scale-squared loss weighting',
        'training':{'seed':42,'epochs':2,'batch':1024,'window':16,'lr':1e-4,'weight_decay':1e-4,'clip':1.,'task_weights':[.375,.625]},
        'deployment':'replace y2 peer128 member only; remaining members and output routing fixed',
        'gate':'y2 gain >=.001 both early folds and >=6/8 positive days; scaled variant must also beat weighted-only both folds; choose largest minimum-fold gain',
        'futility':'stop after A if neither variant reaches required .001 gain','late_dates':'not used in this pilot',
        'data_sha256':common.EXPECTED_SHA}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);common.save_json(OUT/'preflight.json',preflight())
    src,_=outer.prepare();pilot(src)

if __name__=='__main__':main()
