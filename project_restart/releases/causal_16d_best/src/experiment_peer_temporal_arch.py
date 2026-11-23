"""From-scratch, parameter-matched temporal encoder replacements for PeerGRU."""
import argparse
import gc
import hashlib
import inspect
import json
import time
import numpy as np
import torch
import peer_temporal_tcn as tcn
import peer_temporal_lru as lru
import experiment_peer_scale as scales
import experiment_peer_prune64 as current
import experiment_target_peer_gru as graphs
import experiment_dynamic_peer_gru as peer
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_y2_normalized_outer as outer
import experiment_normalized_y2_huber as components
import experiment_mixed_market_routing as routing

OUT=common.ROOT/'artifacts/project_restart_generalization/peer_temporal_arch'
MODES=('tcn_local','tcn_dilated','complex_lru')
BEST=current.OUT/'commit_peer_prune64'


def verify_best():
    expected=json.loads((current.OUT/'official_results.json').read_text())['hashes']
    actual={n:common.sha256(BEST/n) for n in expected}
    if actual!=expected:raise RuntimeError('Original best package changed')
    return actual


def build(mode):
    if mode=='complex_lru':return lru.PeerLRU()
    if mode=='tcn_local':return tcn.PeerTCN('local')
    if mode=='tcn_dilated':return tcn.PeerTCN('dilated')
    raise ValueError(mode)


def objective(prediction,target,weight):
    error=(prediction-target).square()
    return ((error*prediction.new_tensor([.375,.625])).sum(-1)*weight).mean()


def train(src,X,y,tid,scale,graph,mode,seed,path):
    torch.manual_seed(seed);model=build(mode).to('cuda:0')
    fp={'mode':mode,'seed':seed,'rows':len(X),'window':16,'epochs':2,'row_batch':1024,
        'plan_sha256':common.sha256(OUT/'plan.json'),
        'graph_sha256':hashlib.sha256(np.asarray(graph,dtype=np.float32).tobytes()).hexdigest()}
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['fingerprint']!=fp or meta['sha256']!=common.sha256(path) or len(meta['records'])!=2:raise RuntimeError('Checkpoint mismatch')
        model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));return model
    Y=torch.from_numpy(y[:,1:3]).to('cuda:0');S=torch.from_numpy(scale).to('cuda:0')
    weight_mean=float(np.mean(np.square(scale.astype(np.float64))**-1));W=S.square().reciprocal()/weight_mean
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    resume=path.with_suffix('.resume.pth');first=0;records=[]
    if resume.exists():
        saved=torch.load(resume,map_location='cuda:0',weights_only=True)
        if saved['fingerprint']!=fp or abs(saved['weight_mean']-weight_mean)>1e-10:raise RuntimeError('Resume mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);first=saved['next_epoch'];records=saved['records']
        if first not in (0,1,2) or len(records)!=first:raise RuntimeError('Incomplete resume epochs')
    started=time.perf_counter();batches=(len(X)+1023)//1024
    for epoch in range(first,2):
        model.train();seen=0;total=np.zeros(2);weighted=0.;gradient_norms=[]
        order=np.random.default_rng(seed+epoch).permutation(len(X))
        for batch,start in enumerate(range(0,len(X),1024),1):
            I=order[start:start+1024]
            inp=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            prediction=model(inp)*S[I,None];loss=objective(prediction,Y[I],W[I])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            opt.step();n=len(I);seen+=n;total+=(prediction-Y[I]).detach().square().sum(0).cpu().numpy();weighted+=float(loss.detach())*n
            if batch==1 or batch%250==0 or batch==batches:
                gradient_norms.append(float(grad))
                print(json.dumps({'stage':'training','model':path.name,'epoch':epoch+1,'batch':batch,'batches':batches,
                    'head_mse':(total/seen).tolist(),'weighted_loss':weighted/seen,'gradient_norm':float(grad),
                    'seconds':round(time.perf_counter()-started,1),'peak_gpu_GiB':round(torch.cuda.max_memory_allocated()/2**30,3)}),flush=True)
        if seen!=len(X):raise RuntimeError('Training row coverage mismatch')
        records.append({'head_mse':(total/seen).tolist(),'weighted_loss':weighted/seen,'rows':seen,'batches':batches,'gradient_samples':gradient_norms})
        tmp=resume.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'next_epoch':epoch+1,
            'fingerprint':fp,'weight_mean':weight_mean,'records':records},tmp);tmp.replace(resume)
    tmp=path.with_suffix('.pending.pth');torch.save(model.state_dict(),tmp);tmp.replace(path)
    common.save_json(path.with_suffix('.json'),{'fingerprint':fp,'records':records,'weight_mean':weight_mean,
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,'sha256':common.sha256(path)})
    return model


def predict(src,model,X,tid,scale):
    model.eval();heads=[];started=time.perf_counter()
    with torch.no_grad():
        for batch,start in enumerate(range(0,len(X),2048),1):
            I=np.arange(start,min(start+2048,len(X)))
            inp=torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()
            heads.append(model(inp).cpu().numpy()*scale[I,None])
            if batch==1 or batch%200==0 or start+2048>=len(X):
                print(json.dumps({'stage':'validation','rows':int(I[-1])+1,'total':len(X),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    heads=np.concatenate(heads)
    if not np.isfinite(heads).all():raise RuntimeError('Nonfinite validation predictions')
    return heads


def control(src,X,tid,scale,fold,p1):
    checkpoint=scales.OUT/f'scaled_target_{fold}_s42.pth';meta=json.loads(checkpoint.with_suffix('.json').read_text())
    if meta['variant']!='scaled_target' or meta['seed']!=42 or len(meta['losses'])!=2 or common.sha256(checkpoint)!=meta['sha256']:
        raise RuntimeError('Original row control checkpoint mismatch')
    old=np.load(scales.OUT/f'scaled_target_{fold}_prediction.npy');old_final=np.load(current.OUT/f'prediction_{fold}.npy')
    if old.shape!=(len(X),2) or old_final.shape!=(len(X),):raise RuntimeError('Control shape mismatch')
    model=peer.PeerGRU().to('cuda:0');model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=True));model.eval()
    day_starts=np.flatnonzero(tid[::50]==0)*50
    boundary=(day_starts[:,None]+np.array([0,49,50,15*50,16*50])[None,:]).reshape(-1)
    I=np.unique(np.concatenate([np.arange(50),boundary,np.linspace(0,len(X)-1,30).astype(np.int64)]))
    with torch.no_grad():
        pred=model(torch.from_numpy(src.make_batch(X,I,tid,16)).to('cuda:0').float()).cpu().numpy()*scale[I,None]
    error=float(np.abs(pred-old[I]).max())
    final_error=float(np.abs(routing.routed(p1,old[:,1])-old_final).max())
    if not np.isfinite([error,final_error]).all() or error>5e-5 or final_error>1e-6:raise RuntimeError('Cached baseline verification failed')
    del model;gc.collect();torch.cuda.empty_cache()
    return old,old_final,{'checkpoint':str(checkpoint),'sha256':meta['sha256'],'max_abs_heads':error,'max_abs_final':final_error,
        'head_prediction_sha256':common.sha256(scales.OUT/f'scaled_target_{fold}_prediction.npy'),
        'final_prediction_sha256':common.sha256(current.OUT/f'prediction_{fold}.npy')}


def passes_fold(mode,rec):
    comparisons=[rec]+([rec['vs_local']] if mode=='tcn_dilated' else [])
    for c in comparisons:
        if len(c['daily'])!=4:return False
        for value in (c,*c['daily'].values()):
            if not np.isfinite([value['base'],value['candidate'],value['delta']]).all():return False
    return rec['delta']>=.001 and (mode!='tcn_dilated' or rec['vs_local']['delta']>0)


def pilot(src):
    result={'folds':{},'controls':{},'late_dates_evaluated':False};ref=scales.reference();modes=list(MODES)
    for fold,spec in common.FOLDS.items():
        graph=np.load(graphs.OUT/f'graph_{fold}.npz')['target_graph']
        raw,y,tid=norm.load_days(spec['train'],normalized=True);raw=raw.astype(np.float32)
        X=peer.PeerArray(raw,graphs.context(raw,graph));scale=scales.load_scale(spec['train'],ref)
        print(json.dumps({'stage':'loaded','fold':fold,'rows':len(X),'modes':modes}),flush=True)
        for mode in modes:
            model=train(src,X,y,tid,scale,graph,mode,42,OUT/f'{mode}_{fold}_s42.pth')
            del model;gc.collect();torch.cuda.empty_cache()
        del X,raw,y,tid,scale;gc.collect()
        raw,y,tid=norm.load_days(spec['validation'],normalized=True);raw=raw.astype(np.float32)
        X=peer.PeerArray(raw,graphs.context(raw,graph));scale=scales.load_scale(spec['validation'],ref)
        p1,_,_=components.early_components(fold,spec['validation'])
        old_heads,old,check=control(src,X,tid,scale,fold,p1);result['controls'][fold]=check
        predictions={};fold_result={}
        for mode in modes:
            model=build(mode).to('cuda:0');model.load_state_dict(torch.load(OUT/f'{mode}_{fold}_s42.pth',map_location='cuda:0',weights_only=True))
            heads=predict(src,model,X,tid,scale);final=routing.routed(p1,heads[:,1]);predictions[mode]=final
            np.save(OUT/f'{mode}_{fold}_heads.npy',heads);np.save(OUT/f'{mode}_{fold}_prediction.npy',final)
            rec=scales.evaluate_prediction(final,old,y,spec['validation'],scale)
            if mode=='tcn_dilated':rec['vs_local']=scales.evaluate_prediction(final,predictions['tcn_local'],y,spec['validation'],scale)
            rec['head_corr']=norm.corr(heads[:,1],y[:,2]);rec['head_std']=float(heads[:,1].std())
            rec['head_mse']=float(np.mean((heads[:,1].astype(np.float64)-y[:,2])**2));rec['final_std']=float(final.std())
            fold_result[mode]=rec;result['folds'][fold]=fold_result;common.save_json(OUT/'pilot_partial.json',result)
            print(json.dumps({'stage':'variant_complete','fold':fold,'mode':mode,'corr':rec['candidate'],
                'gain_vs_gru':rec['delta'],'gain_vs_local':rec.get('vs_local',{}).get('delta'),
                'head_corr':rec['head_corr'],'head_std':rec['head_std']}),flush=True)
            del model,heads;gc.collect();torch.cuda.empty_cache()
        del X,raw,y,tid,scale,p1,old,old_heads,predictions;gc.collect()
        if fold=='A':
            eligible={m:passes_fold(m,r) for m,r in fold_result.items()};result['first_fold_gate']=eligible
            next_modes=[m for m in MODES if eligible.get(m,False)]
            if 'tcn_dilated' in next_modes and 'tcn_local' not in next_modes:next_modes.insert(0,'tcn_local')
            print(json.dumps({'stage':'first_fold_gate','eligible':eligible,'continue':next_modes}),flush=True)
            if not next_modes:break
            modes=next_modes
    result['eligibility']={};eligible=[]
    for mode in MODES:
        if not all(mode in result['folds'].get(f,{}) for f in common.FOLDS):
            result['eligibility'][mode]={'passed':False,'reason':'Stopped after first fold'};continue
        records=[result['folds'][f][mode] for f in common.FOLDS]
        count=sum(d['delta']>0 for r in records for d in r['daily'].values())
        passed=all(passes_fold(mode,r) for r in records) and count>=6
        result['eligibility'][mode]={'passed':passed,'daily_positive':count,'minimum_gain_vs_gru':min(r['delta'] for r in records)}
        if passed:eligible.append((min(r['delta'] for r in records),mode))
    result['selected']=max(eligible)[1] if eligible else None;result['best_hashes_unchanged']=verify_best()
    common.save_json(OUT/'pilot_results.json',result)
    print(json.dumps({'stage':'pilot_complete','selected':result['selected'],'eligibility':result['eligibility']}),flush=True)


def preflight(src):
    tests={'tcn':tcn.preflight(),'lru':lru.preflight()}
    torch.set_num_threads(4)
    models={}
    for mode in MODES:
        torch.manual_seed(42);models[mode]=build(mode)
    for mode in MODES[1:]:
        for name in ('head1','head2','input'):
            for key,v in getattr(models[MODES[0]],name).state_dict().items():
                torch.testing.assert_close(v,getattr(models[mode],name).state_dict()[key],atol=0,rtol=0)
    counts={m:sum(p.numel() for p in v.parameters()) for m,v in models.items()}
    if any(abs(n-203778)/203778>.005 for n in counts.values()):raise RuntimeError('Parameter budget mismatch')
    # Keep original zero-prefix, day-reset semantics in the actual make_batch loader.
    rng=np.random.default_rng(4);X=rng.normal(size=(2*20*50,400)).astype(np.float32)
    tid=np.tile(np.repeat(np.arange(20),50),2);I=np.array([0,49,50,999,1000,1001,1999])
    windows=src.make_batch(X,I,tid,16)
    for k,i in enumerate(I):
        length=min(16,int(tid[i])+1);expected=X[np.arange(i-(length-1)*50,i+1,50)]
        np.testing.assert_array_equal(windows[k,-length:],expected)
        np.testing.assert_array_equal(windows[k,:16-length],0)
    p=torch.randn(9,2,requires_grad=True);target=torch.randn_like(p);w=torch.rand(9)+.1
    original=(((p-target).square()*torch.tensor([.375,.625])).sum(-1)*w).mean()
    torch.testing.assert_close(objective(p,target,w),original,atol=0,rtol=0)
    tests['integration']={'identical_new_model_input_projection_and_heads':True,'parameters':counts,
        'baseline_parameters':203778,'actual_loader_daily_zero_prefix_verified':True,'main_loss_unchanged':True}
    return tests


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','pilot'],required=True);args=parser.parse_args()
    src,_=outer.prepare();torch.set_num_threads(4)
    print(json.dumps({'stage':'preflight_start'}),flush=True);tests=preflight(src);torch.set_num_threads(4)
    if args.stage=='preflight':print(json.dumps(tests),flush=True);return
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data hash mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'modes':list(MODES),'initialization':'All candidates from scratch; same seed42 reference head initialization and shared400->128 projection; final100 input columns start zero',
        'tcn':'400->128; four residual LN128,Linear128->192,depthwise causal k2,SiLU*sigmoid96,Linear96->128; local d1/1/1/1 RF5 vs dilated d1/2/4/8 RF16; identical weights at initialization',
        'lru':'400->128; two pre-LN residual complex diagonal LRU blocks, state72, gate MLP128->200 split100 and100->128; stable exp(-exp(nu)+i exp(theta)); learned gamma; rinit .8.. .99, phase0..2pi; exact real-tensor doubling scan',
        'parameters':tests['integration']['parameters'],'baseline_parameters':203778,
        'training':{'seed':42,'epochs':2,'window':16,'row_batch':1024,'lr':1e-4,'weight_decay':1e-4,'clip':1.,'main_weights':[.375,.625]},
        'loss':'unchanged scaled heads S and global inverseS2 weights, original two-target MSE',
        'fixed':'same raw-to-causal normalized inputs, train-only graph, lazy400 views, row permutation, S/W, p0/p1 and final routing',
        'padding':'original fixed16 zero-left-padded inputs, no pad skipping or new mask; per-prediction whole-window computation, no state persisted across windows/days',
        'control':'original row GRU checkpoints and archived full predictions verified by SHA and spread sample re-evaluation; previous experiment also reproduced weights and predictions exactly',
        'gate':'each fold >=.001 y3 final correlation vs GRU and >=6/8 positive daily gains; dilated additionally >local both folds; stop failing candidates after A, retain necessary local control in B',
        'selection':'largest minimum-fold gain among passing candidates; same-budget architecture pilot, not architecture-specific hyperparameter search',
        'late_dates':'not read in pilot; last4 historically reused; evaluate only if earlier two folds pass',
        'deployment':'new encoder requires a separate deployment class and stream/batch checks; never map new weights into old GRU tensors',
        'references':{'TCN':'https://arxiv.org/abs/1803.01271','LRU':'https://proceedings.mlr.press/v202/orvieto23a.html'},
        'adaptation':'Task-sized gated convolution and LRU backbone variants, not literal paper benchmark reproductions',
        'data_sha256':common.EXPECTED_SHA,'best_hashes':verify_best(),
        'source_sha256':{str(p):common.sha256(p) for p in (__file__,tcn.__file__,lru.__file__,scales.__file__,current.__file__,peer.__file__,graphs.__file__,norm.__file__,common.__file__,outer.__file__,components.__file__,routing.__file__,inspect.getfile(peer.JointMarketGRU),src.__file__)}}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Frozen plan mismatch')
    common.save_json(path,plan);common.save_json(OUT/'preflight.json',tests);pilot(src)

if __name__=='__main__':main()
