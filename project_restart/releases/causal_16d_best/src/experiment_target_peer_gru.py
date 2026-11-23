"""Train-only y2 correlation graph prior; paired randomized graph control."""
import gc
import json
import numpy as np
import torch
import experiment_dynamic_peer_gru as dynamic
import experiment_y2_gru_capacity as common
import experiment_y2_generalization as norm
import experiment_y2_normalized_outer as outer
import experiment_joint_market_gru as joint
import experiment_normalized_y2_huber as components
import experiment_mixed_market_routing as routing
import experiment_norm_refinement as refine

OUT=outer.ROOT/'artifacts/project_restart_generalization/target_peer_gru'
VARIANTS=('target_graph','shuffled')


def correlation(days):
    daily=[]
    for day in days:
        y=np.load(norm.CACHE/f'day_{day:02d}_y.npy',mmap_mode='r')[:,:,2].astype(np.float64)
        y-=y.mean(1,keepdims=True)
        daily.append(np.nan_to_num(np.corrcoef(y.T)))
    return np.mean(daily,0)


def fit_graph(days):
    c=correlation(days);np.fill_diagonal(c,-2.)
    ids=np.argsort(c,axis=1)[:,-8:]
    values=np.take_along_axis(c,ids,axis=1);weights=np.exp((values-values.max(1,keepdims=True))/.1)
    weights/=weights.sum(1,keepdims=True)
    a=np.zeros((50,50),dtype=np.float32);np.put_along_axis(a,ids,weights.astype(np.float32),1)
    p=dynamic.PERM.numpy();graphs={'target_graph':a,'shuffled':a[p][:,p]}
    for graph in graphs.values():
        np.testing.assert_allclose(graph.sum(1),1,atol=1e-6)
        assert np.all(graph.diagonal()==0) and np.all((graph>0).sum(1)==8)
    h=len(days)//2;b=correlation(days[:h]);d=correlation(days[h:]);off=np.triu_indices(50,1)
    info={'fitted_days':days,'label':'y2 training labels only, same-tick stock mean removed',
          'mean_selected_correlation':float(values.mean()),'train_halves_edge_correlation':float(np.corrcoef(b[off],d[off])[0,1]),
          'inference':'fixed graph applied to observed input features, never to true labels or predictions'}
    return graphs,info


def context(raw,graph):
    result=np.empty(raw.shape,dtype=np.float16);a=torch.from_numpy(graph)
    for start in range(0,len(raw),6400):
        g=torch.from_numpy(raw[start:start+6400].reshape(-1,50,100))
        result[start:start+len(g)*50]=(a@g-g.mean(1,keepdim=True)).numpy().reshape(-1,100).astype(np.float16)
    return result


def main():
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data hash mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    src,_=outer.prepare();OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'graph_fit':'mean within-day stock-relative y2 correlation over training dates only',
          'graph':'top8 positive-ranked off-diagonal neighbors, softmax correlation/.1',
          'control':'fixed simultaneous row/column permutation seed2026',
          'input':'original own100/mean100/std100 plus fixed graph observed feature context minus market mean',
          'model':'same PeerGRU400 hidden128 with training-only auxiliary y1 head, extra100 input weights start at zero',
          'training':{'seed':42,'epochs':2,'window':16,'batch':1024,'lr':1e-4,'weight_decay':1e-4,'clip':1.,'loss_weights':[.375,.625]},
          'deployment':'replace only auxiliary y2 head, all remaining members and output routing fixed',
          'gate':'y2 gain >=.002 in both folds vs auxiliary baseline, beat shuffled both folds, >=6/8 daily gains',
          'late_dates':'not accessed; retrospective evaluation permitted only after gate',
          'reason':'Training-only diagnostic: residual y2 edge matrices have split-half correlation about .64-.66',
          'data_sha256':common.EXPECTED_SHA}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);common.save_json(OUT/'preflight.json',dynamic.preflight())
    results={'variants':{v:{} for v in VARIANTS},'graphs':{}}
    for fold,spec in common.FOLDS.items():
        graphs,info=fit_graph(spec['train']);np.savez(OUT/f'graph_{fold}.npz',**graphs)
        results['graphs'][fold]=info;common.save_json(OUT/'graph_manifest.json',results['graphs'])
        print(json.dumps({'stage':'graph_fitted','fold':fold,**info}),flush=True)
        raw,y,tid=norm.load_days(spec['train'],normalized=True);raw=raw.astype(np.float32)
        for variant in VARIANTS:
            X=dynamic.PeerArray(raw,context(raw,graphs[variant]))
            model=dynamic.train(src,X,y,tid,variant,42,OUT/f'{variant}_{fold}_s42.pth')
            del model,X;gc.collect();torch.cuda.empty_cache()
        del raw,y,tid;gc.collect()
        raw,y,tid=norm.load_days(spec['validation'],normalized=True);raw=raw.astype(np.float32)
        p1,_,p64=components.early_components(fold,spec['validation'])
        old_head=np.load(joint.OUT/f'shared_{fold}_prediction.npy')[:,1]
        base=routing.routed(p1,.5*(old_head+p64))
        for variant in VARIANTS:
            X=dynamic.PeerArray(raw,context(raw,graphs[variant]));model=dynamic.PeerGRU().to('cuda:0')
            model.load_state_dict(torch.load(OUT/f'{variant}_{fold}_s42.pth',map_location='cuda:0',weights_only=True))
            p=joint.predict(src,model,X,tid);np.save(OUT/f'{variant}_{fold}_prediction.npy',p)
            candidate=routing.routed(p1,.5*(p[:,1]+p64))
            rec=refine.metrics(candidate,base,y[:,2],spec['validation'])
            rec['total_score_delta']=.5*rec['delta'];rec['direct_y2_corr']=norm.corr(p[:,1],y[:,2])
            rec['baseline_direct_y2_corr']=norm.corr(old_head,y[:,2])
            results['variants'][variant][fold]=rec;common.save_json(OUT/'pilot_partial.json',results)
            print(json.dumps({'stage':'fold_complete','fold':fold,'variant':variant,**rec}),flush=True)
            del model,X,p,candidate;gc.collect();torch.cuda.empty_cache()
        del raw,y,tid,p1,p64,old_head,base;gc.collect()
    normal=results['variants']['target_graph'];shuffled=results['variants']['shuffled']
    count=sum(d['delta']>0 for r in normal.values() for d in r['daily'].values())
    gains={f:normal[f]['candidate']-shuffled[f]['candidate'] for f in common.FOLDS}
    results['eligibility']={'passed':all(r['delta']>=.002 for r in normal.values()) and min(gains.values())>0 and count>=6,
        'daily_positive':count,'gain_vs_shuffled':gains}
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete',**results['eligibility']}),flush=True)

if __name__=='__main__':main()
