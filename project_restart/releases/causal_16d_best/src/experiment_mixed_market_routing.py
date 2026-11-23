"""Recheck fixed market routing after the early-selected normalized y1 improvement."""
import json
import numpy as np
import experiment_y2_normalized_outer as outer
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_normalized_y0 as y0run
import experiment_normalized_y1 as y1run
import experiment_normalized_y2_huber as helper
import experiment_norm_refinement as refine

OUT=outer.ROOT/'artifacts/project_restart_generalization/mixed_market_routing'

def routed(p1,own):
    a=p1.reshape(-1,50);b=own.reshape(-1,50)
    return (.5*(a+b)+.25*(a.mean(1,keepdims=True)-b.mean(1,keepdims=True))).ravel()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'baseline':'0.3222 mixed normalized y0/y1 package',
          'candidate':'market_y2=0.75*mean50(mixed_y1)+0.25*mean50(normalized_y2_own); relative y2 stays 50/50',
          'gate':'positive y2 pooled delta in both early folds and >=6/8 positive daily deltas',
          'weight':.75,'outer_train':list(range(16)),'outer_validation':[16,17,18,19],
          'note':'Reuses trained weights. This is a follow-up to raw-y1 routing with a changed y1 component. Late dates already exposed.'}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan)
    results={'folds':{}}
    for fold,spec in common.FOLDS.items():
        ds=spec['validation'];p1,p128,p64=helper.early_components(fold,ds)
        y=np.concatenate([np.load(norm.CACHE/f'day_{d:02d}_y.npy').reshape(-1,3)[:,2] for d in ds])
        own=.5*(p128+p64);base=.5*(p1+own);p=routed(p1,own)
        results['folds'][fold]=refine.metrics(p,base,y,ds)
    positives=sum(d['delta']>0 for r in results['folds'].values() for d in r['daily'].values())
    results['daily_positive']=positives
    results['passed']=all(r['delta']>0 for r in results['folds'].values()) and positives>=6
    common.save_json(OUT/'development.json',results)
    print(json.dumps({'stage':'development',**results}),flush=True)
    if not results['passed']:return
    z=np.load(outer.OUT/'validation_components.npz')
    p0=.5*(z['y0']+np.load(y0run.OUT/'outer_prediction.npy'))
    p1=.5*(z['y1']+np.load(y1run.OUT/'outer_prediction.npy'));own=.5*(z['norm128']+z['norm64'])
    base=np.stack([p0,p1,.5*(p1+own)],1);candidate=base.copy();candidate[:,2]=routed(p1,own)
    np.save(OUT/'outer_prediction.npy',candidate[:,2])
    y=z['target'];result={'baseline':outer.metrics(base,y),'candidate':outer.metrics(candidate,y),'daily':{}}
    start=0
    for day,n in zip(range(16,20),z['day_rows']):
        sl=slice(start,start+n)
        result['daily'][str(day)]={'baseline':outer.metrics(base[sl],y[sl]),'candidate':outer.metrics(candidate[sl],y[sl])};start+=n
    common.save_json(OUT/'outer_results.json',result)
    print(json.dumps({'stage':'outer_complete',**result}),flush=True)

if __name__=='__main__':main()
