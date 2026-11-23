"""Remove the legacy GRU64 branch; freeze formula on early folds before late evaluation."""
import argparse
import gc
import json
import shutil
import numpy as np
import torch
import experiment_peer_scale as baseline
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_y2_normalized_outer as outer
import experiment_normalized_y2_huber as components
import experiment_normalized_y0 as y0run
import experiment_normalized_y1 as y1run
import experiment_mixed_market_routing as routing

OUT=common.ROOT/'artifacts/project_restart_generalization/peer_prune64'
ORIGINAL=baseline.OUT/'commit_peer_scale'


def evaluate():
    plan={'change':'Remove legacy norm64; own=p128 instead of .5*(p128+p64)',
          'frozen':'all trained weights, graph, target scales, p0/p1, market routing .75',
          'folds':common.FOLDS,'gate':'y2 gain >=.001 both folds and >=6/8 positive days',
          'training':'none; reuse prior frozen checkpoints and predictions',
          'late_dates':'16-19; retrospective validation only after early gate',
          'data_sha256':common.EXPECTED_SHA}
    if (OUT/'plan.json').exists() and json.loads((OUT/'plan.json').read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(OUT/'plan.json',plan)
    early={'folds':{}}
    for fold,spec in common.FOLDS.items():
        days=spec['validation'];y=np.concatenate([np.load(norm.CACHE/f'day_{d:02d}_y.npy').reshape(-1,3) for d in days])
        p1,_,p64=components.early_components(fold,days)
        p128=np.load(baseline.OUT/f'scaled_target_{fold}_prediction.npy')[:,1]
        old=routing.routed(p1,.5*(p128+p64));new=routing.routed(p1,p128)
        rec=baseline.evaluate_prediction(new,old,y,days,baseline.load_scale(days,baseline.reference()))
        early['folds'][fold]=rec
        np.save(OUT/f'prediction_{fold}.npy',new)
    early['daily_positive']=sum(d['delta']>0 for r in early['folds'].values() for d in r['daily'].values())
    early['passed']=all(r['delta']>=.001 for r in early['folds'].values()) and early['daily_positive']>=6
    common.save_json(OUT/'pilot_results.json',early)
    print(json.dumps({'stage':'early','passed':early['passed'],'daily_positive':early['daily_positive'],
        'gains':{f:r['delta'] for f,r in early['folds'].items()}}),flush=True)
    if not early['passed']:return
    z=np.load(outer.OUT/'validation_components.npz');y=z['target']
    p0=.5*(z['y0']+np.load(y0run.OUT/'outer_prediction.npy'))
    p1=.5*(z['y1']+np.load(y1run.OUT/'outer_prediction.npy'))
    p128=np.load(baseline.OUT/'outer_joint_heads.npy')[:,1]
    old=np.stack([p0,p1,np.load(baseline.OUT/'outer_prediction.npy')],1)
    new=old.copy();new[:,2]=routing.routed(p1,p128)
    result={'baseline':outer.metrics(old,y),'candidate':outer.metrics(new,y),'daily':{},
            'late_dates_status':'Repeatedly exposed retrospective validation; not blind',
            'early_gate':early,'training':'No additional training; same first16-day models'}
    archived=json.loads((baseline.OUT/'outer_results.json').read_text())['candidate']
    for key in ('corr','score'):np.testing.assert_allclose(result['baseline'][key],archived[key],rtol=0,atol=1e-10)
    offset=0
    for day,n in zip(range(16,20),z['day_rows']):
        sl=slice(offset,offset+n);b=outer.metrics(old[sl],y[sl]);c=outer.metrics(new[sl],y[sl]);offset+=n
        result['daily'][str(day)]={'baseline':b,'candidate':c,'score_delta':c['score']-b['score']}
    np.save(OUT/'outer_prediction.npy',new)
    common.save_json(OUT/'outer_results.json',result)
    print(json.dumps({'stage':'outer','baseline':result['baseline'],'candidate':result['candidate'],
        'daily_score_delta':{d:r['score_delta'] for d,r in result['daily'].items()}}),flush=True)


EXTENSION='''

class PrunedScalePeerTarget(ScalePeerTarget):
    def __init__(self,config):
        super().__init__(config)
        del self.members64
    def predict(self,x):
        observed=x[:,0].to(torch.float16).double()
        if self.scale_state is None:
            self.scale_state=self.scale_reference.square().expand(50).clone()
        self.scale_state=self.scale_state+(2./65.)*(observed.square()-self.scale_state)
        scale=(self.scale_state.sqrt()/self.scale_reference).clamp(.5,2.).float()[:,None]
        xn=_normalized_frame(self,x)
        market=_push_history(self,'hist_market',_market_frame(x))
        frame=_market_frame(xn)
        norm_market=_push_history(self,'hist_norm_market',frame)
        context=(self.peer_graph@xn-xn.mean(0,keepdim=True)).to(torch.float16).float()
        history=_push_history(self,'hist_peer',torch.cat([frame,context],-1))
        p128=torch.stack([m(history) for m in self.members128]).mean(0)
        if self.scale_output:p128=p128*scale
        rawy1=torch.stack([m(market) for m in self.market_y1]).mean(0)
        normy1=torch.stack([m(norm_market) for m in self.normalized_market_y1]).mean(0)
        p1=.5*(rawy1+normy1)
        return .5*(p1+p128)+(self.market_weight-.5)*(p1.mean(0,keepdim=True)-p128.mean(0,keepdim=True))
'''


def package():
    result=json.loads((OUT/'outer_results.json').read_text())
    if not result['early_gate']['passed'] or result['candidate']['score']<=result['baseline']['score']:
        print(json.dumps({'stage':'not_promoted','candidate':result['candidate']['rounded']}),flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    original_report=json.loads((baseline.OUT/'official_results.json').read_text())
    for name,h in original_report['hashes'].items():
        if common.sha256(ORIGINAL/name)!=h:raise RuntimeError('Baseline package fingerprint mismatch')
    commit=OUT/'commit_peer_prune64';(commit/'conf').mkdir(parents=True,exist_ok=True);(commit/'model').mkdir(exist_ok=True)
    for name in ('base.py','network.py'):shutil.copy2(ORIGINAL/name,commit/name)
    for j in range(3):
        for directory,suffix in [('conf','json'),('model','pth')]:shutil.copy2(ORIGINAL/directory/f'Model_{j}.{suffix}',commit/directory/f'Model_{j}.{suffix}')
    code=(ORIGINAL/'network.py').read_text()+EXTENSION
    compile(code,str(commit/'network.py'),'exec');(commit/'network.py').write_text(code)
    conf=json.loads((commit/'conf'/'Model_2.json').read_text());conf['model']='PrunedScalePeerTarget'
    common.save_json(commit/'conf'/'Model_2.json',conf)
    state=torch.load(ORIGINAL/'model'/'Model_2.pth',map_location='cpu',weights_only=True)
    removed=[k for k in state if k.startswith('members64.')]
    if len(removed)!=18:raise RuntimeError('Unexpected removed keys')
    for key in removed:del state[key]
    torch.save(state,commit/'model'/'Model_2.pth')
    src,_=outer.prepare()
    from script.evaluate import Evaluator
    evaluator=Evaluator(str(common.DATA),str(commit),'cuda:0')
    expected=np.load(OUT/'outer_prediction.npy');z=np.load(outer.OUT/'validation_components.npz')
    max_error=0.;offset=0
    for day,n in zip(range(16,20),z['day_rows']):
        if day not in (16,17):break
        raw,_,_=src.load_block([day]);observed=[]
        for model in evaluator.models:model.reset()
        with torch.no_grad():
            for x in raw[:128*50].reshape(-1,50,100):
                inp=torch.from_numpy(x).to('cuda:0')
                observed.append(torch.cat([m.predict(inp.clone()) for m in evaluator.models],1).cpu().numpy())
        observed=np.concatenate(observed);error=float(np.abs(observed-expected[offset:offset+len(observed)]).max())
        max_error=max(error,max_error)
        if error>5e-4:raise RuntimeError(f'Stream/batch mismatch day{day}: {error}')
        offset+=n;del raw,observed;gc.collect()
    print(json.dumps({'stage':'streaming_preflight','days':[16,17],'max_abs_error':max_error}),flush=True)
    del z,expected;gc.collect()
    score=evaluator.evaluate(date=[16,17,18,19])
    if any(abs(score[k]-result['candidate']['rounded'][k])>.00011 for k in score):raise RuntimeError('Official/vector mismatch')
    for key in ('corr0','corr1'):
        if score[key]!=original_report['pooled_official'][key]:raise RuntimeError('Unchanged target changed')
    for name,h in original_report['hashes'].items():
        if common.sha256(ORIGINAL/name)!=h:raise RuntimeError('Original modified')
    report={'pooled_official':score,'baseline_official':original_report['pooled_official'],
        'delta_score':round(score['score']-original_report['pooled_official']['score'],4),
        'commit':str(commit),'streaming_max_abs_error':max_error,'removed_parameter_keys':removed,
        'early_gate':result['early_gate'],'daily':result['daily'],'note':result['late_dates_status'],
        'hashes':{str(p.relative_to(commit)):common.sha256(p) for p in sorted(commit.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.pth')}}
    common.save_json(OUT/'official_results.json',report)
    print(json.dumps({'stage':'complete','pooled_official':score,'delta_score':report['delta_score'],'commit':str(commit)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['evaluate','package'],required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data hash mismatch')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='evaluate':evaluate()
    else:package()

if __name__=='__main__':main()
