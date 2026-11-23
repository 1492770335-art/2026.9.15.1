"""Complete the shorter-only updates in the historical production path and replay that path in full."""
import importlib
import json
import numpy as np
import experiment_joint_horizon_completion_v1 as joint
import replay_verified_forecast_parent_v1 as descendant
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

RAW=ART/'legacy_horizon_completion_raw_v1'
OUT=ART/'legacy_horizon_completion_v1'
BASE=ART/'propagated_return_kernel_v1'
STAGES=[('stock_calibration','endpoint_innovation'),('cross_horizon_rank_readout','dynamic_factor_return_net'),
        ('dynamic_factor_return_net','pairwise_book_tree'),('field_tokens_skew_means','categorical_suffix_responses'),
        ('categorical_suffix_responses','conditional_tail_payouts'),('conditional_tail_payouts','supervised_gradient_kernel'),
        ('rolling_drift_strength','propagated_return_kernel')]
STAGES=[(a+'_v1',b+'_v1',b+'_v1') for a,b in STAGES]
NAMES=tuple(dict.fromkeys(x for a,b,_ in STAGES for x in (a,b)))


def completion():
    cov=joint.residual_covariances();ws=np.zeros((len(STAGES),2,3,3))
    for i,(a,b,_) in enumerate(STAGES):
        root=ART/b
        if b=='endpoint_innovation_v1':
            with np.load(root/'model.npz') as data:mask=data['weights']!=0
        elif b=='dynamic_factor_return_net_v1':mask=np.any(np.load(root/'coef.npy')!=0,axis=1)
        elif b=='pairwise_book_tree_v1':mask=np.stack((np.zeros(3,bool),np.any(np.load(root/'coef.npy')!=0,axis=0)))
        else:mask=np.array(json.loads((root/'results.json').read_text())['eligible'])
        for part in range(2):
            for h in (1,2):
                seen=np.flatnonzero(mask[part,:h])
                if mask[part,h] or len(seen)==0:continue
                daily=[]
                for g in cov[:,part]:
                    sg=g[np.ix_(seen,seen)]
                    daily.append(np.linalg.solve(sg+.025*np.diag(sg.diagonal()),g[seen,h]))
                ws[i,part,seen,h]=np.median(daily,axis=0)
    return ws


def cached(weights):
    RAW.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);np.save(RAW/'completion.npy',weights)
    rows={RAW:[],OUT:[]}
    for d in range(16,20):
        src={k:np.load(ART/k/f'pred_{d}.npy') for k in NAMES}
        base=src[BASE.name];q=joint.apply(base,src,weights,STAGES)
        parent=np.load(joint.OUT/f'pred_{d}.npy');final=(parent+(q-base)).astype('float32');y=labels(d)
        for name,p,new in ((RAW,base,q),(OUT,parent,final)):
            np.save(name/f'pred_{d}.npy',new);rows[name].append((p.reshape(-1,3),new.reshape(-1,3),y))
    reports={}
    for name in rows:
        r=measure(rows[name]);reports[name]=r
        (name/'results.json').write_text(json.dumps({'20':r,'stages':STAGES,'completion':weights.tolist()},indent=2))
        print(name.name,r,flush=True)
    return reports[OUT]['gain']>0


def raw_replay(weights):
    import os
    import replay_paired_book_trees_v1 as paired
    import replay_curve_quantile_market_v1 as source
    state={}
    def observe(name,value):
        if name=='stock_calibration_v1':state.clear()
        if name in NAMES:state[name]=np.array(value,dtype='float32',copy=True)
    source.stage_observer=observe
    orig={}
    def wrap(name,build):
        def setup():
            directory,fn=build()
            def run(raw,p):
                q=fn(raw,p);observe(name,q);return q
            return directory,run
        return setup
    try:
        for key in ('field_tokens_skew_means_v1','categorical_suffix_responses_v1','conditional_tail_payouts_v1',
                    'supervised_gradient_kernel_v1','rolling_drift_strength_v1','propagated_return_kernel_v1'):
            module=importlib.import_module('experiment_'+key);orig[module]=module.refinement
            module.refinement=wrap(key,module.refinement)
        root=importlib.import_module('experiment_propagated_return_kernel_v1')
        _,core=root.refinement()
        def apply(raw,p):
            observe('pairwise_book_tree_v1',p);base=core(raw,p)
            assert set(NAMES)==set(state) and all(v.shape==p.shape for v in state.values())
            out=joint.apply(base,state,weights,STAGES);state.clear();return out
        os.environ['SAVE_RAW_PREDICTION_TRACE']='1'
        paired.main((RAW,apply))
    finally:
        source.stage_observer=None
        for mod,build in orig.items():mod.refinement=build
    descendant.main(joint.OUT,OUT,lambda raw,p,old,complete:(p+(complete-old)).astype('float32'),(BASE,RAW))


if __name__=='__main__':
    w=completion()
    if cached(w):raw_replay(w)
