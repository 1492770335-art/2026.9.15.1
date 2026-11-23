"""Causal state-similarity peers vs shuffled connections, trained from scratch.

Current auxiliary MarketGRU is the baseline. Only its y2 member is replaced;
old y0/y1, normalized GRU64, and output routing remain fixed.
"""
import argparse
import gc
import json
import time
import numpy as np
import torch
from torch import nn
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_y2_normalized_outer as outer
import experiment_joint_market_gru as joint
import experiment_normalized_y2_huber as components
import experiment_mixed_market_routing as routing
import experiment_norm_refinement as refine
from joint_market_gru import JointMarketGRU

OUT=outer.ROOT/'artifacts/project_restart_generalization/dynamic_peer_gru'
VARIANTS=('similar','shuffled')
PERM=torch.from_numpy(np.random.default_rng(2026).permutation(50))


def peer_context(grid,variant):
    """Each timestamp is independent; no labels, future, or fitted test statistics."""
    mean=grid.mean(1,keepdim=True)
    state=grid[:,:,:83]-mean[:,:,:83]
    state=state/state.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    sim=state@state.transpose(1,2)
    sim[:,torch.arange(50),torch.arange(50)]=-torch.inf
    values,ids=sim.topk(8,dim=-1)
    graph=torch.zeros_like(sim).scatter_(-1,ids,torch.softmax(values/.25,dim=-1))
    if variant=='shuffled':graph=graph[:,PERM][:,:,PERM]
    elif variant!='similar':raise ValueError(variant)
    return graph@grid-mean


class PeerGRU(nn.Module):
    def __init__(self):
        super().__init__()
        base=JointMarketGRU('shared')
        self.shared=nn.GRU(400,128,batch_first=True)
        self.head1=base.head1;self.head2=base.head2
        state=base.shared.state_dict()
        state['weight_ih_l0']=torch.cat([state['weight_ih_l0'],torch.zeros(384,100)],1)
        self.shared.load_state_dict(state)

    def forward(self,x):
        h,_=self.shared(x);h=h[:,-1]
        return torch.cat([self.head1(h),self.head2(h)],-1)


class PeerArray:
    def __init__(self,raw,context):
        if raw.shape!=context.shape or raw.shape[1]!=100 or len(raw)%50:raise ValueError('Invalid rows')
        self.raw=raw;self.context=context
        grid=raw.reshape(-1,50,100)
        self.mean=grid.mean(1);self.std=grid.std(1,ddof=1)
        self.shape=(len(raw),400)
    def __len__(self):return len(self.raw)
    def __getitem__(self,i):
        return np.concatenate([self.raw[i],self.mean[i//50],self.std[i//50],self.context[i]],-1)


def preflight():
    torch.manual_seed(42);base=JointMarketGRU('shared')
    torch.manual_seed(42);candidate=PeerGRU()
    torch.manual_seed(82);x=torch.randn(7,16,400)
    torch.testing.assert_close(base(x[:,:,:300]),candidate(x),atol=1e-6,rtol=1e-5)
    y=torch.randn(7,2);loss=(candidate(x)-y).square().mean();loss.backward()
    g=float(candidate.shared.weight_ih_l0.grad[:,300:].norm())
    if g<=0:raise RuntimeError('Peer branch has no gradient')
    grid=torch.randn(9,50,100)
    checks={}
    for v in VARIANTS:
        p=peer_context(grid,v)
        changed=grid.clone();changed[5:]+=torch.randn_like(changed[5:])*5
        torch.testing.assert_close(p[:5],peer_context(changed,v)[:5],atol=0,rtol=0)
        chunks=torch.cat([peer_context(grid[:4],v),peer_context(grid[4:],v)])
        torch.testing.assert_close(p,chunks,atol=1e-6,rtol=1e-5)
        checks[v]={'future_perturbation_invariant':True,'chunk_invariant':True}
    return {'zero_initialized_extension_matches_baseline':True,'peer_gradient_norm':g,
            'parameters':sum(p.numel() for p in candidate.parameters()),'causality':checks}


def cache_context(days):
    for variant in VARIANTS:(OUT/variant).mkdir(exist_ok=True)
    for day in days:
        paths={v:OUT/v/f'day_{day:02d}.npy' for v in VARIANTS}
        if all(p.exists() for p in paths.values()):continue
        x=np.load(norm.OUT/'normalized_days'/f'day_{day:02d}_x.npy',mmap_mode='r')
        arrays={v:np.lib.format.open_memmap(p.with_suffix('.pending.npy'),mode='w+',dtype=np.float16,shape=x.shape) for v,p in paths.items()}
        for start in range(0,len(x),128):
            grid=torch.from_numpy(np.array(x[start:start+128],dtype=np.float32))
            for v in VARIANTS:arrays[v][start:start+128]=peer_context(grid,v).numpy().astype(np.float16)
        for v,p in paths.items():
            arrays[v].flush();del arrays[v];p.with_suffix('.pending.npy').replace(p)
        print(json.dumps({'stage':'context_cached','day':day,'ticks':len(x)}),flush=True)


def load_context(days,variant):
    return np.concatenate([np.load(OUT/variant/f'day_{d:02d}.npy',mmap_mode='r').reshape(-1,100) for d in days])


def train(src,X,y,tid,variant,seed,path):
    torch.manual_seed(seed);model=PeerGRU().to('cuda:0')
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
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=mse.detach().cpu().numpy()*len(I)
            if batch==1 or batch%250==0 or batch==batches:
                print(json.dumps({'stage':'training','model':path.name,'epoch':epoch+1,'batch':batch,
                    'batches':batches,'mse':(total/min(batch*1024,len(X))).tolist(),
                    'seconds':round(time.perf_counter()-started,1)}),flush=True)
        losses.append((total/len(X)).tolist())
        tmp=resume.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
            'next_epoch':epoch+1,'variant':variant,'seed':seed,'losses':losses},tmp);tmp.replace(resume)
    tmp=path.with_suffix('.pending.pth');torch.save(model.state_dict(),tmp);tmp.replace(path)
    common.save_json(path.with_suffix('.json'),{'variant':variant,'seed':seed,'losses':losses,
        'seconds':time.perf_counter()-started,'sha256':common.sha256(path)})
    return model


def pilot(src):
    results={'variants':{v:{} for v in VARIANTS}}
    for fold,spec in common.FOLDS.items():
        raw,y,tid=norm.load_days(spec['train'],normalized=True);raw=raw.astype(np.float32)
        for variant in VARIANTS:
            print(json.dumps({'stage':'load_context','fold':fold,'variant':variant}),flush=True)
            X=PeerArray(raw,load_context(spec['train'],variant))
            model=train(src,X,y,tid,variant,42,OUT/f'{variant}_{fold}_s42.pth')
            del X,model;gc.collect();torch.cuda.empty_cache()
        del raw,y,tid;gc.collect()
        raw,y,tid=norm.load_days(spec['validation'],normalized=True);raw=raw.astype(np.float32)
        p1,_,p64=components.early_components(fold,spec['validation'])
        baseline_head=np.load(joint.OUT/f'shared_{fold}_prediction.npy')[:,1]
        baseline=routing.routed(p1,.5*(baseline_head+p64))
        for variant in VARIANTS:
            X=PeerArray(raw,load_context(spec['validation'],variant));model=PeerGRU().to('cuda:0')
            model.load_state_dict(torch.load(OUT/f'{variant}_{fold}_s42.pth',map_location='cuda:0',weights_only=True))
            p=joint.predict(src,model,X,tid);np.save(OUT/f'{variant}_{fold}_prediction.npy',p)
            candidate=routing.routed(p1,.5*(p[:,1]+p64))
            rec=refine.metrics(candidate,baseline,y[:,2],spec['validation'])
            rec['total_score_delta']=.5*rec['delta']
            rec['direct_y2_corr']=norm.corr(p[:,1],y[:,2])
            rec['baseline_direct_y2_corr']=norm.corr(baseline_head,y[:,2])
            rec['prediction_corr_with_baseline']=norm.corr(p[:,1],baseline_head)
            results['variants'][variant][fold]=rec
            common.save_json(OUT/'pilot_partial.json',results)
            print(json.dumps({'stage':'fold_complete','fold':fold,'variant':variant,**rec}),flush=True)
            del X,model,p,candidate;gc.collect();torch.cuda.empty_cache()
        del raw,y,tid,p1,p64,baseline,baseline_head;gc.collect()
    sim=results['variants']['similar'];rnd=results['variants']['shuffled']
    count=sum(d['delta']>0 for r in sim.values() for d in r['daily'].values())
    gains={f:sim[f]['candidate']-rnd[f]['candidate'] for f in common.FOLDS}
    results['eligibility']={'passed':all(r['delta']>=.002 for r in sim.values()) and min(gains.values())>0 and count>=6,
        'daily_positive':count,'gain_vs_shuffled':gains}
    common.save_json(OUT/'pilot_results.json',results)
    print(json.dumps({'stage':'pilot_complete',**results['eligibility']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','pilot'],required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='preflight':print(json.dumps(preflight()),flush=True);return
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError('Data SHA mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'folds':common.FOLDS,'input':'original own100/mean100/std100 plus peer-relative100',
          'graph':'causally normalized continuous83, subtract same-tick market mean, cosine top8 excluding self, softmax temperature .25',
          'control':'same graph with a fixed simultaneous row/column permutation, seed2026',
          'initialization':'from scratch seed42; original MarketGRU300 parameters matched, extra100 input weights zero',
          'training':{'epochs':2,'batch':1024,'lr':1e-4,'weight_decay':1e-4,'clip':1.,'window':16,'loss_weights':[.375,.625]},
          'deployment':'replace auxiliary MarketGRU y2 member only; existing y0/y1/norm64/output routing unchanged',
          'gate':'y2 gain >=.002 both folds vs current auxiliary baseline, beat shuffled both folds, >=6/8 positive days',
          'late_dates':'not read for this pilot; previously exposed retrospective validation',
          'data_sha256':common.EXPECTED_SHA}
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Plan mismatch')
    common.save_json(path,plan);common.save_json(OUT/'preflight.json',preflight())
    src,_=outer.prepare();cache_context(list(range(16)));pilot(src)

if __name__=='__main__':main()
