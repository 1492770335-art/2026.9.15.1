"""Full 20-day refit of the spectral expert fusion model, deterministic and resumable."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
import experiment_peer_dual_temporal as dual
import experiment_normalized_y0 as y0
import experiment_normalized_y1 as y1
import experiment_spectral_expert_fusion_v1 as spectral
import reproduce_restart_k16 as repro

prior=spectral.prior
OUT=prior.ART/'spectral_full20_v1'
DAYS=list(range(20))
SEEDS=(42,43,44)
RAW=prior.ROOT/'artifacts/project_restart_full20_reproduction'

def read(path):
    return json.loads(path.read_text())

def raw_sources():
    plan=read(RAW/'plan.json')
    if plan['train_dates']!=DAYS or plan['data_sha256']!=dual.common.EXPECTED_SHA or plan['window']!=16:
        raise RuntimeError('Raw checkpoint scope mismatch')
    result={}
    for row in read(RAW/'results.json')['train']:
        if row['is_market'] and row['label'] in (0,1):
            path=Path(row['ckpt'])
            if dual.common.sha256(path)!=row['sha256'] or row['E']!=(4,2)[row['label']]:
                raise RuntimeError('Raw weights differ')
            result[row['seed'],row['label']]=path
    assert len(result)==6
    return result

def norm_day(day):
    if day<16:
        return np.load(dual.norm.OUT/'normalized_days'/f'day_{day:02d}_x.npy',mmap_mode='r').reshape(-1,100)
    raw=np.load(prior.CACHE/f'day_{day:02d}_x.npy',mmap_mode='r')
    return dual.outer.transform(raw,*dual.norm.global_stats())

def prepare_data():
    counts=[len(prior.labels(d)) for d in DAYS]
    n=sum(counts)
    assert n==4681350
    raw=np.empty((n,100),dtype=np.float32)
    tid=np.empty(n,dtype=np.int64)
    y=np.empty((n,3),dtype=np.float32)
    start=0
    for d,size in zip(DAYS,counts):
        raw[start:start+size]=norm_day(d)
        y[start:start+size]=prior.labels(d)
        tid[start:start+size]=np.arange(size)//50
        start+=size
    return raw,y,tid

def train():
    OUT.mkdir(exist_ok=True)
    raw_sources()
    if dual.common.sha256(dual.common.DATA)!=dual.common.EXPECTED_SHA:
        raise RuntimeError('Data mismatch')
    src,market=dual.outer.prepare()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    plan=dict(days=DAYS,seeds=SEEDS,rows=4681350,window=16,epochs={'norm0':4,'norm1':2,'tcn_local':2,'complex_lru':2},
              normalized_stats='original first four days',graph='refit all 20',projector='refit all 20',calibration='chronological OOF from days 8 through 19',source_sha=dual.common.sha256(__file__))
    if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:raise RuntimeError('Different training plan')
    dual.common.save_json(OUT/'plan.json',plan)
    graph,info=dual.graphs.fit_graph(DAYS)
    np.save(OUT/'graph.npy',graph['target_graph']);np.save(OUT/'projector.npy',spectral.projector(DAYS))
    print('loading full normalized grid',flush=True)
    raw,y,tid=prepare_data()
    market_view=repro.LazyMarketArray(raw)
    for seed in SEEDS:
        for label in (0,1):
            path=OUT/f'norm{label}_{seed}.pth'
            print('member',path.name,flush=True)
            if label==0:model=y0.train(src,market,market_view,y,tid,seed,4,path)
            else:model=y1.train(src,market,market_view,y,tid,seed,path)
            del model;gc.collect();torch.cuda.empty_cache()
    del market_view;gc.collect()
    X=dual.peer.PeerArray(raw,dual.graphs.context(raw,graph['target_graph']))
    scale=dual.scales.load_scale(DAYS,dual.scales.reference())
    for seed in SEEDS:
        for mode in ('tcn_local','complex_lru'):
            path=OUT/f'{mode}_{seed}.pth'
            print('member',path.name,flush=True)
            model=dual.original.train(src,X,y,tid,scale,graph['target_graph'],mode,seed,path)
            del model;gc.collect();torch.cuda.empty_cache()
    print('all full20 members ready',flush=True)

if __name__=='__main__':
    train()
