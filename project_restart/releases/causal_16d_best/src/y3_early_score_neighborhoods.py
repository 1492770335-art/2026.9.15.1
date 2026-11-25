"""Build strictly chronological reference forecasts on days 5–8 for warm-starting conditional stock comparisons."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps/tree_models'))
import lightgbm as lgb
import experiment_y3_paired_gross_flow_v1 as book
import experiment_y3_order_response_filter_v1 as filt
import y3_disjoint_validation as trial

OUT=trial.clock.ART/'y3_early_score_neighborhoods_v1'


def features(raw):
    pressure,level,move=book.components(raw)
    src=np.concatenate([pressure,level,move,raw[:,:,:1].astype('float32')],2)
    fast=filt.smooth(src,8);slow=filt.smooth(src,96)
    market=np.broadcast_to(fast.mean(1,keepdims=True),fast.shape)
    return np.concatenate((np.arcsinh(raw.astype('float32')).clip(-6,6),fast,fast-slow,market),2).astype('float16')


def prepare():
    OUT.mkdir(exist_ok=True)
    history=[];results=[]
    for d in range(8):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        x=features(raw)
        if d>=4:
            if d%2==0:
                train,truth=[np.concatenate(v) for v in zip(*history)]
                params=dict(objective='huber',alpha=1.,learning_rate=.055,num_leaves=23,min_data_in_leaf=350,
                            lambda_l2=70,feature_fraction=.8,verbosity=-1,force_col_wise=True,num_threads=2,max_bin=63,seed=409)
                models=[lgb.train(params,lgb.Dataset(train,label=truth[:,j]),num_boost_round=95) for j in range(3)]
            def predict(a):
                mat=a.reshape(-1,a.shape[-1])
                return np.stack([m.predict(mat,num_threads=2) for m in models],1).reshape(len(a),50,3).astype('float32')
            p=predict(x)
            np.testing.assert_allclose(p[:181],predict(features(raw[:181])),atol=1e-7,rtol=0)
            report=[float(np.corrcoef(p[:,:,j].ravel(),y[:,:,j].ravel())[0,1]) for j in range(3)]
            results.append(dict(day=d+1,last_training_day=d-d%2,corr=report))
            np.savez(OUT/f'day_{d:02d}.npz',p=p,raw_hash=trial.previous.prior.digest(raw),train_end=d-d%2)
            print(results[-1],flush=True)
        t=np.arange(d%7,len(y),7)
        history.append((x[t].reshape(-1,x.shape[-1]),y[t].reshape(-1,3).astype('float32')))
    (OUT/'verified.json').write_text(json.dumps(results,indent=2))


def load(d,raw):
    assert 4<=d<8
    with np.load(OUT/f'day_{d:02d}.npz') as f:
        assert int(f['train_end'])==d-d%2 and f['raw_hash'].item()==trial.previous.prior.digest(raw)
        return f['p']


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
