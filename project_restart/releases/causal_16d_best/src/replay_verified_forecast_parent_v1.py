"""Replay a downstream forecaster using the independently reconstructed full and truncated parent predictions."""
import hashlib
import json
import numpy as np
import polars as pl
from replay_curve_quantile_market_v1 import DATA,summary


def stats(pred,y):
    a=pred.reshape(-1,3).astype(float)
    return np.stack((np.full(3,len(y)),a.sum(0),y.sum(0),(a*a).sum(0),(y*y).sum(0),(a*y).sum(0)))


def main(parent,dest,advance,auxiliary_parents=()):
    baseline=json.loads((parent/'raw_replay.json').read_text())
    totals=np.zeros((6,3));base_stats=totals.copy();record={'parent_raw_replay':str(parent/'raw_replay.json'),'daily':{}}
    dest.mkdir(exist_ok=True);trace=dest/'raw_trace';trace.mkdir(exist_ok=True)
    for d in range(16,20):
        scan=pl.scan_parquet(DATA).filter(pl.col('date')==d).sort('time','code')
        frame=scan.select('time','code',*[f'f{i:03d}' for i in range(100)]).collect()
        ti=frame['time'].to_numpy().reshape(-1,50);code=frame['code'].to_numpy().reshape(-1,50)
        assert (ti==np.arange(len(ti))[:,None]).all() and (code==np.arange(50)[None,:]).all()
        raw=frame.select(pl.exclude('time','code')).to_numpy().astype(np.float16).reshape(-1,50,100)
        del frame
        digest=hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest()
        with np.load(parent/'raw_trace'/f'day_{d}.npz') as st:
            assert str(st['raw_sha256'])==digest
            previous=st['full'];head=st['head'];n=int(st['prefix_ticks']);stream=float(st['opening_stream_max_error'])
        assert n==baseline['daily'][str(d+1)]['prefix_ticks']
        assert previous.shape==raw.shape[:2]+(3,) and head.shape==previous[:n].shape
        full_aux=[];head_aux=[]
        for origin in auxiliary_parents:
            with np.load(origin/'raw_trace'/f'day_{d}.npz') as extra:
                assert str(extra['raw_sha256'])==digest and int(extra['prefix_ticks'])==n
                full_aux.append(extra['full']);head_aux.append(extra['head'])
                assert full_aux[-1].shape==previous.shape and head_aux[-1].shape==head.shape
            assert (origin/'raw_replay.json').exists()
        q=advance(raw,previous,*full_aux);partial=advance(raw[:n],head,*head_aux)
        err=float(np.max(abs(q[:n]-partial)))
        assert np.isfinite(q).all() and err<1e-4
        np.savez(trace/f'day_{d}.npz',full=q,head=partial,raw_sha256=digest,prefix_ticks=n,opening_stream_max_error=stream)
        # Both downstream forecast paths have been emitted before any evaluation label is read.
        y=scan.select('y0','y1','y2').collect().to_numpy().astype(float)
        ds=stats(q,y);orig=stats(previous,y);totals+=ds;base_stats+=orig
        assert abs(summary(orig)['score']-baseline['daily'][str(d+1)]['score'])<2e-9
        cache=np.load(dest/f'pred_{d}.npy')
        entry=dict(**summary(ds),prefix_ticks=n,opening_stream_max_error=stream,final_prefix_max_error=err,
                   cache_rmse=float(np.sqrt(np.mean((q-cache)**2,dtype=float))),cache_max_error=float(np.max(abs(q-cache))))
        record['daily'][d+1]=entry;print(d+1,entry,flush=True)
    if auxiliary_parents:record['auxiliary_raw_replays']=[str(origin/'raw_replay.json') for origin in auxiliary_parents]
    record['pooled']=summary(totals)
    assert record['pooled']['rows']==936250 and abs(summary(base_stats)['score']-baseline['pooled']['score'])<2e-9
    expected=json.loads((dest/'results.json').read_text())['20']['score']
    assert abs(expected-record['pooled']['score'])<2e-6
    (dest/'raw_replay.json').write_text(json.dumps(record,indent=2))
    print('verified raw parent and descendant',record['pooled'],flush=True)
