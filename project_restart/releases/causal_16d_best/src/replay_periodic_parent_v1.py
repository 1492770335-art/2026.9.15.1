"""Replay a deterministic downstream layer against digest-checked session and prefix forecasts."""
import hashlib
import json
import os

import numpy as np
import polars as pl
import experiment_periodic_near_carry_v1 as parent
import experiment_book_transition_rewards_v1 as previous
import replay_session_experts_16d as original
from experiment_causal_cross_horizon_ridge_v1 import measure

CACHE = parent.OUT / 'raw_trace'


def digest(raw):
    return hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest()


def prepare():
    CACHE.mkdir(exist_ok=True)
    model = None
    ref = json.loads((parent.OUT/'reload_verification.json').read_text())
    rows = []
    for day in range(16,20):
        raw = original.features(original.DATA,day)
        token = digest(raw)
        out = CACHE/f'day_{day}.npz'
        if not out.exists():
            if model is None:
                model = parent.load()
            with np.load(previous.parent.OUT/'raw_trace'/out.name) as p:
                assert str(p['raw_sha256']) == token
                n = int(p['prefix_ticks'])
                a,b = model(raw,p['full']), model(raw[:n],p['head'])
            tmp = out.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,full=a,head=b,raw_sha256=token,prefix_ticks=n)
            tmp.replace(out)
        with np.load(out) as st:
            assert str(st['raw_sha256']) == token
            full,head,n = st['full'],st['head'],int(st['prefix_ticks'])
        err = float(abs(full[:n]-head).max())
        assert full.shape == (*raw.shape[:2],3) and err < 1e-4
        y = labels(day)
        flat = full.reshape(-1,3)
        row = (flat,flat,y)
        assert abs(measure([row])['score']-ref['daily'][str(day+1)]['score'])<2e-9
        rows.append(row)
        print('verified parent date',day+1,flush=True)
    s = measure(rows)
    np.testing.assert_allclose(s['candidate'],ref['pooled']['candidate'],atol=2e-9,rtol=0)
    assert s['rows'] == 936250


def labels(day):
    return pl.scan_parquet(original.DATA).filter(pl.col('date')==day).sort('time','code').select('y0','y1','y2').collect().to_numpy()


def evaluate(layer, on_day=None):
    rows=[];daily={}
    ref=json.loads((parent.OUT/'reload_verification.json').read_text())
    for day in range(16,20):
        raw=original.features(original.DATA,day)
        with np.load(CACHE/f'day_{day}.npz') as st:
            assert str(st['raw_sha256'])==digest(raw)
            p,head,n=st['full'],st['head'],int(st['prefix_ticks'])
        q=layer(raw,p)
        h=layer(raw[:n],head)
        err=float(abs(q[:n]-h).max())
        assert np.isfinite(q).all() and err<1e-4
        row=(p.reshape(-1,3),q.reshape(-1,3),labels(day))
        s=measure([row])
        assert abs(s['baseline_score']-ref['daily'][str(day+1)]['score'])<2e-9
        daily[str(day+1)]={'score':s['score'],'gain':s['gain'],'prefix_error':err}
        if on_day is not None:
            on_day(day,raw,q,h)
        rows.append(row)
    s=measure(rows)
    np.testing.assert_allclose(s['base'],ref['pooled']['candidate'],atol=2e-9,rtol=0)
    assert s['rows']==936250
    return dict(pooled=s,daily=daily)


def verify(output, factory, reference):
    old=json.loads((reference/'reload_verification.json').read_text())
    first=None
    for name in ('raw_replay', 'reload_verification'):
        result=evaluate(factory())
        result['parent_comparison']={'model':reference.name,'score':old['pooled']['score'],
                                     'gain':result['pooled']['score']-old['pooled']['score']}
        if first is not None:
            np.testing.assert_allclose(result['pooled']['candidate'],first['pooled']['candidate'],atol=1.e-10,rtol=0)
            for day in result['daily']:
                assert abs(result['daily'][day]['score']-first['daily'][day]['score'])<1.e-10
        (output/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(name,result['pooled']['score'],result['parent_comparison'],flush=True)
        if result['parent_comparison']['gain']<=0:return False
        first=result
    return True


if __name__=='__main__':
    prepare()
