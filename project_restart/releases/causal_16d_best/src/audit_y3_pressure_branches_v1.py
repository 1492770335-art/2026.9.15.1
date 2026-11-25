"""Replay sibling Y3 corrections on the same independently rebuilt parent grid and compare paired scores."""
import importlib
import json
import os
import sys

import numpy as np
import torch

import experiment_book_transition_rewards_v1 as older
import experiment_y3_stock_pressure_dynamics_v1 as parent
import experiment_y3_residual_peer_curve_v1 as paired
import replay_periodic_parent_v1 as replay

CACHE=parent.OUT/'verified_parent_trace'


def prepare():
    CACHE.mkdir(exist_ok=True)
    official=json.loads((parent.OUT/'composed_replay.json').read_text())
    predict=None;rows=[]
    for d in range(16,20):
        target=CACHE/f'day_{d}.npz'
        raw=replay.original.features(replay.original.DATA,d)
        sha=replay.digest(raw)
        if not target.exists():
            if predict is None:predict=parent.load()
            with np.load(older.parent.OUT/'raw_trace'/target.name) as s:
                assert str(s['raw_sha256'])==sha
                n=int(s['prefix_ticks'])
                full=predict(raw,s['full'])
                head=predict(raw[:n],s['head'])
            assert np.isfinite(full).all() and np.max(abs(full[:n]-head))<.0001
            day_score=older.measure([(full.reshape(-1,3),full.reshape(-1,3),replay.labels(d))])['score']
            assert abs(day_score-official['daily'][str(d+1)]['score'])<1e-10
            tmp=target.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,full=full,head=head,prefix_ticks=n,raw_sha256=sha,
                     full_sha256=replay.digest(full),head_sha256=replay.digest(head))
            tmp.replace(target)
        full,head,n=load_day(d,raw)
        assert np.max(abs(full[:n]-head))<.0001
        a=full.reshape(-1,3);row=(a,a,replay.labels(d))
        assert abs(older.measure([row])['score']-official['daily'][str(d+1)]['score'])<1e-10
        rows.append(row);print('parent day',d+1,flush=True)
    verified=older.measure(rows)
    assert verified['rows']==936250
    np.testing.assert_allclose(official['pooled']['candidate'],verified['candidate'],atol=1e-10,rtol=0)


def load_day(d,raw):
    with np.load(CACHE/f'day_{d}.npz') as s:
        assert str(s['raw_sha256'])==replay.digest(raw)
        full,head=s['full'],s['head']
        assert str(s['full_sha256'])==replay.digest(full) and str(s['head_sha256'])==replay.digest(head)
        return full,head,int(s['prefix_ticks'])


def verify(name):
    module=importlib.import_module(name)
    assert module.incumbent is parent
    official=json.loads((module.OUT/'reload_verification.json').read_text())
    baseline=json.loads((parent.OUT/'composed_replay.json').read_text())
    delta=module.correction()
    daily={};rows=[];blocks=[]
    for d in range(16,20):
        raw=replay.original.features(replay.original.DATA,d)
        y=replay.labels(d)
        p,head,n=load_day(d,raw)
        q=p.copy();q[:,:,2]+=delta(raw)
        head[:,:,2]+=delta(raw[:n])
        err=float(abs(q[:n]-head).max())
        assert np.isfinite(q).all() and err<.0001
        row=(p.reshape(-1,3),q.reshape(-1,3),y)
        s=older.measure([row])
        assert abs(s['score']-official['daily'][str(d+1)]['score'])<1e-10
        assert abs(s['baseline_score']-baseline['daily'][str(d+1)]['score'])<1e-10
        daily[d+1]=dict(score=s['score'],gain=s['gain'],prefix_error=err)
        rows.append(row)
        blocks.append(paired.blockstats(p,y.reshape(p.shape),[q[:,:,2].astype(float)-p[:,:,2]]))
    score=older.measure(rows)
    assert score['rows']==936250
    np.testing.assert_allclose(score['candidate'],official['pooled']['candidate'],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(score['candidate'][:2],score['base'][:2])
    ci=np.array(paired.intervals(blocks))[:,0]*.5
    change=paired.pooled(sum(b.sum(0) for b in blocks))[0]*.5
    assert abs(change-score['gain'])<2e-10
    report=dict(pooled=score,daily=daily,parent=parent.OUT.name,paired_score_percentiles=ci.tolist(),
                resamples=1600,block_ticks=120)
    (module.OUT/'composed_replay.json').write_text(json.dumps(report,indent=2))
    print(name,score['score'],score['gain'],ci,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if len(sys.argv)==1:prepare()
    else:
        for name in sys.argv[1:]:verify(name)
