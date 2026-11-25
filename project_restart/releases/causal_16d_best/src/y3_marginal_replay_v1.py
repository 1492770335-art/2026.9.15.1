"""Replay one Y3-only update against the recorded four-day incumbent."""
import json
import numpy as np


def replay(parent,update,cached):
    baseline=json.loads((parent.OUT/'reload_verification.json').read_text())
    verifier,before=(parent.parent if cached else parent).replay_with()
    if cached:market=parent.correction_from_heads()
    rows=[];daily={}
    for day in range(16,20):
        raw=verifier.trace.replay.original.features(verifier.trace.replay.original.DATA,day)
        old,short,n=before(day,raw)
        if cached:
            opening=np.load(parent.ART/'opening_oof_cache_v1'/f'day_{day:02d}.npy').reshape(old.shape)
            heads=[np.load(parent.ART/parent.experts.experts.SOURCES[i]/f'side_{day}.npy').reshape(old.shape)
                   for i in parent.EXPERT_IDS]
            old[:,:,2]+=market(opening,heads)
        new=old.copy()
        new[:,:,2]+=update(raw,old)
        np.testing.assert_array_equal(new[:,:,:2],old[:,:,:2])
        assert np.isfinite(new).all()
        row=(old.reshape(-1,3),new.reshape(-1,3),verifier.trace.replay.labels(day))
        score=verifier.trace.older.measure([row])
        assert abs(score['baseline_score']-baseline['daily'][str(day+1)]['score'])<(3e-7 if cached else 1e-10)
        daily[day+1]=dict(score=score['score'],gain=score['gain'],corr=score['candidate'])
        if not cached:
            short[:,:,2]+=update(raw[:n],short)
            err=float(abs(new[:n]-short).max())
            assert err<1e-4
            daily[day+1]['prefix_error']=err
        rows.append(row)
    pooled=verifier.trace.older.measure(rows)
    assert pooled['rows']==936250
    return dict(parent=parent.OUT.name,pooled=pooled,daily=daily)


def evaluate(parent,folder,load,cached):
    report=replay(parent,load(),cached)
    name='cached_validation' if cached else 'raw_replay'
    (folder/(name+'.json')).write_text(json.dumps(report,indent=2))
    print(name,report['pooled'],flush=True)
    if not cached:
        assert replay(parent,load(),False)==report
        (folder/'reload_verification.json').write_text(json.dumps(report,indent=2))
        print('reload verified',report['pooled'],flush=True)
