"""Cache verified predictions from the deployable probability normalization at each historical cutoff."""
import json
import sys

import numpy as np

import audit_y3_rank_capacity_effect_v1 as audit
import experiment_y3_rank_capacity_readout_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage

previous = incumbent.history
rank = incumbent.rank
base,trial,study,stack = previous.base,previous.trial,previous.study,previous.stack
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def read(day,raw):
    return read_stage(FULL,day,raw)


def references(day,raw):
    return read_verified(CACHE,day,raw)


def verify():
    FULL.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    nets,weight = incumbent.load_models()
    reader = incumbent.replay_with()[1]
    rows,daily = [],{}
    for d in range(16,20):
        raw = trial.clock.book.rawday(d)
        old,_,_ = previous.read(d,raw)
        q,h,n = reader(d,raw)
        independent = (audit.signals(raw,nets)[:,:,1]*weight).astype('float32')
        err = float(abs((old[:,:,2]+independent)-q[:,:,2]).max())
        assert err<3e-6 and abs(q[:n]-h).max()<1e-4 and np.array_equal(q[:,:,:2],old[:,:,:2])
        row = old.reshape(-1,3),q.reshape(-1,3),stack.verifier.trace.replay.labels(d)
        rows.append(row)
        score = stack.verifier.trace.older.measure([row])['score']
        assert abs(score-report['daily'][str(d+1)]['score'])<1e-10
        daily[d+1] = dict(score=score,independent_readout_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled = stack.verifier.trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-report['pooled']['score'])<1e-12
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))
    print('verified',pooled['score'],flush=True)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    info = json.loads((rank.OUT/'capacity_training.json').read_text())
    training,records,checks = [],[],{}
    for day in range(16):
        if day in (4,8,12):
            nets = rank.fit(training)[1:]
            if day>=8:
                w = base.calibrate(records)[0]
                fold = []
        if day>=4:
            raw = trial.clock.book.rawday(day)
            y = trial.clock.labels(day).reshape(-1,50,3)
            z = audit.signals(raw,nets)
            ps = previous.references(day,raw) if day>=8 else previous.previous.incumbent.parent.warm.read(day,raw)
            pair = [base.records(p,y,z) for p in ps]
            records.append(pair)
            if day>=8:
                fold.append(pair)
                q = ps.copy()
                q[:,:,:,2] += incumbent.change(raw,nets,w)
                for i,(a,b) in enumerate(zip(ps,q)):
                    gain = np.corrcoef(b[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-np.corrcoef(a[:,:,2].ravel(),y[:,:,2].ravel())[0,1]
                    assert abs(gain-base.score(pair[i],np.full(3,w))[0][2])<1e-8
                np.savez(CACHE/f'day_{day:02d}.npz',forecasts=q,cutoff=day-day%2,calibration_cutoff=day-day%4,
                         raw_hash=digest(raw),forecast_hash=digest(q))
                if day in (11,15):
                    gain = [base.score(m,np.full(3,w))[0][2] for m in np.sum(fold,0)]
                    np.testing.assert_allclose(gain,np.asarray(info['folds'])[(day-11)//4,:,0,2],atol=1e-8,rtol=0)
                    checks[day+1] = gain
        training.append(rank.samples(day))
    (CACHE/'verified.json').write_text(json.dumps(dict(folds=checks),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as release
        release.run(incumbent,read,rank,incumbent.rank_history.read)
        release.run(incumbent,read)
