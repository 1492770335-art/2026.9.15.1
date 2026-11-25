"""Transfer the three liquidity-age responses separately to the 5-, 20- and 60-tick forecasts."""
import json
import sys

import numpy as np
import torch

import experiment_joint_order_clock_calibration_v1 as prior
import y3_touch_age_readout_references as history

age, study, trial = history.incumbent, history.study, history.trial
OUT = trial.clock.ART/'joint_touch_age_calibration_v1'
ARMS = prior.ARMS
fitted, gain = prior.fitted, prior.gain


def basis(old,p,x):
    # Column 0 is the sum, columns 1:4 the three horizons, then 3 ages by 3 horizons.
    ages = x[:,:,4:13].reshape(len(x),50,3,3).sum(3)
    return np.concatenate(((p[:,:,2]-old[:,:,2])[:,:,None],ages),2)


def record(d):
    raw = trial.clock.book.rawday(d)
    old = history.previous.references(d,raw)
    forecasts = history.references(d,raw)
    x = age.features(raw,np.arange(len(raw)),age.touch.queues.calibration(),age.learned(d-d%2))
    y = trial.clock.labels(d).reshape(-1,50,3)
    grams = []
    for a,p in zip(old,forecasts):
        z = basis(a,p,x)
        v = np.concatenate((np.ones_like(p[:,:,:1]),p,z,y),2).reshape(-1,11).astype(float)
        grams.append(v.T@v)
    return grams


def train():
    OUT.mkdir(exist_ok=True)
    stats = [record(d) for d in range(8,16)]
    weights, picks, targets = [], [], []
    for k in range(3):
        daily, folds = [], []
        for n in (2,4,6):
            ws = fitted(stats[:n],k)
            folds.append([gain(g,ws,k) for g in np.sum(stats[n:n+2],0)])
            daily.extend([[gain(g,ws,k) for g in pair] for pair in stats[n:n+2]])
        f, d = np.asarray(folds), np.asarray(daily)
        mean = f.mean(0)
        eligible = [j for j in range(1,len(ARMS)) if mean[:,j].min()>0 and f[-1,:,j].min()>0 and
                    (f[:,:,j]>0).sum(0).min()>=2 and (d[:,:,j]>0).sum(0).min()>=4]
        pick = max(eligible,key=lambda j:mean[:,j].min()) if eligible else 0
        weights.append(fitted(stats,k)[pick])
        picks.append(pick)
        targets.append(dict(folds=f.tolist(),daily=d.tolist(),mean=mean.tolist()))
        print('target',k+1,'pick',ARMS[pick],'gain',mean[:,pick],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=picks,arms=ARMS,targets=targets),indent=2))
    if any(picks):
        np.savez(OUT/'model.npz',w=np.column_stack(weights),selected=picks)
    return any(picks)


def readout():
    with np.load(age.OUT/'model.npz') as archive:
        st = dict(archive)
    with np.load(OUT/'model.npz') as archive:
        w = archive['w']
    def apply(raw, old):
        x = age.features(raw,np.arange(len(raw)),st,st['heads'])
        z = np.concatenate((trial.clock.center(old),x),2)
        p = old.copy()
        p[:,:,2] += age.changes(z,st)[int(st['pick'])]
        return p, p+(basis(old,p,x)@w).astype('float32')
    return apply


def layer():
    previous = history.previous.incumbent.layer()
    apply = readout()
    return lambda raw, p:apply(raw,previous(raw,p))[1]


def replay_with():
    ver, loader = history.previous.replay_with()
    apply = readout()
    def read(d,raw):
        p,h,n = loader(d,raw)
        return apply(raw,p)[1],apply(raw[:n],h)[1],n
    return ver, read


def replay():
    original = json.loads((age.OUT/'reload_verification.json').read_text())
    older = None
    for name in ('raw_replay','reload_verification'):
        ver,loader = history.previous.replay_with()
        apply = readout()
        daily,rows = {},[]
        for d in range(16,20):
            raw = ver.trace.replay.original.features(ver.trace.replay.original.DATA,d)
            p,h,n = loader(d,raw)
            baseline,q = apply(raw,p)
            _,head = apply(raw[:n],h)
            err = float(np.max(abs(q[:n]-head)))
            assert err<1e-4 and np.isfinite(q).all()
            pair = (baseline.reshape(-1,3),q.reshape(-1,3),ver.trace.replay.labels(d))
            rows.append(pair)
            score = ver.trace.older.measure([pair])
            assert abs(score['baseline_score']-original['daily'][str(d+1)]['score'])<1e-10
            daily[str(d+1)] = dict(score=score['score'],gain=score['gain'],prefix_error=err,corr=score['candidate'])
        score = ver.trace.older.measure(rows)
        assert score['rows']==936250
        np.testing.assert_allclose(score['base'],original['pooled']['candidate'],rtol=0,atol=1e-10)
        report = dict(parent=age.OUT.name,pooled=score,daily=daily)
        if older:
            assert older==report
        (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,score,flush=True)
        older = report
        if score['gain']<=0:
            break


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay()
