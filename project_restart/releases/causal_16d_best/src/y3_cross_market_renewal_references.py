"""Verify the external event-clock correction and supply forecast references for later experiments."""
import json
import sys

import numpy as np
import torch

import experiment_y3_cross_market_marked_renewal_v1 as incumbent
from y3_touch_clock_references import digest,read_stage,verify_stage
from y3_expanding_path_references import read_verified

previous = incumbent.history
trial,stack,study = previous.trial,previous.stack,previous.study
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def read(d,raw):
    return read_stage(FULL,d,raw)


def references(d,raw):
    return read_verified(CACHE,d,raw)


def replay_with():
    return stack.verifier,read


def independent(raw,table,stocks,coef):
    # Advance each name's own state one observation at a time. The selected correction is
    # the change in the 21–60 payoff after introducing FI conditional event signs.
    delta = table[3,0]-table[0,0]
    signal = np.zeros(raw.shape[:2],np.float32)
    ids = np.arange(50)
    for t,r in enumerate(raw[:,:,0].astype(float)-.00136):
        if t in (0,2401):
            last = np.zeros(50,int)
            pattern = np.full(50,15,int)
            lag = np.zeros(50,int)
            drift = np.zeros(50)
            count = np.zeros(50,int)
        move = abs(r)>.022
        signed = np.sign(r).astype(int)
        match = (signed==last)|(count==0)
        pattern[move] = ((pattern[move]*2+match[move])%16)
        lag = np.where(move,0,np.minimum(lag+1,31))
        drift[move] = .925*drift[move]+.075*signed[move]
        last[move] = signed[move]
        count += move
        orientation = np.digitize(drift*last,[-.13,.13])
        signal[t,ids] = delta[stocks,pattern,orientation,lag]*last*(count>=5)
    return ((signal-signal.mean(1,keepdims=True))*coef).astype('float32')


def verify():
    with np.load(incumbent.OUT/'model.npz') as st:
        assert int(st['pick'])==6
        w,g,s = st['w'],st['stocks'],float(st['scale'])
    def apply(raw,p):
        q = p.copy()
        q[:,:,2] += independent(raw,w,g,s)
        return q
    verify_stage(incumbent,previous.read,apply)
    snapshot = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    blocks = []
    for d in range(16,20):
        raw = stack.verifier.trace.replay.original.features(stack.verifier.trace.replay.original.DATA,d)
        p,h,n = previous.read(d,raw)
        q,_,_ = read(d,raw)
        y = trial.clock.labels(d).reshape(-1,50,3)
        v = np.stack((p[:,:,2]-incumbent.RATIO*p[:,:,1],q[:,:,2]-incumbent.RATIO*q[:,:,1],
                      y[:,:,2]-incumbent.RATIO*y[:,:,1]),axis=2).reshape(-1,3).astype(float)
        v = np.column_stack((np.ones(len(v)),v))
        blocks.append(v.T@v)
    def ic(m):
        c = incumbent.covariance(m)
        return (c[:2,2]/np.sqrt(c.diagonal()[:2]*c[2,2])).tolist()
    report = dict(rows=snapshot['pooled']['rows'],columns=['previous','candidate'],
                  far=ic(sum(blocks)),daily=[ic(m) for m in blocks])
    (incumbent.OUT/'tail_gain.json').write_text(json.dumps(report,indent=2))


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = int(report['selected'])
    stats,history,errors = [],[],[]
    stocks = incumbent.pace()[0]
    for d in range(16):
        raw = trial.clock.book.rawday(d)
        if d>=4:
            if d%2==0:
                w = incumbent.table(history)
                if d>=8:
                    scale = incumbent.calibrate(stats)[pick-1]
                    group = []
            z = incumbent.signals(raw,w,stocks)
            y = trial.clock.labels(d).reshape(-1,50,3)
            pp = np.asarray(incumbent.historical(d,raw))
            group_stats = [incumbent.records(p,y,z) for p in pp]
            if d>=8:
                delta = independent(raw,w,stocks,scale)
                np.testing.assert_allclose(delta,z[:,:,pick-1]*scale,rtol=0,atol=1e-7)
                # Match the deployed accumulation order.
                delta = (z[:,:,pick-1]*scale).astype('float32')
                pp[:,:,:,2] += delta
                group.append(group_stats)
                if d>=11 and d%2:
                    obs = [incumbent.score(m,incumbent.calibrate(stats[:-1])) for m in np.sum(group,axis=0)]
                    err = float(np.max(abs(np.asarray(obs)[:,:,pick]-np.asarray(report['folds'])[(d-11)//2,:,:,pick])))
                    assert err<1.e-10
                    errors.append(err)
                np.savez(CACHE/f'day_{d:02d}.npz',forecasts=pp,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(pp))
            stats.append(group_stats)
        history.append(incumbent.moment(raw))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_array_equal(f['w'],incumbent.table(history))
        np.testing.assert_array_equal(f['stocks'],stocks)
        assert abs(float(f['scale'])-incumbent.calibrate(stats)[pick-1])<1.e-12
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_y3_release_gain_v1 as gain
        gain.run(incumbent,read,previous.incumbent,previous.read)
        gain.run(incumbent,read)
    else:
        rebuild()
