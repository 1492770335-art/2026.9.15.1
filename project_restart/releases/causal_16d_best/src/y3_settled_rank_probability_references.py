"""Check the deployed rank-odds update by recomputing eligible histories at each decision time."""
import json
import sys

import numpy as np

import experiment_y3_settled_rank_probabilities_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage

previous = incumbent.history
base,trial,study,stack = previous.base,previous.trial,previous.study,previous.stack
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def read(d,raw):
    return read_stage(FULL,d,raw)


def references(d,raw):
    return read_verified(CACHE,d,raw)


def independent(raw,nets,weight):
    from scipy.stats import rankdata
    free,pay = incumbent.panel(raw,nets)
    before = incumbent.balanced(free)
    k,coef,missing = incumbent.proxy.historical_geometry()
    cumulative = incumbent.proxy.compound.compound(raw,k)
    present = incumbent.proxy.compound.present(raw,missing)
    result = np.zeros(raw.shape[:2],np.float32)
    for lo,hi in base.segments(len(raw)):
        observed,expected = [],[]
        next_origin = lo
        for start in range(lo,hi,40):
            # Consume a forecast only when both endpoints have arrived.
            while next_origin+60<=start:
                t = next_origin
                stop = t+60
                good = present[t:stop+1].all(0)
                mass = np.zeros((50,10))
                if good.sum()>=35:
                    def ret(h,j):
                        return coef[j,0]+coef[j,1]*np.expm1(cumulative[t+h+1]-cumulative[t+1])/k
                    value = (ret(60,2)-base.RATIO*ret(20,1))[good]
                    a = (rankdata(value,method='min')-1)*10/good.sum()
                    b = rankdata(value,method='max')*10/good.sum()
                    for j in range(10):
                        mass[good,j] = (np.minimum(b,j+1)-np.maximum(a,j)).clip(0)/(b-a)
                else:
                    good[:] = False
                observed.append(mass)
                expected.append(before[t]*good[:,None])
                next_origin += 4
            if not observed:
                continue
            def odds(n):
                real = np.sum(observed[-n:],0)+12
                prior = np.sum(expected[-n:],0,dtype=float)+12
                return np.log(real/prior).clip(-1.2,1.2)
            factor = np.exp(.5*(odds(90)-odds(360)))
            sl = slice(start,min(start+40,hi))
            prob = free[sl]*factor[None]
            # Direct alternating mass normalization, retaining the same four iterations.
            for _ in range(4):
                prob = (prob/prob.sum(1,keepdims=True))*5
                prob = prob/prob.sum(2,keepdims=True)
            result[sl] = ((prob-before[sl])*pay[sl,None]).sum(2)*weight
    return result


def verify():
    FULL.mkdir(exist_ok=True)
    info = json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']
    nets,_ = incumbent.parent.load_models()
    with np.load(incumbent.OUT/'model.npz') as f:
        assert incumbent.ARMS[int(f['pick'])+1]=='conditional log-odds'
        w = float(f['w'])
    reader = incumbent.replay_with()[1]
    rows,daily = [],{}
    for d in range(16,20):
        raw = trial.clock.book.rawday(d)
        p,_,_ = previous.read(d,raw)
        q,h,n = reader(d,raw)
        delta = independent(raw,nets,w)
        err = float(abs(p[:,:,2]+delta-q[:,:,2]).max())
        assert err<4e-6 and abs(q[:n]-h).max()<1e-4 and np.array_equal(p[:,:,:2],q[:,:,:2])
        row = p.reshape(-1,3),q.reshape(-1,3),stack.verifier.trace.replay.labels(d)
        rows.append(row)
        daily[d+1] = dict(independent_error=err,score=stack.verifier.trace.older.measure([row])['score'])
        np.savez(FULL/f'day_{d}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled = stack.verifier.trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-info['score'])<1e-12
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))


def rebuild():
    CACHE.mkdir(exist_ok=True)
    rank = incumbent.rank
    info = json.loads((incumbent.OUT/'results.json').read_text())
    pick = info['selected']-1
    training,records,checks = [],[],{}
    for d in range(16):
        if d in (4,8,12):
            nets = rank.fit(training)[1:]
            if d>=8:
                coef = base.calibrate(records)[pick]
        if d>=4:
            raw = trial.clock.book.rawday(d)
            y = trial.clock.labels(d).reshape(-1,50,3)
            ps = previous.references(d,raw) if d>=8 else previous.previous.previous.incumbent.parent.warm.read(d,raw)
            z = incumbent.signals(raw,nets)
            records.append([base.records(p,y,z) for p in ps])
            if d>=8:
                q = ps.copy()
                q[:,:,:,2] += (coef*z[:,:,pick]).astype('float32')
                gains = []
                for p,s in zip(ps,q):
                    gains.append(np.corrcoef(s[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-
                                 np.corrcoef(p[:,:,2].ravel(),y[:,:,2].ravel())[0,1])
                np.testing.assert_allclose(gains,np.asarray(info['daily'])[d-8,:,0,pick+1],atol=1e-8,rtol=0)
                checks[d+1] = gains
                np.savez(CACHE/f'day_{d:02d}.npz',forecasts=q,cutoff=d-d%2,calibration_cutoff=d-d%4,
                         raw_hash=digest(raw),forecast_hash=digest(q))
        training.append(rank.samples(d))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(base.calibrate(records)[pick],f['w'],atol=1e-9,rtol=0)
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=checks),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as release
        release.run(incumbent,read,previous.incumbent,previous.read)
        release.run(incumbent,read)
