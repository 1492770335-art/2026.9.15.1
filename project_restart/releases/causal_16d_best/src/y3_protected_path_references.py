"""Reproduce the final Y3 path readout and prepare causal historical predictions."""
import json
import sys

import numpy as np
import torch

import experiment_y3_protected_path_decoder_v1 as incumbent
from y3_touch_clock_references import read_stage,digest
from y3_expanding_path_references import read_verified

previous = incumbent.history
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
    with np.load(incumbent.OUT/'model.npz') as f:
        weights = f['w']
        nets = [incumbent.path.Decoder(int(f['n'])).cuda().eval() for _ in incumbent.path.MODES]
        for i,net in enumerate(nets):
            net.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in net.state_dict()})
    trace = stack.verifier.trace
    run = incumbent.replay_with()[1]
    rows,attribution,checks = [],[],{}
    for day in range(16,20):
        raw = trial.clock.book.rawday(day)
        y = trial.clock.labels(day).reshape(-1,50,3).astype(float)
        p,_,_ = previous.read(day,raw)
        q,h,n = run(day,raw)
        z = incumbent.decoded(raw,nets)
        # Independently sum the local, shared and incumbent channels of the readout.
        local = np.einsum('tsf,f->ts',z[:,:,:16],weights[:16])
        common = z[:,0,16:]@weights[16:32]
        means = p.mean(1,keepdims=True)
        delta = (local+common[:,None]+((p-means)@weights[32:35])+(means@weights[35:])).astype('float32')
        err = float(abs(p[:,:,2]+delta-q[:,:,2]).max())
        assert err<2e-7 and np.array_equal(p[:,:,:2],q[:,:,:2])
        assert abs(q[:n]-h).max()<1e-4
        # Attribute the difference between the rounded predictions actually scored.
        gram = incumbent.moment(p,y,z)
        flat = np.column_stack((np.ones(y.shape[:2]).ravel(),p[:,:,2].ravel(),
                                (q[:,:,2]-p[:,:,2]).ravel(),*(np.asarray(x).ravel() for x in
                                (y[:,:,0]*incumbent.path.SCALE[2]/incumbent.path.SCALE[0],
                                 y[:,:,1]*base.RATIO-y[:,:,0]*incumbent.path.SCALE[2]/incumbent.path.SCALE[0],
                                 y[:,:,2]-y[:,:,1]*base.RATIO))))
        order = [0,1,3,4,5,2]
        small = (flat.T@flat)[np.ix_(order,order)]
        attribution.append(small)
        change = incumbent.changes(base.covariance(small),np.ones(1))
        np.testing.assert_allclose(change,incumbent.changes(base.covariance(gram),weights),atol=3e-8,rtol=0)
        assert abs(change.sum()/2-report['daily'][str(day+1)]['gain'])<1e-10
        checks[day+1] = dict(readout_max_error=err,y3_contributions=change.tolist())
        rows.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        np.savez(FULL/f'day_{day}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled = trace.older.measure(rows)
    change = incumbent.changes(base.covariance(np.sum(attribution,0)),np.ones(1))
    assert abs(change.sum()/2-pooled['gain'])<1e-10
    assert abs(pooled['score']-report['pooled']['score'])<1e-10 and pooled['rows']==936250
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=checks,y3_contributions=change.tolist()),indent=2))
    print('pooled',pooled,'horizon attribution',change,flush=True)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    grams = np.load(incumbent.OUT/'training_moments.npz')['grams']
    result = json.loads((incumbent.OUT/'results.json').read_text())
    pick = result['selected']
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_array_equal(incumbent.fit(grams)[pick],f['w'])
    daily = {}
    for start in (8,12):
        model = incumbent.path.fit([incumbent.path.data(d) for d in range(start)])
        weights = incumbent.fit(grams[:start-4])[pick]
        for day in range(start,start+4):
            raw = trial.clock.book.rawday(day)
            y = trial.clock.labels(day).reshape(-1,50,3).astype(float)
            ps = previous.references(day,raw)
            z = incumbent.decoded(raw,model)
            q = ps.copy()
            for i,p in enumerate(ps):
                g = incumbent.moment(p,y,z)
                np.testing.assert_allclose(g,grams[day-4,i],atol=1e-7,rtol=1e-10)
                q[i,:,:,2] += (incumbent.channels(z,p)@weights).astype('float32')
            gains = []
            for i,p in enumerate(ps):
                a = (p[:,:,2].ravel(),y[:,:,2].ravel())
                b = (q[i,:,:,2].ravel(),a[1])
                value = np.corrcoef(*b)[0,1]-np.corrcoef(*a)[0,1]
                expected = np.asarray(result['daily'])[day-8,i,pick].sum()
                assert abs(value-expected)<1e-8
                gains.append(float(value))
            daily[day+1] = gains
            np.savez(CACHE/f'day_{day:02d}.npz',forecasts=q,cutoff=day-day%2,calibration_cutoff=start,
                     raw_hash=digest(raw),forecast_hash=digest(q))
        print('historical views through',start+4,flush=True)
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=daily,cutoffs=[8,12],readout=pick),indent=2))



if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as audit
        audit.run(incumbent,read,previous.incumbent,previous.read)
        audit.run(incumbent,read)
