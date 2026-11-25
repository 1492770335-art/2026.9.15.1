"""Verify the selected probability contribution and rebuild the training-only panel forecasts."""
import json
import sys

import numpy as np
import torch

import experiment_y3_joint_rank_payout_v1 as incumbent
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


@torch.inference_mode()
def independent(raw):
    with np.load(incumbent.OUT/'model.npz') as f:
        assert incumbent.ARMS[int(f['pick'])+1]=='balanced probability contribution'
        nets = [incumbent.Model(int(f['width']),i).cuda().eval() for i in (1,2)]
        for i,net in enumerate(nets,1):
            net.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in net.state_dict()})
        coefficient = float(f['w'])
    x,ph = incumbent.inputs.features(raw,np.arange(len(raw)))
    output = []
    capacity_error = 0.
    for start in range(0,len(x),96):
        v = torch.from_numpy(x[start:start+96]).cuda()
        s = torch.from_numpy(ph[start:start+96]).cuda()
        free,a,p = nets[0](v,s)
        _,b,q = nets[1](v,s)
        capacity_error = max(capacity_error,(b.sum(1)-5).abs().max().item())
        # Change only memberships once at each endpoint of the payout change.
        at_free = (b*p[:,None]).sum(-1)-free
        at_balanced = ((b*q[:,None]).sum(-1)-(a*q[:,None]).sum(-1))
        output.append((coefficient*.5*(at_free[:,:,2]+at_balanced[:,:,2])).cpu().numpy())
    return np.concatenate(output),capacity_error


def verify():
    FULL.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    trace = stack.verifier.trace
    reader = incumbent.replay_with()[1]
    rows,check = [],{}
    for d in range(16,20):
        raw = trial.clock.book.rawday(d)
        y = trace.replay.labels(d)
        q,h,n = reader(d,raw)
        p,_,_ = previous.read(d,raw)
        change,cap = independent(raw)
        err = float(abs(p[:,:,2]+change-q[:,:,2]).max())
        assert err<5e-6 and abs(q[:n]-h).max()<1e-4 and np.array_equal(p[:,:,:2],q[:,:,:2])
        row = p.reshape(-1,3),q.reshape(-1,3),y
        rows.append(row)
        score = trace.older.measure([row])['score']
        assert abs(score-report['daily'][str(d+1)]['score'])<1e-10
        check[d+1] = dict(score=score,independent_readout_error=err,max_bin_capacity_error=cap)
        np.savez(FULL/f'day_{d}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled = trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-report['pooled']['score'])<1e-12
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=check),indent=2))
    print('independent score',pooled['score'],flush=True)


def rebuild():
    CACHE.mkdir(exist_ok=True)
    info = json.loads((incumbent.OUT/'results.json').read_text())
    pick = info['selected']-1
    records,rows,checks = [],[],{}
    for day in range(16):
        if day in (4,8,12):
            nets = incumbent.fit(rows)
            if day>=8:
                weight = base.calibrate(records)[pick]
        if day>=4:
            raw = trial.clock.book.rawday(day)
            z = incumbent.signals(raw,nets)
            y = trial.clock.labels(day).reshape(-1,50,3)
            ps = previous.references(day,raw) if day>=8 else previous.previous.incumbent.parent.warm.read(day,raw)
            records.append([base.records(p,y,z) for p in ps])
            if day>=8:
                q = ps.copy()
                q[:,:,:,2] += (z[:,:,pick]*weight).astype('float32')
                differences = []
                for old,new in zip(ps,q):
                    differences.append(float(np.corrcoef(new[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-
                                             np.corrcoef(old[:,:,2].ravel(),y[:,:,2].ravel())[0,1]))
                np.testing.assert_allclose(differences,np.asarray(info['daily'])[day-8,:,0,pick+1],atol=1e-8,rtol=0)
                checks[day+1] = differences
                np.savez(CACHE/f'day_{day:02d}.npz',forecasts=q,cutoff=day-day%2,calibration_cutoff=day-day%4,
                         raw_hash=digest(raw),forecast_hash=digest(q))
        rows.append(incumbent.samples(day))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(base.calibrate(records)[pick],f['w'],atol=1e-10,rtol=0)
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=checks,selected=pick+1),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as audit
        audit.run(incumbent,read,previous.incumbent,previous.read)
        audit.run(incumbent,read)
