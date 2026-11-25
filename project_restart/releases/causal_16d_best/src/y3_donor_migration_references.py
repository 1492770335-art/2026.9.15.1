"""Independently compute disjoint donor prices and reconstruct expanding-window predictions."""
import json
import sys

import numpy as np
import torch

import experiment_y3_donor_migration_prices_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_touch_clock_references import digest,read_stage

previous = incumbent.history
base,trial,study,stack = previous.base,previous.trial,previous.study,previous.stack
FULL = incumbent.OUT/'full_chain_verification'
CACHE = incumbent.OUT/'historical_references'


def read(day,raw):
    return read_stage(FULL,day,raw)


def references(day,raw):
    return read_verified(CACHE,day,raw)


def group_prices(a,b,forecast):
    result = np.zeros_like(forecast,dtype=float)
    for partition in (np.arange(50),np.random.default_rng(62017).permutation(50)):
        for side in (0,1):
            src,dst = partition[side::2],partition[1-side::2]
            donor = forecast[:,src]-forecast[:,src].mean(1,keepdims=True)
            receiver = np.zeros((len(a),len(dst)))
            for k in range(4):
                origin,destination = a[:,k],b[:,k]
                sender = origin[:,src].transpose(0,2,1)
                cells = sender@destination[:,src]
                price = ((sender*donor[:,None,:])@destination[:,src])/(2+cells)
                target = origin[:,dst]
                freq = (target.transpose(0,2,1)@destination[:,dst])/target.sum(1)[:,:,None].clip(.001)
                # Center the destination probability separately for each origin and recipient panel.
                relative = destination[:,dst,None,:]-freq[:,None,:,:]
                receiver += (target[:,:,:,None]*relative*price[:,None,:,:]).sum((2,3))
            result[:,dst] += receiver/8
    return result-result.mean(1,keepdims=True)


@torch.inference_mode()
def independent(raw,p,net,weight,check=False):
    x,ph = incumbent.rank.inputs.features(raw,np.arange(len(raw)))
    chunks = []
    for i in range(0,len(raw),64):
        v,s = [torch.from_numpy(z[i:i+64]).cuda() for z in (x,ph)]
        a = incumbent.migration.groups(v)
        b = net(v,s,a)[0]
        a,b = [z.cpu().numpy().astype(float) for z in (a,b)]
        forecast = p[i:i+64,:,2].astype(float)
        value = group_prices(a,b,forecast)
        if check and i==0:
            altered = forecast.copy()
            altered[:,17] += 1.
            used = group_prices(a,b,altered)
            np.testing.assert_allclose(value[:,17],used[:,17],rtol=0,atol=3e-8)
            assert abs(used-value).max()>1e-5
        chunks.append(value)
    return (weight*np.concatenate(chunks)).astype('float32')


def verify():
    FULL.mkdir(exist_ok=True)
    expected = json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']
    with np.load(incumbent.OUT/'model.npz') as f:
        assert incumbent.ARMS[int(f['pick'])+1]=='full return donor allocation'
        w = float(f['w'])
        net = incumbent.migration.Migration(int(f['width'])).cuda().eval()
        net.load_state_dict({k:torch.from_numpy(f['1_'+k]) for k in net.state_dict()})
    reader = incumbent.replay_with()[1]
    rows,daily = [],{}
    for d in range(16,20):
        raw = trial.clock.book.rawday(d)
        p,_,_ = previous.read(d,raw)
        q,h,n = reader(d,raw)
        alt = p[:,:,2]+independent(raw,p,net,w,check=d==16)
        err = float(abs(alt-q[:,:,2]).max())
        assert err<2e-6 and abs(h-q[:n]).max()<1e-4 and np.array_equal(q[:,:,:2],p[:,:,:2])
        row = p.reshape(-1,3),q.reshape(-1,3),stack.verifier.trace.replay.labels(d)
        rows.append(row)
        daily[d+1] = dict(independent_error=err,score=stack.verifier.trace.older.measure([row])['score'])
        np.savez(FULL/f'day_{d}.npz',full=q,head=h,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(h))
    pooled = stack.verifier.trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-expected['score'])<1e-12
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = report['selected']-1
    source,cal,checks = [],[],{}
    for d in range(16):
        if d in (4,8,12):
            nets = incumbent.migration.fit(source)
            if d>=8:
                weight = base.calibrate(cal)[pick]
        if d>=4:
            raw = trial.clock.book.rawday(d)
            y = trial.clock.labels(d).reshape(-1,50,3)
            ps = previous.references(d,raw) if d>=8 else incumbent.parent.training_references(d,raw)
            x,ph = incumbent.rank.inputs.features(raw,np.arange(len(raw)))
            zs = incumbent.panels(x,ph,ps,nets)
            cal.append([base.records(p,y,z) for p,z in zip(ps,zs)])
            if d>=8:
                qs = ps.copy()
                gains = []
                for q,p,z in zip(qs,ps,zs):
                    q[:,:,2] += (weight*z[:,:,pick]).astype('float32')
                    gains.append(np.corrcoef(q[:,:,2].ravel(),y[:,:,2].ravel())[0,1]-
                                 np.corrcoef(p[:,:,2].ravel(),y[:,:,2].ravel())[0,1])
                np.testing.assert_allclose(gains,np.asarray(report['daily'])[d-8,:,pick+1],atol=1e-8,rtol=0)
                checks[d+1] = gains
                np.savez(CACHE/f'day_{d:02d}.npz',forecasts=qs,cutoff=d-d%2,calibration_cutoff=d-d%4,
                         raw_hash=digest(raw),forecast_hash=digest(qs))
        source.append(incumbent.migration.samples(d))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(base.calibrate(cal)[pick],f['w'],atol=1e-9,rtol=0)
    (CACHE/'verified.json').write_text(json.dumps(dict(daily=checks),indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['history']:
        rebuild()
    else:
        verify()
        import audit_y3_release_gain_v1 as release
        release.run(incumbent,read,previous.incumbent,previous.read)
        release.run(incumbent,read)
