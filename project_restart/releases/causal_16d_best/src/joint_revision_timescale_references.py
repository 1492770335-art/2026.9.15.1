"""Verify the three revised forecasts and reconstruct chronological forecasts for subsequent experiments."""
import json
import sys
import numpy as np
import torch

import experiment_joint_revision_timescale_transfer_v1 as incumbent
from y3_touch_clock_references import digest,read_stage
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


def independent(anchors,w):
    p = anchors[-1]
    v = np.diff(np.stack([a[:,:,2] for a in anchors],axis=2),axis=2).astype(float)
    def average(span):
        out = np.empty_like(v)
        rho = np.exp(-1/span)
        for i in range(len(v)):
            out[i] = v[i] if i in (0,2401) else (1-rho)*v[i]+rho*out[i-1]
        return out
    a,b = average(6),average(64)
    z = p.astype(float)-p.mean(axis=1,keepdims=True,dtype=float)
    scale = np.sqrt(.003+np.mean(z*z,axis=1,keepdims=True))
    u = z/scale
    gate = np.tanh(.5*(u[:,:,0]+u[:,:,1])*u[:,:,2])
    # Selected models only use columns 3..14 for Y1/Y3 and 9..11 for Y2.
    assert not w[:,1:4].any() and not w[:,16:].any()
    parts = (v-a,a-b,b,v*gate[:,:,None])
    q = p.copy()
    for k in range(3):
        adj = p[:,:,k].astype(float)*w[k,0]
        for j,part in enumerate(parts):
            adj += np.einsum('tsc,c->ts',part.astype('float32'),w[k,4+3*j:7+3*j])
        q[:,:,k] += adj.astype('float32')
    return q


def rebuild():
    CACHE.mkdir(exist_ok=True)
    report = json.loads((incumbent.OUT/'results.json').read_text())
    sel = report['selected']
    rows,errors = [],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        ref = [h.references(d,raw) for h in incumbent.CHECKPOINTS]
        preds = ref[-1].copy()
        if d>=10:
            if d%2==0:
                w = np.stack([incumbent.fit(rows,k)[1][i] for k,i in enumerate(sel)])
                group = []
            mats = []
            for j,anchors in enumerate(zip(*ref)):
                q = independent(anchors,w)
                v = np.concatenate((np.ones_like(y[:,:,:1]),anchors[-1],q,y),axis=2).reshape(-1,10).astype(float)
                mats.append(v.T@v)
                preds[j] = q
            group.append(mats)
            if d%2:
                obs = []
                for m in np.sum(group,axis=0):
                    c = stack.covariance(m)
                    sd = np.sqrt(c.diagonal())
                    corr = c/sd[:,None]/sd[None,:]
                    obs.append(np.diag(corr[3:6,6:]-corr[:3,6:]))
                target = [np.asarray(r['folds'])[(d-11)//2,:,i] for r,i in zip(report['targets'],sel)]
                errors.append(float(abs(np.asarray(obs)-np.stack(target,axis=1)).max()))
                assert errors[-1]<1e-8
        np.savez(CACHE/f'day_{d:02d}.npz',forecasts=preds,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(preds))
        rows.append(incumbent.records(d))
    with np.load(incumbent.OUT/'model.npz') as f:
        np.testing.assert_allclose(f['w'],np.stack([incumbent.fit(rows,k)[1][i] for k,i in enumerate(sel)]),rtol=0,atol=1e-10)
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def verify():
    FULL.mkdir(exist_ok=True)
    trace = stack.verifier.trace
    report = json.loads((incumbent.OUT/'reload_verification.json').read_text())
    with np.load(incumbent.OUT/'model.npz') as f:
        w = f['w']
    apply = incumbent.adjustment()
    rebuild_snapshots = incumbent.snapshot_layers()
    rows,daily = [],{}
    for d in range(16,20):
        raw = trace.replay.original.features(trace.replay.original.DATA,d)
        states = [h.read(d,raw) for h in incumbent.CHECKPOINTS]
        n = states[0][2]
        result = []
        for j,grid in enumerate((raw,raw[:n])):
            anchors = [s[j] for s in states]
            obtained = rebuild_snapshots(grid,anchors[0])
            for actual,expected in zip(obtained,anchors):
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
            q = apply(anchors)
            np.testing.assert_allclose(independent(anchors,w),q,rtol=0,atol=1e-6)
            np.testing.assert_allclose(apply(obtained),q,rtol=0,atol=1e-6)
            result.append(q)
        q,head = result
        err = float(abs(q[:n]-head).max())
        assert err<1e-4
        row = (states[-1][0].reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        rows.append(row)
        stat = trace.older.measure([row])
        assert abs(stat['score']-report['daily'][str(d+1)]['score'])<1e-10
        daily[d+1] = dict(score=stat['score'],prefix_error=err)
        np.savez(FULL/f'day_{d}.npz',full=q,head=head,raw_hash=digest(raw),full_hash=digest(q),head_hash=digest(head))
    pooled = trace.older.measure(rows)
    assert pooled['rows']==936250 and abs(pooled['score']-report['pooled']['score'])<1e-10
    (FULL/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled,daily=daily),indent=2))
    print('verified',pooled['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        verify()
    elif sys.argv[1:]==['audit']:
        import audit_early_book_horizon_gain_v1 as horizon
        import audit_y3_release_gain_v1 as gain
        gain.run(incumbent,read,previous.incumbent,previous.read)
        gain.run(incumbent,read)
        horizon.run(sys.modules[__name__],previous)
    else:
        rebuild()
