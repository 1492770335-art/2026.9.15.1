"""Reconstruct out-of-fold contextual readouts and check the selected receiver-conditioned readout independently."""
import json
import sys
import numpy as np
import torch

import experiment_y3_horizon_agreement_skill_v1 as incumbent
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


def rebuild(mod=incumbent,directory=CACHE):
    directory.mkdir(exist_ok=True)
    report = json.loads((mod.OUT/'results.json').read_text())
    pick = report['selected']
    hist,errors = [],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        preds = mod.history.references(d,raw)
        row = tuple(np.mean(v,axis=0) for v in zip(*[mod.record(p,y) for p in preds]))
        if d>=10:
            if d%2==0:
                st,group = mod.fit(hist),[]
            mats = []
            for p in preds:
                old = p[:,:,2].copy()
                p[:,:,2] += mod.signals(p,st)[:,:,pick]
                v = np.column_stack((np.ones(old.size),old.ravel(),p[:,:,2].ravel(),y[:,:,2].ravel()))
                mats.append(v.T@v)
            group.append(mats)
            if d%2:
                obs = []
                for m in np.sum(group,axis=0):
                    c = stack.covariance(m)
                    r = c[:2,2]/np.sqrt(c.diagonal()[:2]*c[2,2])
                    obs.append(r[1]-r[0])
                errors.append(float(np.max(abs(np.array(report['folds'])[(d-11)//2,:,pick]-obs))))
                assert errors[-1]<1e-8
        np.savez(directory/f'day_{d:02d}.npz',forecasts=preds,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(preds))
        hist.append(row)
    with np.load(mod.OUT/'model.npz') as f:
        for key,v in mod.fit(hist).items():
            np.testing.assert_allclose(v,f[key],rtol=0,atol=1e-9)
    (directory/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def independent(p,st):
    assert int(st['pick'])==5
    risk = st['risk']
    g = risk+.8*np.diag(risk.diagonal().clip(.001))
    z = trial.clock.center(p.astype(float))
    change = np.zeros(p.shape[:2])
    def weights(x):
        v = x/np.sqrt(np.mean(np.square(x),axis=1,keepdims=True)+.003)
        return np.tanh(.65*v[:,:,2]*(v[:,:,0]+v[:,:,1]))
    def state(x,w):
        m = np.einsum('ns,nsi,nsj->nij',w,x,x)/w.sum(1)[:,None,None]
        s = incumbent.study.ema(m,128)
        f = .8*incumbent.study.ema(m,6)+.2*s
        a,b = [np.sqrt(v.diagonal(axis1=1,axis2=2)+.003) for v in (f,s)]
        cor = f/a[:,:,None]/a[:,None,:]-s/b[:,:,None]/b[:,None,:]
        return np.column_stack((np.log(a/b),cor[:,0,1],cor[:,0,2],cor[:,1,2]))
    for k in (0,1):
        x,y = z[:,k::2],z[:,1-k::2]
        gate = .85*weights(x)
        basic = np.column_stack((np.ones(len(p)),state(x,np.ones_like(gate))))
        diff = state(x,1+gate)-state(x,1-gate)
        b = st['both']
        def evaluate(v):
            coef = np.linalg.solve(g,v.T).T
            sd = np.sqrt((coef*(coef@risk)).sum(1)).clip(1e-8)
            return np.einsum('nsi,ni->ns',y,coef)*(.55*np.minimum(1.,.18*np.sqrt(risk[2,2])/sd))[:,None]
        core = evaluate(basic@b[:7,:3]+diff@b[7:,:3])
        short = evaluate(basic@b[:7,3:])
        cross = evaluate(diff@b[7:,3:])
        change[:,1-k::2] = core+weights(y)*(short+cross)
    q = p.copy()
    q[:,:,2] += change.astype('float32')
    return q


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['verify']:
        with np.load(incumbent.OUT/'model.npz') as f:
            st = dict(f)
        verify_stage(incumbent,previous.read,lambda raw,p:independent(p,st))
    elif sys.argv[1:]==['audit']:
        import audit_early_book_horizon_gain_v1 as horizon
        import audit_y3_release_gain_v1 as gain
        base = previous.previous.previous
        gain.run(incumbent,read,base.incumbent,base.read)
        gain.run(incumbent,read)
        horizon.run(sys.modules[__name__],base)
    else:
        rebuild()
