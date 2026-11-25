"""Evaluate independent middle/terminal Y3 heads against the verified category-history model."""
import json
import os

import numpy as np
import torch

import experiment_y3_state_transition_sources_v1 as parent
import y3_terminal_validation as previous

clock, scoring, curve = previous.clock, previous.scoring, previous.curve
CACHE = parent.OUT / 'historical_references'


def prepare():
    hist = []
    CACHE.mkdir(exist_ok=True)
    report = json.loads((parent.OUT / 'results.json').read_text())
    pick = report['selected']
    for d in range(16):
        raw = clock.book.rawday(d)
        y = clock.labels(d).reshape(-1,50,3)
        z = parent.state(raw)
        if d >= 8:
            if d % 2 == 0:
                model, _ = parent.fit(hist)
                stats = [[], []]
            change = parent.changes(z,model)[pick]
            refs = previous.refs(d, raw)
            for j,p in enumerate(refs):
                stats[j].append([v.sum(0) for v in previous.statistics(p,y,[change])])
                p[:,:,2] += change
            path = CACHE / f'day_{d:02d}.npz'
            temp = path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(temp,p=refs,hash=previous.prior.digest(refs),raw_hash=previous.prior.digest(raw),train_end=d-d%2)
            temp.replace(path)
            if d % 2:
                found = [[scoring.audit.pooled(v)[0] for v in np.sum(group,axis=0)] for group in stats]
                np.testing.assert_allclose(found,np.array(report['folds'][(d-9)//2])[:,:,pick],atol=1e-9,rtol=0)
        hist.append(parent.observe(z,y,d))
    (CACHE/'verified.json').write_text(json.dumps(dict(parent=parent.OUT.name,pick=pick)))


def refs(d,raw):
    with np.load(CACHE/f'day_{d:02d}.npz') as f:
        p = f['p']
        assert int(f['train_end'])==d-d%2 and 8<=d<16
        assert f['hash'].item()==previous.prior.digest(p) and f['raw_hash'].item()==previous.prior.digest(raw)
    return p


def targets(y,t):
    coeff = curve.settled.historical_geometry()[1]
    return clock.center(np.stack([y[t+h,:,1] for h in (20,40)],axis=2))*(coeff[2,1]/coeff[1,1])


def statistics(p,y,changes):
    full,tail,last = previous.statistics(p,y,changes)
    t = np.arange(len(p)-60)
    pred,truth = np.zeros_like(p[t]),np.zeros_like(y[t])
    pred[:,:,2] = scoring.late(p[t])
    truth[:,:,2] = targets(y,t)[:,:,0]
    mid = scoring.audit.blockstats(pred,truth,[clock.center(x[t]) for x in changes])
    return full,tail,mid,last


def layer(module):
    old, update = parent.layer(), module.correction()
    def predict(raw,p):
        q=old(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return predict


def run(module):
    module.OUT.mkdir(exist_ok=True)
    if not (CACHE/'verified.json').exists():
        prepare()
    if previous.prior.screen(module,refs):
        scoring.outer(module,reference=parent,preceding=(previous.prior.prior.incumbent,previous.prior.incumbent,previous.incumbent))


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
