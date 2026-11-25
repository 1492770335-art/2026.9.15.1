"""Replay chronological predictions from the two-sided path encoder with the readout fit before each test pair of dates."""
import json
import os

import numpy as np
import torch

import experiment_y3_two_sided_path_encoder_v1 as incumbent
import y3_crossing_validation as previous

trial=previous.trial
CACHE=incumbent.OUT/'historical_references'
digest=previous.previous.digest


def prepare():
    report=json.loads((incumbent.OUT/'results.json').read_text())
    pick=report['selected']
    assert pick is not None
    with np.load(incumbent.OUT/'model.npz') as f:
        early_weights={k:f[k] for k in f.files if '/' in k}
    CACHE.mkdir(exist_ok=True)
    hist,errors=[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        z=incumbent.encode(raw,early_weights)
        y=trial.clock.labels(d).reshape(-1,50,3)
        preds=np.stack([incumbent.early.load(d,raw)]) if d<8 else previous.references(d,raw)
        records=[incumbent.record_with_state(z,p,y,d) for p in preds]
        if d>=8:
            if d%2==0:
                model=dict(w=incumbent.readout(hist))
                groups=[[],[]]
            for j,p in enumerate(preds):
                delta=incumbent.changes_with_reference(z,model,p)[pick]
                groups[j].append([v.sum(0) for v in trial.previous.statistics(p,y,[delta])])
                p[:,:,2]+=delta
            path=CACHE/f'day_{d:02d}.npz'
            tmp=path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,p=preds,cutoff=d-d%2,raw_hash=digest(raw),pred_hash=digest(preds))
            tmp.replace(path)
            if d%2:
                found=[[trial.scoring.audit.pooled(s)[0] for s in np.sum(g,0)] for g in groups]
                expected=np.asarray(report['folds'][(d-9)//2])[:,:,pick]
                errors.append(float(np.max(abs(found-expected))))
                assert errors[-1]<1e-9
        hist.append([np.mean(v,0) for v in zip(*records)])
    (CACHE/'verified.json').write_text(json.dumps(dict(pick=pick,dates=list(range(9,17)),errors=errors),indent=2))


def references(d,raw):
    assert 8<=d<16 and (CACHE/'verified.json').exists()
    with np.load(CACHE/f'day_{d:02d}.npz') as f:
        p=f['p']
        assert int(f['cutoff'])==d-d%2 and digest(raw)==f['raw_hash'].item() and digest(p)==f['pred_hash'].item()
    return p


def outer(module):
    old=trial.previous.previous
    trial.scoring.outer(module, reference=incumbent, preceding=(old.prior.prior.incumbent,old.prior.incumbent,
        old.incumbent,trial.previous.parent,trial.incumbent,previous.previous.incumbent,previous.incumbent))


def run(module):
    if not (CACHE/'verified.json').exists():
        prepare()
    previous.previous.incumbent.screen.run(module,references=references,outer=outer)


def correction(module):
    return previous.correction(module)


def layer(module):
    base,update=incumbent.layer(),module.correction()
    def predict(raw,p):
        q=base(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return predict


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
