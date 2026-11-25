"""Generate rolling forecasts including the retained pair-crossing readout and verify them against the original folds."""
import json
import os

import numpy as np
import torch

import experiment_y3_pair_crossing_premia_v1 as incumbent
import y3_risk_rank_validation as previous

trial = previous.parent
CACHE = incumbent.OUT/'historical_references'


def prepare():
    report = json.loads((incumbent.OUT/'results.json').read_text())
    pick = report['selected']
    assert pick is not None
    CACHE.mkdir(exist_ok=True)
    hist, errors = [], []
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        preds = previous.references(d, raw)
        records = [incumbent.record(p,y,d) for p in preds]
        if d>=10:
            if d%2==0:
                model = incumbent.fit(hist)
                sums = [[],[]]
            for j,p in enumerate(preds):
                change = incumbent.changes_with_reference(None,model,p)[pick]
                sums[j].append([v.sum(0) for v in trial.previous.statistics(p,y,[change])])
                p[:,:,2]+=change
            path = CACHE/f'day_{d:02d}.npz'
            tmp = path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(tmp,p=preds,raw_hash=previous.digest(raw),pred_hash=previous.digest(preds),cutoff=d-d%2)
            tmp.replace(path)
            if d%2:
                measured = [[trial.scoring.audit.pooled(s)[0] for s in np.sum(g,axis=0)] for g in sums]
                expected = np.asarray(report['folds'][(d-11)//2])[:,:,pick]
                errors.append(float(np.max(abs(measured-expected))))
                assert errors[-1]<1e-9
        hist.append([np.mean(a,axis=0) for a in zip(*records)])
    (CACHE/'verified.json').write_text(json.dumps(dict(errors=errors,pick=pick,days=[11,12,13,14,15,16]),indent=2))


def references(d,raw):
    if d<10:
        return previous.references(d,raw)
    assert 10<=d<16 and (CACHE/'verified.json').exists()
    with np.load(CACHE/f'day_{d:02d}.npz') as f:
        p=f['p']
        assert f['cutoff']==d-d%2 and previous.digest(raw)==f['raw_hash'].item() and previous.digest(p)==f['pred_hash'].item()
    return p


def outer(module):
    old=trial.previous.previous
    trial.scoring.outer(module,reference=incumbent,preceding=(old.prior.prior.incumbent,old.prior.incumbent,
                        old.incumbent,trial.previous.parent,trial.incumbent,previous.incumbent))


def run(module):
    if not (CACHE/'verified.json').exists():
        prepare()
    previous.incumbent.screen.run(module,references=references,outer=outer)


def correction(module):
    return previous.correction(module)


def layer(module):
    base, update = incumbent.layer(),module.correction()
    def predict(raw,p):
        q=base(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return predict


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
