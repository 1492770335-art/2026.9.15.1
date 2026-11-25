"""Fit and verify chronological reference predictions for the retained pair-distribution model."""
import json
import os

import numpy as np
import torch

import experiment_y3_honest_pair_distribution_v1 as incumbent
import y3_disjoint_validation as previous

clock, scoring, curve = previous.clock, previous.scoring, previous.curve
CACHE = incumbent.OUT / 'historical_references'


def prepare():
    report = json.loads((incumbent.OUT/'results.json').read_text())
    chosen = report['selected']
    hist = incumbent.warm_history()
    CACHE.mkdir(exist_ok=True)
    errors = []
    for d in range(8, 16):
        raw = clock.book.rawday(d)
        y = clock.labels(d).reshape(-1, 50, 3)
        z = incumbent.state(raw)
        refs = previous.refs(d, raw)
        if d%2 == 0:
            model = incumbent.fit(hist)
            folds = [[], []]
        records = []
        for j, p in enumerate(refs):
            update = incumbent.changes_with_reference(z, model, p)[chosen]
            records.append(incumbent.record(z, p, y, d))
            folds[j].append([a.sum(0) for a in previous.statistics(p, y, [update])])
            p[:, :, 2] += update
        hist.append(incumbent.merge_records(records))
        path = CACHE/f'day_{d:02d}.npz'
        temp = path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(temp, p=refs, raw_hash=previous.previous.prior.digest(raw), hash=previous.previous.prior.digest(refs),
                 train_end=d-d%2)
        temp.replace(path)
        if d%2:
            measured = [[scoring.audit.pooled(m)[0] for m in np.sum(v, axis=0)] for v in folds]
            wanted = np.array(report['folds'][(d-9)//2])[:, :, chosen]
            errors.append(float(np.max(abs(measured-wanted))))
            assert errors[-1]<1e-9
            print('verified dates to', d+1, errors[-1], flush=True)
    (CACHE/'verified.json').write_text(json.dumps(dict(selected=chosen, errors=errors, dates=list(range(9, 17)))))


def refs(d, raw):
    assert 8<=d<16 and (CACHE/'verified.json').is_file()
    with np.load(CACHE/f'day_{d:02d}.npz') as f:
        p = f['p']
        assert int(f['train_end'])==d-d%2 and f['raw_hash'].item()==previous.previous.prior.digest(raw)
        assert f['hash'].item()==previous.previous.prior.digest(p)
    return p


def layer(module):
    base, correction = incumbent.layer(), module.correction()
    def predict(raw, p):
        result = base(raw, p)
        result[:, :, 2] += correction(raw, result)
        return result
    return predict


def run(module):
    module.OUT.mkdir(exist_ok=True)
    if not (CACHE/'verified.json').is_file():
        prepare()
    if previous.previous.prior.screen(module, refs):
        p = previous.previous
        scoring.outer(module, reference=incumbent, preceding=(p.prior.prior.incumbent, p.prior.incumbent,
                                                              p.incumbent, previous.parent))


if __name__ == '__main__':
    torch.set_num_threads(2)
    prepare()
