"""Use verified rolling predictions of the interval-rank model to select final-twenty specialists."""
import json
import os

import numpy as np
import torch

import experiment_y3_interval_rank_heads_v1 as incumbent
import y3_filter_validation as prior

clock, scoring, curve = prior.clock, prior.scoring, prior.curve
CACHE = incumbent.OUT / 'historical_references'


def prepare():
    expected = json.loads((incumbent.OUT / 'results.json').read_text())
    pick = expected['selected']
    hist = []
    CACHE.mkdir(exist_ok=True)
    for d in range(16):
        raw = clock.book.rawday(d)
        labels = clock.labels(d).reshape(-1, 50, 3)
        z = incumbent.state(raw)
        if d >= 8:
            if d % 2 == 0:
                model, _ = incumbent.fit(hist)
                groups = [[], []]
            delta = incumbent.changes(z, model)[pick]
            refs = prior.refs(d, raw)
            for j, pred in enumerate(refs):
                groups[j].append([m.sum(0) for m in scoring.statistics(pred, labels, [delta])])
                pred[:, :, 2] += delta
            path = CACHE / f'day_{d:02d}.npz'
            temp = path.with_suffix(f'.{os.getpid()}.npz')
            np.savez(temp, p=refs, hash=prior.digest(refs), raw_hash=prior.digest(raw), train_end=d-d%2)
            temp.replace(path)
            if d % 2:
                found = [[scoring.audit.pooled(m)[0] for m in np.sum(g, axis=0)] for g in groups]
                wanted = np.asarray(expected['folds'][(d-9)//2])[:, :, pick]
                np.testing.assert_allclose(found, wanted, atol=1e-9, rtol=0)
        hist.append(incumbent.observe(z, labels, d))
    (CACHE/'verified.json').write_text(json.dumps(dict(source=incumbent.OUT.name, pick=pick, dates=list(range(9,17)))))


def refs(day, raw):
    assert 8 <= day < 16 and (CACHE/'verified.json').is_file()
    with np.load(CACHE/f'day_{day:02d}.npz') as f:
        p = f['p']
        assert f['hash'].item() == prior.digest(p) and f['raw_hash'].item() == prior.digest(raw)
        assert int(f['train_end']) == day-day%2
        return p


def statistics(p, y, changes):
    full, tail = scoring.statistics(p, y, changes)
    t = np.arange(len(y)-60)
    coeff = curve.settled.historical_geometry()[1]
    late, truth = np.zeros_like(p[t]), np.zeros_like(y[t])
    late[:, :, 2] = scoring.late(p[t])
    truth[:, :, 2] = clock.center(y[t+40, :, 1]) * (coeff[2,1]/coeff[1,1])
    final = scoring.audit.blockstats(late, truth, [clock.center(a[t]) for a in changes])
    return full, tail, final


def layer(module):
    parent, update = incumbent.layer(), module.correction()
    def predict(raw, p):
        q = parent(raw, p)
        q[:, :, 2] += update(raw, q)
        return q
    return predict


def run(module):
    module.OUT.mkdir(exist_ok=True)
    if not (CACHE/'verified.json').is_file():
        prepare()
    if prior.screen(module, refs):
        scoring.outer(module, reference=incumbent, preceding=(prior.prior.incumbent, prior.incumbent))


if __name__ == '__main__':
    torch.set_num_threads(2)
    prepare()
