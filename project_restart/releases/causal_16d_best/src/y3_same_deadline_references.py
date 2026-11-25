"""Reconstruct rolling training predictions after the same-deadline Y3 fit."""
import json

import numpy as np
import torch

import experiment_y3_same_deadline_event_curve_v1 as incumbent

previous, study, trial = incumbent.parent, incumbent.study, incumbent.trial
CACHE = incumbent.OUT / 'historical_references'


def rebuild(model, reference_loader, directory):
    result = json.loads((model.OUT / 'results.json').read_text())
    pick = result['selected']
    directory.mkdir(exist_ok=True)
    records, errors = [], []
    target_error = getattr(model, 'training_error', study.error)
    for day in range(4, 16):
        raw = trial.clock.book.rawday(day)
        y = trial.clock.labels(day).reshape(-1, 50, 3)
        train = np.arange(90 + day % 21, len(raw) - 60, 21)
        evaluate = np.arange(day % 4, len(raw) - 60, 4)
        refs = (reference_loader(day, raw) if day >= 8 else
                [study.screen.history.market.prior.path.early.load(day, raw)])
        if day >= 8 and day % 2 == 0:
            state = model.fit(records)
            groups = [[], []]
        rec = []
        design = model.historical_design(day) if hasattr(model, "historical_design") else model.design
        for j, p in enumerate(refs):
            x = design(raw, p, np.arange(len(raw)) if day >= 8 else train)
            rec.append((x[train] if day >= 8 else x, target_error(p, y, train)))
            if day >= 8:
                delta = model.changes(x, state)[pick]
                stats = trial.scoring.audit.blockstats(p[evaluate], y[evaluate], [delta[evaluate]])
                groups[j].append(stats.sum(0))
                p[:, :, 2] += delta
        records.append(tuple(np.concatenate(a) for a in zip(*rec)))
        if day >= 8:
            digest = previous.previous.previous.cache.cached.digest
            np.savez(directory / f'day_{day:02d}.npz', forecasts=refs, cutoff=day-day%2,
                     raw_hash=digest(raw), forecast_hash=digest(refs))
            if day % 2:
                actual = [trial.scoring.audit.pooled(np.sum(group, axis=0))[0] for group in groups]
                expected = np.asarray(result['folds'][(day-9)//2])[:, 0, pick]
                errors.append(float(np.max(np.abs(expected-actual))))
                assert errors[-1] < 1e-9, errors
    (directory / 'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def prepare():
    rebuild(incumbent, previous.references, CACHE)


def references(day, raw):
    return previous.previous.previous.cache.history.rolling.read_verified(CACHE, day, raw)


def outer(module):
    update = previous.incumbent.correction()
    final = incumbent.correction()
    cache = previous.previous.cache
    def load(day, raw):
        p, head, n = cache.load(day, raw)
        for func in (update, final):
            p[:, :, 2] += func(raw, p)
            head[:, :, 2] += func(raw[:n], head)
        return p, head, n
    cache.scoring.outer(module, reference=incumbent, load_reference=load)


if __name__ == '__main__':
    torch.set_num_threads(2)
    prepare()
