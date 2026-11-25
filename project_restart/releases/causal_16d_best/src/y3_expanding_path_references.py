"""Rebuild rolling and outer references after the dense-path forecast correction."""
import json

import numpy as np

import cache_y3_context_reference as cached
import experiment_y3_expanding_path_complement_v1 as incumbent

previous = incumbent.history
CACHE = incumbent.OUT / 'historical_references'


def rebuild(model, parent_references, directory):
    report = json.loads((model.OUT / 'results.json').read_text())
    pick = report['selected']
    directory.mkdir(exist_ok=True)
    history = model.warm_history()
    errors = []
    for day in range(8, 16):
        raw = model.trial.clock.book.rawday(day)
        labels = model.trial.clock.labels(day).reshape(-1, 50, 3)
        state = model.historical_state(raw, day)
        forecasts = parent_references(day, raw)
        if day % 2 == 0:
            readout = model.fit(history)
            folds = [[], []]
        records = []
        for index, forecast in enumerate(forecasts):
            records.append(model.record_with_state(state, forecast, labels, day))
            delta = model.changes_with_reference(state, readout, forecast)[pick]
            folds[index].append([s.sum(0) for s in model.trial.previous.statistics(forecast, labels, [delta])])
            forecast[:, :, 2] += delta
        if day % 2:
            pooled = [[model.trial.scoring.audit.pooled(s)[0] for s in np.sum(f, 0)] for f in folds]
            expected = np.array(report['folds'][(day - 9) // 2])[:, :, pick]
            errors.append(float(np.max(abs(pooled - expected))))
            assert errors[-1] < 1e-9
        history.append([np.mean(values, axis=0) for values in zip(*records)])
        np.savez(directory / f'day_{day:02d}.npz', forecasts=forecasts, cutoff=day-day%2,
                 raw_hash=cached.digest(raw), forecast_hash=cached.digest(forecasts))
    (directory / 'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def read_verified(directory, day, raw):
    assert 8 <= day < 16 and (directory / 'verified.json').exists()
    with np.load(directory / f'day_{day:02d}.npz') as data:
        forecasts = data['forecasts']
        assert int(data['cutoff']) == day-day%2 and data['raw_hash'].item() == cached.digest(raw)
        assert data['forecast_hash'].item() == cached.digest(forecasts)
    return forecasts


def prepare():
    rebuild(incumbent, previous.references, CACHE)


def references(day, raw):
    return read_verified(CACHE, day, raw)


def outer(module):
    correct = incumbent.correction()

    def parent(day, raw):
        full, head, n = cached.load(day, raw)
        full[:, :, 2] += correct(raw, full)
        head[:, :, 2] += correct(raw[:n], head)
        return full, head, n

    cached.scoring.outer(module, reference=incumbent, load_reference=parent)


def run(module):
    if not (CACHE / 'verified.json').exists():
        prepare()
    incumbent.history.previous.previous.previous.incumbent.screen.run(module, references, outer)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    prepare()
