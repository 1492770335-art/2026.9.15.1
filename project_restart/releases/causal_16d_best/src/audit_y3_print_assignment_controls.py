"""Compare exact previous-quote equality against other equality tests for the same single-side frames."""
import json

import numpy as np
import torch

import experiment_y3_inferred_print_impacts_v1 as model

history, book, trial = model.history, model.book, model.trial
NAMES = ('previous quote', 'current quote', 'opposite old quote', 'old quote with mirrored price',
         'eight-tick-old quote', 'adjacent outer quote', 'adjacent inner quote')


def design(raw, p, t):
    state = model.parameters()
    mid, width, price, old, sign, live, sizes = model.prints(raw, state)
    ix = (np.arange(len(mid))-1).clip(0)
    stale = (np.arange(len(mid))-8).clip(0)
    tests = (price==old, price==mid+sign*width, price==mid[ix]-sign*width[ix], (2*mid-price)==old,
             price==mid[stale]+sign*width[stale], price==old+2*sign, price==old-2*sign)
    parts = []
    for gate in (live, *tests):
        z = sizes*gate[:, :, None]
        cs = np.concatenate((np.zeros_like(z[:1]), z.cumsum(0)))
        parts.extend([(cs[t+1]-cs[(t+1-h).clip(0)])/np.sqrt(h) for h in (5, 20, 60, 180)])
    return trial.clock.center(np.concatenate((p[t], *parts), axis=2)).astype('float32')


def run():
    history_rows, folds = [], []
    for d in range(4, 16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1, 50, 3)
        refs = history.references(d, raw) if d>=8 else [model.study.screen.history.market.prior.path.early.load(d, raw)]
        train = np.arange(90+d%21, len(raw)-60, 21)
        val = np.arange(d%4, len(raw)-60, 4)
        t = np.union1d(train, val) if d>=8 else train
        if d>=8 and d%2==0:
            groups = [[], []]
            specs = [(np.arange(11+8*i, 19+8*i), 11, .4) for i in range(len(NAMES))]
            state = book.prev.gram(history_rows, specs)
        new = []
        for j, pred in enumerate(refs):
            x = design(raw, pred, t)
            new.append((x[np.searchsorted(t, train)], model.study.error(pred, y, train)))
            if d>=8:
                delta = book.changes(x[np.searchsorted(t, val)], state)
                groups[j].append(trial.scoring.audit.blockstats(pred[val], y[val], delta).sum(0))
        history_rows.append(tuple(np.concatenate(v) for v in zip(*new)))
        if d>=8 and d%2:
            folds.append([trial.scoring.audit.pooled(sum(v)) for v in groups])
    gains = np.asarray(folds)
    expected = np.asarray(json.loads((model.OUT/'results.json').read_text())['folds'])[:, :, 0, 3]
    np.testing.assert_allclose(gains[:, :, 0], expected, atol=1e-9, rtol=0)
    result = dict(arms=NAMES, chronologically_held_days=list(range(9, 17)), folds=gains.tolist(),
                  mean=gains.mean(0).tolist())
    (model.OUT/'equality_controls.json').write_text(json.dumps(result, indent=2))
    print(np.round(gains.mean(0), 6), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
