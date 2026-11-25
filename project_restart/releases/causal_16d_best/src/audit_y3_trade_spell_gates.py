"""Compare execution-run conditioning while giving every control the same near-forecast interaction."""
import json

import numpy as np
import torch

import experiment_y3_trade_spell_survival_v1 as model

study, trial, book = model.study, model.trial, model.book
NAMES = ('run length and speed', 'age and activity only', 'other stock table cell', 'constant flow',
         'run features without near forecast', 'near forecast gate only', 'near forecast of another stock')


def design(raw, p, t, params):
    action, sign, alive, broad, group, length = model.state(raw, params)
    orient = (sign[t]*alive[t])[:, :, None]
    cs = np.vstack((np.zeros(50), action.cumsum(0)))
    flow = np.stack([(cs[t+1]-cs[(t+1-h).clip(0)])/np.sqrt(h) for h in (5, 20, 60)], axis=2)
    base = (p[t], flow, np.log1p(length[t, :, None])*orient)
    gate = np.tanh(sign[t]*p[t, :, 0])[:, :, None]
    other = np.tanh(sign[t]*np.roll(p[t, :, 0], 17, axis=1))[:, :, None]
    full = params['cells'][group[t], :4]*orient
    coarse = params['broad'][broad[t], :4]*orient
    wrong = params['cells'][np.roll(group[t], 17, axis=1), :4]*orient
    constant = np.broadcast_to(orient, full.shape)
    variants = ((full, full*gate), (coarse, coarse*gate), (wrong, wrong*gate), (constant, constant*gate),
                (full,), (full*gate,), (full, full*other))
    return trial.clock.center(np.concatenate((*base, *(np.concatenate(pair, axis=2) for pair in variants)), 2)).astype('float32')


def run():
    records, folds = [], []
    for d in range(4, 16):
        params = model.parameters(max(4, d-d%2))
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1, 50, 3)
        refs = model.history.references(d, raw) if d>=8 else [study.screen.history.market.prior.path.early.load(d, raw)]
        train, val = np.arange(90+d%21, len(raw)-60, 21), np.arange(d%4, len(raw)-60, 4)
        t = np.union1d(train, val) if d>=8 else train
        if d>=8 and d%2==0:
            groups = [[], []]
            widths = (8, 8, 8, 8, 4, 4, 8)
            cols = np.split(np.arange(7, 7+sum(widths)), np.cumsum(widths)[:-1])
            fitted = book.prev.gram(records, [(g, 7, .4) for g in cols])
        rec = []
        for j, pred in enumerate(refs):
            x = design(raw, pred, t, params)
            rec.append((x[np.searchsorted(t, train)], study.error(pred, y, train)))
            if d>=8:
                delta = book.changes(x[np.searchsorted(t, val)], fitted)
                groups[j].append(trial.scoring.audit.blockstats(pred[val], y[val], delta).sum(0))
        records.append(tuple(np.concatenate(v) for v in zip(*rec)))
        if d>=8 and d%2:
            folds.append([trial.scoring.audit.pooled(sum(v)) for v in groups])
    gains = np.asarray(folds)
    expected = np.asarray(json.loads((model.OUT/'results.json').read_text())['folds'])[:, :, 0, 4]
    # Adding control columns changes float32 BLAS reduction order. Check agreement in correlation units.
    error = float(np.max(abs(gains[:, :, 0]-expected)))
    np.testing.assert_allclose(gains[:, :, 0], expected, atol=3e-8, rtol=0)
    report = dict(days=list(range(9, 17)), arms=NAMES, folds=gains.tolist(), mean=gains.mean(0).tolist(), original_max_error=error)
    (model.OUT/'matched_gate_controls.json').write_text(json.dumps(report, indent=2))
    print(np.round(gains.mean(0), 6), flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
