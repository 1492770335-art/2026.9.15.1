"""Build an expanding-window reference with separate market and stock expert weights."""
import json

import numpy as np

import experiment_crossfit_error_precision_v1 as gls
import experiment_cutoff_reference_panel_v1 as panel
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

OUT = ART / 'rolling_expert_reference_v1'
CORE = panel.experts.experts.SOURCES
EXTRA = gls.family.MODELS


def prior():
    w = np.zeros((2, 3, 1 + len(CORE) + len(EXTRA)))
    w[:, :, 0] = .65
    w[0, :, [1, 2, 4]] = .35 / 3
    w[1, :, 1:5] = .35 / 4
    return w


def sources(day):
    opening = ART / 'opening_oof_cache_v1' / f'day_{day:02d}.npy'
    paths = ([opening] + [ART / name / f'side_{day}.npy' for name in CORE]
             + [gls.family.side_path(m, day) for m in EXTRA])
    return [np.load(path, mmap_mode='r').reshape(-1, 50, 3) for path in paths]


def components(forecasts, t):
    x = np.stack([f[t] for f in forecasts], axis=-1)
    market = x.mean(axis=1, keepdims=True)
    return market, x - market


def predict(forecasts, t, weights):
    parts = components(forecasts, t)
    return sum(np.einsum('tskn,kn->tsk', part, w.astype('float32'))
               for part, w in zip(parts, weights))


def moments(forecasts, y, t):
    market, local = components(forecasts, t)
    target = y[t].astype(float)
    mean = target.mean(axis=1, keepdims=True)
    stats = []
    for x, b, w in zip((market, local), (mean, target-mean), prior()):
        x = x.reshape(-1, 3, x.shape[-1]).astype(float)
        b = b.reshape(-1, 3)
        x -= x.mean(0)
        b = b - b.mean(0)
        ref = np.einsum('rkn,kn->rk', x, w)
        # Set the label units using only this completed training day's predictions.
        scale = ((ref*b).mean(0) / np.maximum((ref*ref).mean(0), 1e-10)).clip(.3, 3.)
        gram = np.einsum('rki,rkj->kij', x, x) / len(x)
        rhs = np.einsum('rki,rk->ki', x, b/scale) / len(x)
        stats.append((gram, rhs))
    return stats


def simplex(v):
    ordered = np.sort(v)[::-1]
    levels = (np.cumsum(ordered)-1) / np.arange(1, len(v)+1)
    threshold = levels[np.flatnonzero(ordered > levels)[-1]]
    return np.maximum(v-threshold, 0.)


def fit(history):
    anchor = prior()
    result = anchor.copy()
    if not history:
        return result
    for part in (0, 1):
        grams = np.mean([row[part][0] for row in history], axis=0)
        rhs = np.mean([row[part][1] for row in history], axis=0)
        for k, (gram, b) in enumerate(zip(grams, rhs)):
            # The spread decoder lacks a market head.
            live = np.arange(len(b))
            if part == 0:
                live = live[live != 3]
            g = gram[np.ix_(live, live)]
            reg = .18 * np.median(g.diagonal())
            h = g + reg * np.eye(len(g))
            r = b[live] + reg*anchor[part, k, live]
            step = 1/max(np.linalg.eigvalsh(h)[-1], 1e-10)
            a = anchor[part, k, live].copy()
            for _ in range(500):
                updated = simplex(a-step*(h@a-r))
                if abs(updated-a).max() < 1e-8:
                    break
                a = updated
            result[part, k, live] = a
    return result


def main():
    OUT.mkdir(exist_ok=True)
    history, folds, daily, calibrated_rows, base_rows = [], [], [], [], []
    for end in (8, 10, 12, 14):
        w = fit(history)
        retrofit = [gls.fit(rows, True) for rows in (base_rows, calibrated_rows)] if history else []
        block = [[], [], []]
        for d in (end, end+1):
            experts = sources(d)
            y = labels(d).reshape(-1, 50, 3)
            t = np.arange(len(y))
            base = np.load(panel.OUT/f'panel_{d}.npy', mmap_mode='r')
            np.testing.assert_allclose(predict(experts, t[::13], prior()), base[::13], atol=2e-6, rtol=0)
            p = predict(experts, t, w)
            history.append(moments(experts, y, t[d%5::5]))
            raw = gls.book.rawday(d)
            s, v = gls.book.state(raw)
            sample = np.arange(d%7, len(y)-60, 7)
            for panel_pred, rows in zip((base, p), (base_rows, calibrated_rows)):
                rows.append(gls.record(s, v, panel_pred, y, sample))
            if not retrofit:
                continue
            ix = t[d%4::4]
            xy = gls.design(s, v, p, ix)
            data = [(base[ix], p[ix])] + [(p[ix], p[ix]+xy@coef) for coef in retrofit]
            scores = []
            for target, (original, updated) in zip(block, data):
                row = tuple(a.reshape(-1, 3) for a in (original, updated, y[ix]))
                target.append(row)
                scores.append(measure([row]))
            daily.append(dict(day=d+1, scores=scores))
        if retrofit:
            folds.append([measure(rs) for rs in block])
            print(end+1, [(m['baseline_score'], m['score']) for m in folds[-1]], flush=True)
    gains = np.array([[np.subtract(s['candidate'], s['base']) for s in fold] for fold in folds])
    wins = np.array([[np.subtract(s['candidate'], s['base']) for s in d['scores']] for d in daily])
    # Deploy only corrections that improve the stronger reference on every two-day fold
    # and outperform training the same correction on the old reference in aggregate.
    gate = ((gains[:, 0].min(0)>0) & (gains[:, 2].min(0)>0)
            & ((gains[:, 2]-gains[:, 1]).mean(0)>0) & ((wins[:, 2]>0).sum(0)>=4))
    np.save(OUT/'reference_weights.npy', fit(history))
    report = dict(comparisons=['fixed vs adaptive reference', 'old residual head on adaptive reference',
                               'retrained residual head on adaptive reference'],
                  fold_start_days=[11, 13, 15], folds=folds, daily=daily, retained=gate.tolist())
    (OUT/'results.json').write_text(json.dumps(report, indent=2))
    print('retained', gate, flush=True)
    if not gate.any():
        return
    np.save(OUT/'residual_weights.npy', gls.fit(calibrated_rows, True)*gate)
    res = gls.best.walk.replay(load())
    (OUT/'raw_replay.json').write_text(json.dumps(res, indent=2))
    old = json.loads((gls.incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
    print('official', res['pooled']['score'], res['pooled']['score']-old, flush=True)
    if res['pooled']['score'] > old:
        check = gls.best.walk.replay(load())
        np.testing.assert_allclose(res['pooled']['candidate'], check['pooled']['candidate'], atol=1e-10, rtol=0)
        (OUT/'reload_verification.json').write_text(json.dumps(check, indent=2))


def load():
    parent = gls.incumbent.load()
    weights = np.load(OUT/'residual_weights.npy')
    def apply(raw, p):
        base = parent(raw, p)
        s, v = gls.book.state(raw)
        q = base.copy()
        for i in range(min(4, len(q))):
            t = np.arange(i, len(q), 4)
            q[t] += gls.design(s, v, base, t) @ weights
        return q
    return apply


if __name__ == '__main__':
    main()
