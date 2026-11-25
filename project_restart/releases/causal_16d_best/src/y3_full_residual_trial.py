"""Walk forward over training dates to screen corrections to the full Y3 forecast."""
import json

import numpy as np

import y3_event_response_references as parent

screen, trial = parent.screen, parent.trial


def error(p, y, t):
    q = p[t, :, 2].astype(float)
    z = y[t, :, 2].astype(float)
    beta = np.sum((q-q.mean())*(z-z.mean())) / max(1e-8, np.sum((q-q.mean())**2))
    return trial.clock.center(z/beta.clip(.4, 2.)-q).astype('float32')


def score_run(module, reference=parent):
    module.OUT.mkdir(exist_ok=True)
    train, days, folds = [], [], []
    target_error = getattr(module, 'training_error', error)
    window_component = getattr(module, 'window_component', trial.clock.center)
    for d in range(4, 16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1, 50, 3)
        refs = (reference.references(d, raw) if d >= 8 else
                [screen.history.market.prior.path.early.load(d, raw)])
        if d >= 8 and d % 2 == 0:
            state = module.fit(train)
            groups = [[], []]
        selected = np.arange(90+d%21, len(raw)-60, 21)
        evaluation = np.arange(d%4, len(raw)-60, 4)
        ticks = np.unique(np.r_[selected, evaluation]) if d >= 8 else selected
        design = module.historical_design(d) if hasattr(module, "historical_design") else module.design
        records = []
        for j, p in enumerate(refs):
            x = design(raw, p, ticks)
            records.append((x[np.searchsorted(ticks, selected)], target_error(p, y, selected)))
            if d < 8:
                continue
            delta = module.changes(x[np.searchsorted(ticks, evaluation)], state)
            if d == 8 and j == 0:
                t = evaluation[evaluation < 311]
                for a, b in zip(delta, module.changes(design(raw[:311], p[:311], t), state)):
                    np.testing.assert_allclose(a[:len(t)], b, atol=4e-6, rtol=0)
            stat = trial.scoring.audit.blockstats
            t = evaluation
            stats = [stat(p[t], y[t], delta).sum(0)]
            base = np.zeros_like(p[t])
            base[:, :, 2] = window_component(p[t, :, 2])
            for offset in (0, 20, 40):
                target = np.zeros_like(base)
                target[:, :, 2] = window_component(y[t+offset, :, 1])
                stats.append(stat(base, target, delta).sum(0))
            groups[j].append(stats)
            if j == 0:
                days.append(trial.scoring.audit.pooled(stats[0]))
        train.append(tuple(np.concatenate(a) for a in zip(*records)))
        if d >= 8 and d % 2:
            folds.append([[trial.scoring.audit.pooled(g) for g in np.sum(group, axis=0)] for group in groups])
    scores, daily = np.asarray(folds), np.asarray(days)
    eligible = []
    for j in module.CANDIDATES:
        gain = scores[:, :, 0, j]
        if ((gain > 0).sum(0).min() >= 3 and gain.mean(0).min() > 0 and gain[-1].min() > 0
                and (daily[:, j] > 0).sum() >= 5
                and all((gain-scores[:, :, 0, k]).mean(0).min() > 0 for k in module.CONTROLS)):
            eligible.append(j)
    pick = max(eligible, key=lambda j: scores[:, :, 0, j].mean()) if eligible else None
    report = dict(selected=pick, arms=module.ARMS, folds=scores.tolist(), daily=daily.tolist(),
                  windows=['Y3', '1–20', '21–40', '41–60'])
    (module.OUT / 'results.json').write_text(json.dumps(report, indent=2))
    print(module.OUT.name, 'selected', pick, 'mean Y3 gains', np.round(scores[:, :, 0].mean(0), 6), flush=True)
    if pick is not None:
        np.savez(module.OUT / 'model.npz', pick=pick, **module.fit(train))
        reference.outer(module)


def frozen(module):
    with np.load(module.OUT / 'model.npz') as archive:
        state = dict(archive)
    def update(raw, p):
        x = module.design(raw, p, np.arange(len(raw)))
        return module.changes(x, state)[int(state['pick'])]
    return update


def layer(module, previous):
    predict = previous.layer()
    update = module.correction()
    def apply(raw, initial):
        forecast = predict(raw, initial)
        forecast[:, :, 2] += update(raw, forecast)
        return forecast
    return apply
