"""Separate a forecast update's covariance with future returns from its effect on prediction scale."""
import json

import numpy as np

import y3_quote_token_references as history

trial = history.trial
WINDOWS = ('Y3', '1–5', '1–20', '21–40', '41–60')


def stats(old, new, target):
    # Cross-section centered arrays have zero pooled mean, so second moments are covariances.
    return np.array([old.size, np.sum(old*old), np.sum(new*new), np.sum(target*target),
                     np.sum(old*target), np.sum((new-old)*target)])


def decompose(m):
    n, a, b, c, ay, dy = m
    initial = ay/np.sqrt(a*c)
    signal = dy/np.sqrt(b*c)
    scale = initial*(np.sqrt(a/b)-1)
    return dict(rows=int(n), baseline=float(initial), covariance_effect=float(signal),
                scale_effect=float(scale), correlation_gain=float(signal+scale))


def run():
    assembled = []
    for day in range(8, 16):
        raw = trial.clock.book.rawday(day)
        t = np.arange(day%4, len(raw)-60, 4)
        y = trial.clock.labels(day).reshape(-1, 50, 3)
        targets = [trial.clock.center(v.astype(float)) for v in
                   (y[t, :, 2], y[t, :, 0], y[t, :, 1], y[t+20, :, 1], y[t+40, :, 1])]
        first = history.previous.references(day, raw)
        second = history.references(day, raw)
        rows = []
        for a, b in zip(first, second):
            old, new = [trial.clock.center(p[t, :, 2].astype(float)) for p in (a, b)]
            rows.append([stats(old, new, target) for target in targets])
        assembled.append(rows)
    records = np.array(assembled)
    report = dict(days=list(range(9, 17)), reference_streams=2, windows=WINDOWS,
                  pooled=[[decompose(m) for m in ref] for ref in records.sum(0)],
                  folds=[[[decompose(m) for m in ref] for ref in fold.sum(0)]
                         for fold in np.split(records, 4)])
    (history.incumbent.OUT/'chronological_horizon_attribution.json').write_text(json.dumps(report, indent=2))
    for row in report['pooled']:
        print([(w, round(r['covariance_effect'], 6), round(r['scale_effect'], 6)) for w, r in zip(WINDOWS, row)])


if __name__ == '__main__':
    run()
