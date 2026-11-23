"""Walk-forward calibration using present and past out-of-fold predictions."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts/project_restart_generalization'
CACHE = ROOT / 'artifacts/joint_transformer/cache'
OUT = ART / 'causal_cross_horizon_ridge_v1'
DUAL = ART / 'peer_dual_temporal'
WIDTHS = (1, 4, 16, 64, 256)
WEIGHTS = np.array([.2, .3, .5])


def labels(day):
    return np.load(CACHE / f'day_{day:02d}_y.npy').reshape(-1, 3)


def predictions(fold, days):
    if fold == 'outer':
        p = np.load(DUAL / 'outer_prediction.npy')
    else:
        def load(directory, prefix):
            return np.load(ART / directory / f'{prefix}_{fold}.npy').reshape(-1)
        first = .5 * (load('normalized_y0', 'raw_prediction') + load('normalized_y0', 'normalized_prediction'))
        second_raw = np.concatenate([np.load(ART / 'y2_oof_corrector' / f'day_{d:02d}_oof.npy')[:, 0] for d in days])
        second = .5 * (second_raw + load('normalized_y1', 'prediction'))
        p = np.column_stack([first, second, load('peer_dual_temporal', 'prediction')])
    start = 0
    for day in days:
        y = labels(day)
        stop = start + len(y)
        yield day, p[start:stop], y
        start = stop
    assert start == len(p)


def features(p):
    grid = p.astype(np.float64).reshape(-1, 50, 3)
    total = np.concatenate([np.zeros_like(grid[:1]), grid.cumsum(0)])
    end = np.arange(1, len(grid) + 1)
    cols = []
    for width in WIDTHS:
        start = np.maximum(0, end - width)
        mean = (total[end] - total[start]) / (end-start)[:, None, None]
        market = mean.mean(1, keepdims=True)
        cols.extend([mean-market, np.broadcast_to(market, mean.shape)])
    return np.concatenate(cols, axis=-1).reshape(len(p), -1)


def fit(moments):
    gram, rhs, rows = map(sum, zip(*moments))
    # Diagonal ridge equals a .05 penalty after scaling each column by train RMS.
    return np.linalg.solve(gram + .05*np.diag(np.maximum(np.diag(gram), rows*1e-8)), rhs)


def corr(p, y):
    p = p.astype(np.float64) - p.mean(0, dtype=np.float64)
    y = y.astype(np.float64) - y.mean(0, dtype=np.float64)
    return (p*y).sum(0) / np.sqrt((p*p).sum(0)*(y*y).sum(0))


def measure(entries):
    old, new, truth = [np.concatenate(items) for items in zip(*entries)]
    a, b = corr(old, truth), corr(new, truth)
    return dict(base=a.tolist(), candidate=b.tolist(), baseline_score=float(a@WEIGHTS),
                score=float(b@WEIGHTS), gain=float((b-a)@WEIGHTS), rows=len(old))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(widths=list(WIDTHS),ridge=.05,shrinkage=.5,
                fit='OOF predictions only, ridge against residuals; reset trailing means each day',
                validation='fit days 8,9 -> 10,11; fit 8..11 -> 12..15 (zero indexed)',
                outer_gate='positive pooled weighted score change on both blocks and >=4/6 positive days; refit 8..15')
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2))
    moments, recent, report = [], [], {'early':{},'daily':{}}
    for fold, days in [('A',range(8,12)), ('B',range(12,16))]:
        for day, p, y in predictions(fold, days):
            x = features(p)
            if day in (10,12):
                coef = fit(moments)
                recent = []
            if day >= 10:
                new = p + .5*x@coef
                recent.append((p, new, y))
                report['daily'][day] = measure(recent[-1:])
                if day in (11,15):
                    report['early'][fold] = measure(recent)
            moments.append((x.T@x, x.T@(y.astype(np.float64)-p), len(x)))
    report['passed'] = all(v['gain']>0 for v in report['early'].values()) and sum(v['gain']>0 for v in report['daily'].values())>=4
    if report['passed']:
        coef = fit(moments)
        np.save(OUT/'coefficients.npy', coef)
        recent = []
        for day,p,y in predictions('outer',range(16,20)):
            recent.append((p, p+.5*features(p)@coef,y))
            report['daily'][day] = measure(recent[-1:])
        report['outer'] = measure(recent)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
