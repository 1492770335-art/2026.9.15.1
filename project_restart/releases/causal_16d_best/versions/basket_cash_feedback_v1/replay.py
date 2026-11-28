"""Reload every cumulative correction and evaluate the full and truncated four-day predictions."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from stages import STAGES


def configure(run, prior, rolling=None):
    os.environ['TULIN_RUN_DIR'] = str(run.resolve())
    os.environ['TULIN_PRIOR_SRC'] = str(prior.resolve())
    if rolling is not None:
        os.environ['TULIN_ROLLING'] = str(rolling.resolve())
    sys.path[:0] = [str(Path(__file__).resolve().parent/'src'), str(prior.resolve())]


def predict_day(day, models, book, hashes):
    import version_paths as paths
    from prepare_dense_settled_reference_v1 import update
    from experiment_causal_cross_horizon_ridge_v1 import measure
    grid, raw = [np.load(paths.CACHE/f'day_{day:02d}_{k}.npy',mmap_mode='r') for k in ('time','x')]
    for kind in ('full','prefix'):
        name = f'{kind}_{day}.npy'
        path = book/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'Base release mismatch: {name}')
        p = np.load(path)
        original = p.copy()
        for module, fit in models:
            inputs = raw[:len(p)],grid[:len(p)].astype(int),p
            if hasattr(module,'predict'):
                p = module.predict(*inputs,fit)
            else:
                p = update(module,*inputs,fit['code'],fit['bank'],fit['w'])
        if kind=='full':
            old, full = original, p
        else:
            error = float(np.max(np.abs(full[:len(p)]-p)))
            if error>=1e-4:
                raise ValueError(f'Day {day+1} prefix error {error}')
    y = np.load(paths.CACHE/f'day_{day:02d}_y.npy')
    row = tuple(a.reshape(-1,3) for a in (old,full,y))
    return row, dict(day=day+1,prefix_error=error,**measure([row]))


def load_models():
    models = []
    for name, _, _ in STAGES:
        module = importlib.import_module(name)
        with np.load(module.OUT/'weights.npz') as data:
            models.append((module,dict(data)))
    return models


def main(args):
    configure(args.run_dir,args.prior_src,args.rolling)
    import torch
    from experiment_causal_cross_horizon_ridge_v1 import measure
    import version_paths as paths

    torch.set_num_threads(2)
    book = args.base_reference or paths.ROOT/'artifacts/dense_five_tick_verified_reference_v1'
    info = json.loads((book/'verified.json').read_text())
    models = load_models()
    rows, days = [], []
    for day in range(16,20):
        row, summary = predict_day(day,models,book,info['hashes'])
        rows.append(row)
        days.append(summary)
        print(summary,flush=True)
    result = dict(pooled=measure(rows),daily=days)
    assert result['pooled']['rows'] == 936250
    args.report.write_text(json.dumps(result,indent=2))
    print(result['pooled'],flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-src',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--rolling',type=Path)
    p.add_argument('--base-reference',type=Path,help='Optional separate directory of verified five-tick release predictions')
    p.add_argument('--report',type=Path,default=Path('basket_replay.json'))
    main(p.parse_args())
