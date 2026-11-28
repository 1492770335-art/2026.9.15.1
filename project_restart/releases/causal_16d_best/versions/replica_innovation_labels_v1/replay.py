"""Replay all corrections from the previous release on full days and independent prefixes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from runtime import configure, modules, parser


def replay(args):
    configure(args)
    import version_paths as paths
    from experiment_causal_cross_horizon_ridge_v1 import measure
    book = paths.ROOT / 'artifacts/dense_basket_release_verified_reference_v1'
    metadata = json.loads((book / 'verified.json').read_text())
    if metadata['commit'] != '5006dc8ee454ed84304a92c034a51c5575488fd4':
        raise ValueError('Unexpected upstream release')
    fits = []
    for model in modules():
        with np.load(model.OUT / 'weights.npz') as f:
            fits.append((model, dict(f)))
    rows, daily = [], []
    for day in range(16, 20):
        x, grid = [np.load(paths.CACHE / f'day_{day:02d}_{key}.npy', mmap_mode='r')
                   for key in ('x', 'time')]
        for kind in ('full', 'prefix'):
            name = f'{kind}_{day}.npy'
            path = book / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['hashes'][name]:
                raise ValueError(f'Unexpected base prediction in {name}')
            p = np.load(path)
            if kind == 'full':
                base = p.copy()
            for model, fit in fits:
                p = model.predict(x[:len(p)], grid[:len(p)].astype(int), p, fit)
            if kind == 'full':
                full = p
            else:
                error = float(np.max(np.abs(p - full[:len(p)])))
                if not error < 1e-4:
                    raise ValueError(f'Prefix mismatch on day {day + 1}: {error}')
        y = np.load(paths.CACHE / f'day_{day:02d}_y.npy')
        rows.append(tuple(a.reshape(-1, 3) for a in (base, full, y)))
        daily.append(dict(day=day+1, prefix_error=error, **measure(rows[-1:])))
        print(daily[-1], flush=True)
    result = dict(pooled=measure(rows), daily=daily)
    if result['pooled']['rows'] != 936250:
        raise ValueError('Incomplete four-day replay')
    args.report.write_text(json.dumps(result, indent=2))
    print(result['pooled'], flush=True)


if __name__ == '__main__':
    cli = parser(__doc__)
    cli.add_argument('--report', type=Path, default=Path('replica_replay.json'))
    replay(cli.parse_args())
