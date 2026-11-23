"""Read Parquet using PyArrow and cache complete stock grids, one file per day.

May run in a separate Python environment from training; output is portable NumPy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def prepare(source, output, dates):
    dataset = ds.dataset(source, format='parquet')
    features = sorted((n for n in dataset.schema.names if n.startswith('f')), key=lambda n: int(n[1:]))
    if [int(n[1:]) for n in features] != list(range(100)):
        raise ValueError('expected exactly features 0..99')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    source_info = {'path': str(source.resolve()), 'bytes': source.stat().st_size,
                   'mtime_ns': source.stat().st_mtime_ns}
    manifest = {'source': source_info, 'features': features, 'days': {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source'] != source_info or manifest['features'] != features:
            raise ValueError('cache belongs to a different source; choose a new cache directory')
    for date in dates:
        if str(date) in manifest['days']:
            entry = manifest['days'][str(date)]
            if all((output / entry[key]).exists() for key in ('x', 'y', 'time')):
                continue
        table = dataset.to_table(filter=ds.field('date') == date)
        if table.num_rows == 0:
            raise ValueError(f'date {date} not present')
        columns = {name: table[name].to_numpy() for name in table.column_names}
        order = np.lexsort((columns['code'], columns['time']))
        code, time = columns['code'][order], columns['time'][order]
        if len(code) % 50:
            raise ValueError('incomplete stock grid')
        ticks = len(code) // 50
        if not np.all(code.reshape(-1, 50) == np.arange(50)):
            raise ValueError('missing, duplicated or invalid stock codes')
        time_grid = time.reshape(-1, 50)
        if not np.all(time_grid == time_grid[:, :1]) or not np.all(np.diff(time_grid[:, 0]) > 0):
            raise ValueError('invalid time ordering/grid')
        x = np.column_stack([columns[name][order] for name in features]).astype(np.float16)
        y = np.column_stack([columns[name][order] for name in ('y0', 'y1', 'y2')]).astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('non-finite features/labels')
        if not np.isin(x[:, 83:], [0, 1, 2]).all():
            raise ValueError('invalid categorical features')
        entry = {'ticks': ticks, 'x': f'day_{date:02d}_x.npy',
                 'y': f'day_{date:02d}_y.npy', 'time': f'day_{date:02d}_time.npy'}
        for name, value in [('x', x.reshape(ticks, 50, 100)),
                            ('y', y.reshape(ticks, 50, 3)), ('time', time_grid[:, 0])]:
            target = output / entry[name]
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.save(f, value)
            temporary.replace(target)
        manifest['days'][str(date)] = entry
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
        print(f'date={date} ticks={ticks} rows={ticks * 50}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('project/data/public.parquet'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/joint_transformer/cache'))
    parser.add_argument('--dates', type=int, nargs='+', default=list(range(20)))
    args = parser.parse_args()
    prepare(args.data, args.cache, args.dates)
