"""Materialize verified upstream predictions using the previous version's fitted stages."""
import hashlib
import json
import shutil
import sys
import numpy as np

from runtime import PREVIOUS, parser


def prepare(args):
    # Import the preceding version's loader in this separate process.
    sys.path.insert(0, str(PREVIOUS))
    from replay import configure, load_models
    configure(args.run_dir, args.prior_src, args.rolling)
    import torch
    import version_paths as paths
    from prepare_dense_settled_reference_v1 import update
    torch.set_num_threads(2)
    fits = load_models()
    art = paths.ROOT / 'artifacts'
    source = art / 'dense_five_tick_verified_reference_v1'
    upstream = json.loads((source / 'verified.json').read_text())
    directory = art / 'dense_basket_release_verified_reference_v1'
    directory.mkdir(exist_ok=True)
    hashes = {}
    for day in range(6, 16):
        name = f'rolling_{day:02d}.npy'
        shutil.copyfile(art / 'dense_basket_feedback_verified_reference_v1' / name, directory / name)
    for day in range(16, 20):
        x, grid = [np.load(paths.CACHE / f'day_{day:02d}_{key}.npy', mmap_mode='r')
                   for key in ('x', 'time')]
        for kind in ('full', 'prefix'):
            name = f'{kind}_{day}.npy'
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != upstream['hashes'][name]:
                raise ValueError(f'Invalid upstream forecast {name}')
            p = np.load(source / name)
            for model, fit in fits:
                inputs = x[:len(p)], grid[:len(p)].astype(int), p
                if hasattr(model, 'predict'):
                    p = model.predict(*inputs, fit)
                else:
                    p = update(model, *inputs, fit['code'], fit['bank'], fit['w'])
            if kind == 'full':
                full = p
            elif not np.max(np.abs(p - full[:len(p)])) < 1e-4:
                raise ValueError(f'Upstream prefix mismatch on day {day+1}')
            np.save(directory / name, p)
            hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        print(f'Generated upstream predictions for day {day+1}', flush=True)
    (directory / 'verified.json').write_text(json.dumps(dict(
        commit='5006dc8ee454ed84304a92c034a51c5575488fd4', hashes=hashes), indent=2))


if __name__ == '__main__':
    prepare(parser(__doc__).parse_args())
