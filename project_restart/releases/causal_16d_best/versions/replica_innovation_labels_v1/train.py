"""Fit the retained stages chronologically and generate rolling inputs for subsequent stages."""
import json
import numpy as np

from runtime import LAYERS, configure, modules, parser


def fit_stage(model, layer):
    from prepare_dense_vintage_crossing_reference_v1 import rebuild_rolling
    _, _, reference_name, expected = layer
    model.strip.evaluation.rolling(model)
    chosen = json.loads((model.OUT / 'rolling.json').read_text())['selected']
    if chosen != expected:
        raise ValueError(f'{model.OUT.name} selected {chosen}, expected {expected}')
    directory = model.OUT.parent / reference_name
    directory.mkdir(exist_ok=True)
    with np.load(model.OUT / 'weights.npz') as weights:
        rebuild_rolling(dict(weights), model, directory)
    # The separate short-market correction is applied by each stage's reader.
    for day in range(6, 16):
        name = f'rolling_{day:02d}.npy'
        prediction = np.load(directory / name)
        prediction[..., 0] = np.load(model.REFERENCE / name, mmap_mode='r')[..., 0]
        np.save(directory / name, prediction)


if __name__ == '__main__':
    cli = parser(__doc__)
    cli.add_argument('--start', type=int, default=1)
    cli.add_argument('--stop', type=int, default=len(LAYERS))
    args = cli.parse_args()
    if not 1 <= args.start <= args.stop <= len(LAYERS):
        cli.error('Expected 1 <= start <= stop <= 19')
    configure(args)
    for i, (model, layer) in enumerate(zip(modules(), LAYERS), 1):
        if args.start <= i <= args.stop:
            print(f'Fitting {i}/{len(LAYERS)} {model.OUT.name}', flush=True)
            fit_stage(model, layer)
