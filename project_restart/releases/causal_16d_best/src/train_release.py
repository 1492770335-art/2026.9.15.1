"""Train a forecasting component on the first sixteen public trading sessions."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['backbone', 'clock', 'path-pretrain', 'path-scale', 'price-teacher'])
    parser.add_argument('--output', type=Path, help='Destination .npz for a trained path encoder or one-step price model')
    args = parser.parse_args()
    if args.output and (args.stage not in ('path-pretrain', 'path-scale', 'price-teacher') or args.output.suffix != '.npz'):
        parser.error('--output requires a path encoder or price-teacher stage and a .npz suffix')
    from experiment_causal_cross_horizon_ridge_v1 import CACHE
    train_days = 4 if args.stage == 'price-teacher' else 16
    for day in range(train_days):
        for kind in (('x',) if args.stage == 'price-teacher' else ('x', 'y')):
            if not (CACHE / f'day_{day:02d}_{kind}.npy').is_file():
                parser.error('Prepare the daily feature/label cache first; see README.md.')
    if args.stage == 'price-teacher':
        import experiment_y3_expected_price_paths_v1 as path
        output = args.output or (path.OUT / 'price_teacher.npz')
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, teacher=path.teacher())
        print(output)
    elif args.stage == 'path-scale':
        import torch
        import experiment_y3_scale_conditioned_path_v1 as path
        torch.set_num_threads(2)
        output = args.output or (path.OUT / 'scale16_pretrain.npz')
        output.parent.mkdir(parents=True, exist_ok=True)
        print('Fine-tuning direction and stock-specific movement magnitude on days 1–16', flush=True)
        np.savez(output, **path.train(16))
        print(output)
    elif args.stage == 'path-pretrain':
        import torch
        import experiment_y3_dense_orientation_curve_v1 as path
        torch.set_num_threads(2)
        output = args.output or (path.OUT / 'dense16_pretrain.npz')
        output.parent.mkdir(parents=True, exist_ok=True)
        print('Training the 9-window path encoder on days 1–16', flush=True)
        np.savez(output, **path.train(16, ('dense',)))
        print(output)
    elif args.stage == 'backbone':
        import experiment_phase_future_book_v1 as phase
        phase.OUT.mkdir(parents=True, exist_ok=True)
        phase.train(16)
        print(phase.OUT / 'net_16.pth')
    else:
        import experiment_periodic_book_response_v1 as clock
        shift, _ = clock.clock_geometry()
        history = []
        for day in range(16):
            history.append(clock.statistics(day, shift))
            print('fitted moments for date', day, flush=True)
        market, relative = clock.weights(history)
        # Published architecture chosen on the inner rolling splits. Coefficients are refitted here.
        mask = np.array([[True, True, False], [True, True, False]])
        clock.OUT.mkdir(parents=True, exist_ok=True)
        output = clock.OUT / 'model.npz'
        np.savez(output, m=market, r=relative, shift=shift, keep=mask)
        print(output)


if __name__ == '__main__':
    main()
