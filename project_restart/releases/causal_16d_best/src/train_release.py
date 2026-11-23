"""Train the BookNet backbone or the 20-step correction on the sixteen public training sessions."""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['backbone', 'clock'])
    args = parser.parse_args()
    from experiment_causal_cross_horizon_ridge_v1 import CACHE
    for day in range(16):
        for kind in ('x','y'):
            if not (CACHE / f'day_{day:02d}_{kind}.npy').is_file():
                parser.error('Prepare the daily feature/label cache first; see README.md.')
    if args.stage == 'backbone':
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
