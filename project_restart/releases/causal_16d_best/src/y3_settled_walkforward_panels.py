"""Chronological training panels with the peer, settled-gradient and attention layers fitted at each cutoff."""
import numpy as np

import experiment_y3_settled_error_attention_v1 as attention
from y3_portfolio_walkforward import historical_panels, parent

gradient, peers = attention.gradient, attention.peers
CACHE = parent.ART / 'y3_settled_walkforward_panels_v1'


def build():
    histories, donors, profiles = [], [], []
    geometry = gradient.historical_geometry()
    CACHE.mkdir(exist_ok=True)
    for day, raw, refs, labels in historical_panels():
        corrected = []
        if day >= 8:
            if day % 2 == 0:
                response = gradient.fit(histories)
                neighbors = peers.fit(donors, profiles)
            for p in refs:
                q = p.copy()
                q[:, :, 2] += peers.signals(raw, p, neighbors)[:, :, 4]
                corrected.append(q)
        else:
            corrected = refs
        # Gradient fitting in the released training script uses the peer-only panels.
        histories.append(gradient.record(raw, np.mean(corrected[:2], 0), labels, day, geometry))
        if day >= 8:
            for q in corrected:
                shifts = gradient.shifts(raw, q, geometry, response)[:, :, 2]
                weighting = attention.lookup(raw, q, response[1, 0], geometry)[:, :, 4]
                q[:, :, 2] += shifts
                q[:, :, 2] += weighting
        moments = [peers.observed(raw, p, labels, day) for p in refs[:2]]
        donors.append([tuple(np.mean(v, 0) for v in zip(*s)) for s in zip(*moments)])
        profiles.append(peers.fingerprint(raw))
        dst = CACHE / f'day_{day+1}.npy'
        tmp = dst.with_suffix('.tmp.npy')
        np.save(tmp, np.asarray(corrected))
        tmp.replace(dst)
        print('cached', day+1, flush=True)
    (CACHE/'complete').write_text('training dates 5 through 16\n')


def panels(refresh=False):
    if refresh or not (CACHE/'complete').is_file():
        build()
    for day in range(4, 16):
        raw, _, labels = parent.history.training_day(day)
        yield day, raw, np.load(CACHE/f'day_{day+1}.npy', mmap_mode='r'), labels


if __name__ == '__main__':
    build()
