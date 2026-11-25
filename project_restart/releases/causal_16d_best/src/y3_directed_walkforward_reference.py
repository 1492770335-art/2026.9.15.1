"""Chronological references including the paired-book network, early slopes and explained-price layer."""
import json

import numpy as np
import torch

import experiment_y3_directed_unexpected_prices_v1 as price

slopes = price.previous_stage
pairs = slopes.pairs
OUT = price.OUT.parent/'y3_directed_walkforward_reference'
CHOSEN_PAIR = 3


def fit():
    geometry = slopes.early_slopes.geometry()
    k = geometry[0]
    grams = [price.observe(price.tape(
        np.load(price.CACHE/f'day_{d:02d}_x.npy', mmap_mode='r'), k)) for d in range(4)]
    pair_hist, early_hist, price_hist, price_scores = [], [], [], []
    metric = {}
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(exist_ok=True)
    for d, raw, panels, y, _ in pairs.panels():
        _, old, _ = price.read_day(d)
        p = panels[-1] if d >= 8 else np.mean(panels[:2], 0)
        if d % 2 == 0:
            innov = price.innovation_model(grams)
        x, st, ch = price.designs(raw, old, k, innov)
        x[:, :, 0] = st[2]
        if d >= 6:
            if d % 2 == 0:
                direction = price.directed_response(price_hist)[None]
            z = price.signals(x, direction)
        feats, blocks = slopes.early_slopes.collect(raw, p, geometry)
        pref = pairs.prefix(raw)
        q = p.copy()
        if d >= 8:
            if d % 2 == 0:
                net = pairs.fit(pair_hist, dev)
            early, updates = slopes.changes(p, feats, blocks, slopes.fit(early_hist))
            alignment = pairs.shifts(pref, p, net)[0][:, :, CHOSEN_PAIR-1]
            q[:, :, 2] += alignment+early+updates[:, :, 2]
            # The released directed layer was fitted on the older references, then applied
            # to all earlier corrections at inference. Reproduce that same ordering here.
            size = price.sizing(price_scores)[0]
            direction_final = direction[0]*size
            q[:, :, 2] += price.adjust(raw, q, innov, k, 4, direction_final)
            quick = old.copy()
            quick[:, :, 2] += size*z[:, :, 0]
            delta = q[:, :, 2]-quick[:, :, 2]
            metric[d+1] = dict(rms=float(np.sqrt(np.mean(delta**2))),
                               old_corr=float(np.corrcoef(quick[:, :, 2].ravel(), y[:, :, 2].ravel())[0, 1]),
                               updated_corr=float(np.corrcoef(q[:, :, 2].ravel(), y[:, :, 2].ravel())[0, 1]))
        if d >= 6:
            price_scores.append(price.rscore(old, y, z))
        np.testing.assert_array_equal(q[:, :, :2], old[:, :, :2])
        assert np.isfinite(q).all()
        tmp = OUT/f'day_{d}.tmp.npy'
        np.save(tmp, q)
        tmp.replace(OUT/f'day_{d}.npy')
        grams.append(price.observe(ch))
        price_hist.append(price.moments(x, old, y, d))
        early_hist.append(np.stack([b.sum(0) for b in blocks]))
        pair_hist.append(pairs.sample(pref, p, y, d))
        print('day', d+1, metric.get(d+1, ''), flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(days=metric,
                                                    folds=[[9, 12], [13, 16]], pair_choice=CHOSEN_PAIR), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    fit()
