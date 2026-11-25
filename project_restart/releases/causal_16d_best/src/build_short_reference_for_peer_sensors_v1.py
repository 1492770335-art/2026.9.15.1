"""Rebuild the two previously selected short readouts chronologically at the issued training ticks."""

import json
import numpy as np
import experiment_short_medium_variance_ratio_v1 as mixed

OUT = mixed.OUT.parent / "short_peer_sensor_reference_v1"


def main():
    OUT.mkdir(exist_ok=True)
    geom = mixed.feedback.geometry()
    warm, cal, hist = [], [], []
    stats = [[], []]
    dates = []
    for d, raw, t, q, y, ok, fg in mixed.ratio.prior.training_panels():
        if d >= 6:
            break
        r, st = mixed.ratio.risk_states(raw, t, geom)
        hist.append(tuple(a[ok] for a in (r, st, mixed.centered(q[t]), mixed.centered(y[t]))))
        warm.append(mixed.source_stats(q[t], y[t], mixed.source(raw, q, geom)[t]))
    for d, raw, t, q, before, y, ok, fg in mixed.ratio.training_panels():
        if d % 2 == 0:
            w = mixed.weights(cal or warm)
            model = mixed.ratio.fit(hist)
        src = mixed.source(raw, q, geom)[t]
        cal.append(mixed.source_stats(before, y[t], src))
        p = before.copy()
        p[:, :, :2] += src * w
        r, st = mixed.ratio.risk_states(raw, t, geom)
        z = mixed.design(r, st, mixed.centered(p), model)[0]
        hist.append(tuple(a[ok] for a in (r, st, mixed.centered(p), mixed.centered(y[t]))))
        if d >= 8 and d % 2 == 0:
            sizes = [mixed.price.sizing(hist) for hist in stats]
        out = p.copy()
        for j, col in enumerate((3, 1)):
            stats[j].append(mixed.stat(p, y[t], z[:, :, :, j], j))
            if d >= 8:
                out[:, :, j] += sizes[j][col] * z[:, :, col, j]
        np.savez(OUT / f"day_{d:02d}.npz", t=t, baseline=out[:, :, :2])
        dates.append(d)
    assert dates == list(range(6, 16))
    (OUT / "days.json").write_text(json.dumps([d + 1 for d in dates]))


if __name__ == "__main__":
    main()
