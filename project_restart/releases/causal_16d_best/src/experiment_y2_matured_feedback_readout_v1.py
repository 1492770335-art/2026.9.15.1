"""Fit a separate Y2 feedback coefficient downstream of the frozen Y3 model."""

import argparse
import json
from pathlib import Path
import numpy as np
import experiment_y3_pooled_revision_feedback_v1 as parent

price = parent.price
OUT = parent.OUT.parent / "y2_matured_feedback_readout_v1"
NAMES = (
    "priced revision",
    "priced revision less forecast",
    "permuted revision",
    "permuted residual",
    "fixed revision",
    "fixed residual",
    "price",
    "price less forecast",
    "forecast",
    "forecast control",
)


def train():
    geom = parent.geometry()
    cal, days = [], []
    columns = [0, 3, 4, 7, 8, 11, 12, 15, 16, 19]
    for d, raw, t, q, before, y, ok, geo, *extra in parent.training_panels():
        z = parent.design(raw, q, t, geom)[:, :, columns]
        # Existing Pearson utilities score the last column. Keep Y2 unchanged in the incumbent input.
        p = before[:, :, [0, 2, 1]]
        obs = y[t][:, :, [0, 2, 1]]
        if d >= 8:
            if d % 2 == 0:
                size = price.sizing(cal)
            days.append(price.rscore(p, obs, z * size))
        cal.append(price.rscore(p, obs, z))
    halves = np.stack([price.diff(sum(days[k : k + 4]))[1:] for k in (0, 4)])
    daily = np.stack([price.diff(g)[1:] for g in days])
    eligible = [
        j
        for j in (0, 1)
        if np.all(halves[:, j] > np.maximum.reduce([np.zeros(2), halves[:, j + 2], halves[:, j + 4]]))
        and min(halves[:, j]) > max(min(halves[:, j + 6]), min(halves[:, j + 8]))
        and (daily[:, j] > 0).sum() >= 5
    ]
    pick = max(eligible, key=lambda j: min(halves[:, j])) if eligible else None
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(dict(arms=NAMES, halves=halves.tolist(), daily=daily.tolist(), pick=pick), indent=2)
    )
    print("Y2 halves", halves, "pick", pick, flush=True)
    if pick is not None:
        np.savez(OUT / "model.npz", col=columns[pick], size=price.sizing(cal)[pick])
    return pick


def load():
    geometry = parent.geometry()
    with np.load(OUT / "model.npz") as f:
        col, size = int(f["col"]), float(f["size"])

    def apply(raw, q):
        return size * parent.design(raw, q, np.arange(len(q)), geometry, selected=col)

    return apply


if __name__ == "__main__":
    import torch
    import y3_portfolio_walkforward as replay

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train()
    else:
        replay.replay(OUT, parent.correction, load_final_y2=load)
