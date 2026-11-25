"""Re-issue Y3 at all training ticks using only parameters fitted on preceding days."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    src = args.release_src.resolve()
    os.chdir(src)
    sys.path.insert(0, str(src))
    import torch
    import experiment_y3_prequential_disagreement_gate_v1 as gate

    torch.set_num_threads(1)
    risk, price, feedback = gate.source, gate.price, gate.rev
    mag, occ = risk.prior, feedback.parent
    traj = mag.trajectory
    cohort = traj.cohort
    motion = cohort.motion
    geom = feedback.geometry()
    k, co, _ = geom
    rho = co[2, 1] / co[1, 1]
    axis = motion.parent.direction(k)
    warm = [
        price.observe(price.tape(np.load(price.CACHE / f"day_{i:02d}_x.npy", mmap_mode="r"), k))
        for i in range(4)
    ]
    sd = price.innovation_model(warm)[2]
    geo = (k, sd, axis, rho)

    def features(stage, raw, q, t):
        if stage == 0:
            return motion.design(raw, q, t, *geo[:3])
        if stage == 1:
            return cohort.design(raw, q, t, *geo)
        return traj.design(raw, q, t, geo)[0]

    specs = [(motion.weights, 0), (cohort.fit, 1), (traj.fit, 6)]
    history = [[] for _ in specs]
    scales = [[] for _ in range(8)]
    weights = [None] * 3
    sizes = np.zeros(8)
    mag_history, regimes, risk_history, axes_history = [], [], [], []
    diagnostics = []

    def final_layers(raw, q, t, vol, state, rates, risk_model, axes):
        z = mag.signals(vol[t], q[t], rho, state)[0][:, :, 2]
        o = occ.signals(vol, q, t, rho, state, rates)[:, :, 3]
        f = feedback.design(raw, q, t, geom, selected=3)
        after = q[t].copy()
        after[:, :, 2] += sizes[3] * z
        after[:, :, 2] += sizes[4] * o
        after[:, :, 2] += sizes[5] * f
        r, st = risk.risk_states(raw, t, geom)
        rv = risk.selected_design(r, st, mag.bands(after, rho), risk_model, 5)
        after[:, :, 2] += sizes[6] * rv
        market = gate.design(raw, after, t, axes, geom, rho)[:, 1]
        after[:, :, 2] += sizes[7] * market[:, None]
        return after

    def dense_panel(raw, upstream):
        q = upstream.copy()
        t = np.arange(len(q))
        for j in range(3):
            q[:, :, 2] += sizes[j] * (features(j, raw, q, t) @ weights[j])
        return final_layers(raw, q, t, mag.risk(raw, t), state, rates, risk_model, axes)

    for d, raw, upstream, y in motion.records():
        assert 4 <= d < 16
        t = np.arange(d % 5, len(raw), 5)
        valid = ((t < 2401 - 60) | (t >= 2401)) & (t < len(raw) - 60)
        if d >= 8 and d % 2 == 0:
            sizes = np.array([price.sizing(hist)[0] for hist in scales])
        sparse = upstream.copy()
        for j, (fit, col) in enumerate(specs):
            x = features(j, raw, sparse, t)
            prev = sparse[t]
            if d >= 6:
                if d % 2 == 0:
                    weights[j] = fit(history[j])[:, col]
                z = x @ weights[j]
                scales[j].append(price.rscore(prev, y[t], z[:, :, None]))
                sparse[t, :, 2] += sizes[j] * z
            history[j].append(motion.parent.stat(x[valid], prev[valid], y[t[valid]], rho))
        all_t = np.arange(len(raw))
        vol = mag.risk(raw, all_t)
        p = mag.bands(sparse[t], rho)
        truth = mag.bands(y[t], rho)
        r, st = risk.risk_states(raw, t, geom)
        before = sparse[t].copy()
        if d >= 6:
            if d % 2 == 0:
                state = mag.fit(mag_history)
                rates = occ.fit_teacher(regimes, state[0])
                risk_model = risk.fit(risk_history)
                axes = gate.factors(axes_history)
            signals = [
                mag.signals(vol[t], sparse[t], rho, state)[0][:, :, 2],
                occ.signals(vol, sparse, t, rho, state, rates)[:, :, 3],
                feedback.design(raw, sparse, t, geom, selected=3),
            ]
            for j, signal in enumerate(signals, 3):
                scales[j].append(price.rscore(before, y[t], signal[:, :, None]))
                before[:, :, 2] += sizes[j] * signal
            z = risk.selected_design(r, st, mag.bands(before, rho), risk_model, 5)
            market = gate.design(raw, sparse, t, axes, geom, rho)[:, 1:2]
            scales[6].append(price.rscore(before, y[t], z[:, :, None]))
            scales[7].append(gate.gram(before, y[t], market))
        risk_history.append(tuple(v[valid] for v in (r, st, mag.bands(before, rho), truth)))
        mag_history.append((vol[t[valid]], p[valid], truth[valid]))
        regimes.append((vol, t[valid]))
        axes_history.append(gate.update_geometry(sparse))

        legacy = sparse.copy()
        if d >= 8:
            legacy[t, :, 2] = before[:, :, 2] + sizes[6] * z + sizes[7] * market
            dense = dense_panel(raw, upstream)
            if d == 8:
                cut = 2517
                prefix = dense_panel(raw[:cut], upstream[:cut])
                np.testing.assert_allclose(prefix, dense[:cut], atol=3e-6, rtol=0)
        else:
            dense = upstream
        assert np.isfinite(dense).all()
        unobserved = np.ones(len(raw), bool)
        unobserved[t] = False
        delta = dense[:, :, 2] - legacy[:, :, 2]

        def rms(a):
            return float(np.sqrt(np.mean(a.astype(float) ** 2)))

        row = dict(
            day=d + 1,
            scheduled_rms=rms(delta[t]),
            other_rms=rms(delta[unobserved]),
            dense_tick_change_rms=rms(np.diff(dense[:, :, 2], axis=0)),
            sparse_tick_change_rms=rms(np.diff(legacy[:, :, 2], axis=0)),
        )
        print(row, flush=True)
        diagnostics.append(row)
        np.savez(
            output / f"day_{d:02d}.npz",
            sparse=legacy[:, :, 2],
            dense=dense[:, :, 2],
            short=dense[:, :, :2],
            t=t,
        )
    assert len(diagnostics) == 12
    errors = {}
    for j, module in enumerate((motion, cohort, traj)):
        fit, col = specs[j]
        actual = fit(history[j])[:, col] * price.sizing(scales[j])[0]
        with np.load(module.OUT / "model.npz") as saved:
            errors[module.__name__] = float(np.max(abs(saved["w"] - actual)))
            np.testing.assert_allclose(actual, saved["w"], atol=1e-6, rtol=1e-6)
    for j, module in enumerate((mag, occ, feedback, risk, gate), 3):
        with np.load(module.OUT / "model.npz") as saved:
            error = float(abs(price.sizing(scales[j])[0] - saved["size"]))
            errors[module.__name__] = error
            np.testing.assert_allclose(price.sizing(scales[j])[0], saved["size"], atol=2e-6, rtol=1e-6)
    (output / "parameter_parity.json").write_text(json.dumps(errors, indent=2))
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
