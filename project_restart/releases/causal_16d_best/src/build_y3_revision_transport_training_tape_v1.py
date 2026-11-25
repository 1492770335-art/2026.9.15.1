"""Replay the three Y3 layers added after the dense training tape, fitting each on past dates only."""

import json
import numpy as np
import experiment_y3_matured_revision_factor_transport_v1 as transport

v = transport.v
partial, prior = transport.partial, transport.partial.prior
OUT = transport.OUT.parent / "y3_revision_transport_training_tape_v1"


def main():
    OUT.mkdir(exist_ok=True)
    geom = v.feedback.geometry()
    from experiment_causal_cross_horizon_ridge_v1 import labels

    geometry = []
    history = [[], [], []]
    counts = []
    for d in range(4, 16):
        raw = np.load(v.price.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
        y = labels(d).reshape(-1, 50, 3)
        ticks = np.arange(len(raw))
        with np.load(v.TAPE / f"day_{d:02d}.npz") as f:
            q = np.concatenate((f["short"], f["dense"][:, :, None]), axis=2)
            t = f["t"]
        updates = np.zeros(q.shape[:2] + (3,), dtype=np.float32)
        if d >= 6:
            if d % 2 == 0:
                size = [v.price.sizing(h)[0] if h else 0 for h in history]
                axis = transport.axes(geometry)
            z, p = transport.signals(raw, q, ticks, geom, size[0])
            for j in (0, 1):
                hist_p = q if j == 0 else p
                history[j].append(v.price.rscore(hist_p[t], y[t], z[t, :, j : j + 1]))
                updates[:, :, j] = size[j] * z[:, :, j]
            p[:, :, 2] += updates[:, :, 1]
            transfer = transport.design(raw, p, ticks, geom, axis)[:, :, 2]
            history[2].append(v.price.rscore(p[t], y[t], transfer[t][:, :, None]))
            updates[:, :, 2] = size[2] * transfer
            p[:, :, 2] += updates[:, :, 2]
        else:
            p = q
        geometry.append(transport.geometry(q))
        assert np.isfinite(p).all()
        counts.append(int(len(p)))
        np.savez(OUT / f"day_{d:02d}.npz", prediction=p, allocated=updates, t=t)
    errors = {}
    for mod, hist in zip((prior, partial, transport), history):
        with np.load(mod.OUT / "model.npz") as f:
            error = float(abs(v.price.sizing(hist)[0] - f["size"]))
            assert error < 2e-6, (mod.__name__, error)
            errors[mod.__name__] = error
    (OUT / "report.json").write_text(
        json.dumps(dict(days=list(range(5, 17)), ticks=counts, parameter_error=errors), indent=2)
    )
    print("dense revision tape", errors, flush=True)


if __name__ == "__main__":
    main()
