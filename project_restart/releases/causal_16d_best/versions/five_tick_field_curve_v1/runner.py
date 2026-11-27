"""Use market-state persistence observed earlier in the same session to adjust transferred payoff horizons."""

import json
import numpy as np
import evidence as ref
from evidence import clock, base


def run(
    out,
    arms,
    train,
    views,
    *,
    keep_oof=False,
    prior_head=None,
    prepare_inputs=None,
    targets=(1, 2),
    check_leave_one_out=False,
):
    out.mkdir(exist_ok=True)
    parents = (
        ()
        if prior_head is None
        else prior_head
        if isinstance(prior_head, tuple)
        else (prior_head,)
    )
    predictions = {}
    history = []
    cal = []
    panels = {(k, h): [] for k in arms for h in targets}
    for d in range(16):
        with np.load(clock.clock.OUT / f"day_{d:02d}.npz") as f:
            v = [
                f[k]
                for k in (
                    "stock",
                    "market",
                    "stock_pay",
                    "market_pay",
                    "stock_var",
                    "market_var",
                )
            ]
        if d >= 6:
            if d in (6, 8, 12):
                params = train(history)
            with np.load(clock.live.data.OUT / f"day_{d:02d}.npz") as f:
                m = f["state"]
            with np.load(ref.tape.OUT / f"day_{d:02d}.npz") as f:
                q = f["reference"]
                dm = f["change"]
                fix = f["directions"][:, 2]
            prior = np.zeros(len(q))
            if d >= 8:
                for mod in (ref.precision, ref.nonlinear, clock):
                    change = np.load(mod.OUT / f"oof_{d:02d}.npy")
                    q += change[:, None]
                prior = change[:, 2]
            grid = np.load(base.CACHE / f"day_{d:02d}_time.npy").astype(int)
            y = np.load(base.CACHE / f"day_{d:02d}_y.npy", mmap_mode="r")
            controls = [dm, fix, prior]
            for head in parents:
                saved = (
                    np.load(head.OUT / f"oof_{d:02d}.npy")
                    if d >= 8
                    else np.zeros((len(m), 3), np.float32)
                )
                q += saved[:, None]
                controls.append(saved[:, 2])
            n_controls = 3 + len(controls)
            state = m
            if prepare_inputs is not None:
                raw = np.load(base.CACHE / f"day_{d:02d}_x.npy", mmap_mode="r")
                state = prepare_inputs(raw, grid, m, params)
            forecast = views(state, grid, params)
            z = np.column_stack((*controls, forecast))
            if d in (8, 12):
                w = {
                    key: ref.fit(cal, arms.index(key[0]), key[1], n_controls)
                    for key in panels
                }
            if d >= 8:
                xx = np.column_stack((q.mean(1, dtype=float), z))
                for (k, h), rows in panels.items():
                    change = xx @ w[k, h]
                    if keep_oof:
                        predictions[d, k, h] = change.astype(q.dtype)
                    step = np.broadcast_to(change[:, None], q.shape[:2])
                    rows.append(ref.market.metric(q[..., h], y[..., h], step))
                if d == 8:
                    for n in (333, 2501):
                        prefix = (
                            m[:n]
                            if prepare_inputs is None
                            else prepare_inputs(raw[:n], grid[:n], m[:n], params)
                        )
                        np.testing.assert_allclose(
                            forecast[:n],
                            views(prefix, grid[:n], params),
                            rtol=0,
                            atol=1e-8,
                        )
            cal.append(clock.live.stat(q, y, z))
        history.append(v)
    report = {
        f"{k}_{h}": dict(
            folds=[ref.market.gain(sum(v[:4])), ref.market.gain(sum(v[4:]))],
            days=[ref.market.gain(g) for g in v],
        )
        for (k, h), v in panels.items()
    }
    picks = {}
    for h in targets:
        ok = [
            k
            for k in arms
            if min(report[f"{k}_{h}"]["folds"]) > 0
            and sum(i > 0 for i in report[f"{k}_{h}"]["days"]) >= 5
        ]
        if check_leave_one_out:
            for k in arms:
                group = panels[k, h]
                drops = [
                    ref.market.gain(sum(fold) - day)
                    for fold in (group[:4], group[4:])
                    for day in fold
                ]
                report[f"{k}_{h}"]["leave_one_date_out"] = drops
            ok = [k for k in ok if min(report[f"{k}_{h}"]["leave_one_date_out"]) > 0]
        if ok:
            picks[str(h)] = max(ok, key=lambda k: min(report[f"{k}_{h}"]["folds"]))
    (out / "rolling.json").write_text(
        json.dumps(dict(selected=picks, results=report), indent=2)
    )
    print(picks, {k: v["folds"] for k, v in report.items()}, flush=True)
    if not picks:
        return
    if keep_oof:
        for day in range(8, 16):
            update = np.zeros(
                (len(predictions[day, arms[0], targets[0]]), 3), dtype=np.float32
            )
            for h, k in picks.items():
                update[:, int(h)] = predictions[day, k, int(h)]
            np.save(out / f"oof_{day:02d}.npy", update)
    w = np.zeros((n_controls + len(arms), 3))
    for h, k in picks.items():
        w[:, int(h)] = ref.fit(cal, arms.index(k), int(h), n_controls)
    np.savez(out / "weights.npz", w=w, **train(history))
