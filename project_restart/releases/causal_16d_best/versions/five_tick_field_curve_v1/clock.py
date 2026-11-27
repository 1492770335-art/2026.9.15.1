"""Match the decay of individual-stock book modes to that of market modes before borrowing payoffs."""

import json
import numpy as np
import clock_data as clock
import nonlinear
from nonlinear import live, market, tape, base, precision

OUT = base.ROOT / "artifacts/dense_cross_scale_clock_transfer_v1"
ARMS = ("payoff_clock", "transition_clock", "joint_clock")


def select_weights(src, target, lag, candidates):
    # Both covariance curves use the same frozen, variance-normalized coordinates.
    error = (src - target[None, :]) / 0.10
    log_prior = (np.log(np.asarray(candidates) / lag) / 0.80)[:, None]
    cost = error**2 + log_prior**2
    probs = np.exp(-0.5 * (cost - cost.min(0)))
    return probs / probs.sum(0)


def operator(hist):
    st, mk, sy, my, sv, mv = [np.stack(z) for z in zip(*hist)]
    sd = np.sqrt(np.diag(mk[:, 0].mean(0))).clip(0.006)
    ss = np.sqrt(np.diag(st[:, 0].mean(0))).clip(0.03)
    h = 0.5 * (
        mk[:, 0].mean(0) / np.outer(sd, sd) + st[:, 0].mean(0) / np.outer(ss, ss)
    )
    val, u = np.linalg.eigh(h + 0.25 * np.eye(len(h)))
    whiten = u / np.sqrt(val)
    rm = whiten / sd[:, None]
    rs = whiten / ss[:, None]
    common = (
        rm.T @ (mk[:, 3:5].mean((0, 1))) @ rm + rs.T @ st[:, 3:5].mean((0, 1)) @ rs
    ) / 2
    _, axis = np.linalg.eigh((common + common.T) * 0.5)
    rm = rm @ axis
    rs = rs @ axis
    Ms = rs.T @ st @ rs
    Mm = rm.T @ mk @ rm
    stock_pay = rs.T @ sy
    market_pay = rm.T @ my
    varratio = np.sqrt(mv[:, 2].mean() / sv.mean(0))
    indices = list(clock.LAGS)
    results = []
    diagnostic = []
    for j in (0, 1):
        # Estimate feature clocks and responses on disjoint training dates.
        source = Ms[j::2].mean(0)
        dest = Mm[j::2].mean(0)
        ds = source.diagonal(axis1=1, axis2=2)
        dm = dest.diagonal(axis1=1, axis2=2)
        rho = ds / ds[0].clip(0.008)
        target = dm / dm[0].clip(0.008)
        sym = lambda a: 0.5 * (a + a.T)
        fixed = warped = 0.0
        for h in (20, 40):
            idx = indices.index(h)
            weights = select_weights(rho[1:], target[idx], h, indices[1:])
            adjusted = np.einsum("hi,hij->ij", weights, source[1:])
            fixed += sym(0.5 * (source[idx] + dest[idx]))
            warped += sym(0.5 * (adjusted + dest[idx]))
            diagnostic.append(
                float(np.median(np.sum(weights * np.array(indices[1:])[:, None], 0)))
            )
        # Interpolate only the training lag curve when a return horizon is between two measured lags.
        pay_rho = np.stack(
            [
                np.array(
                    [np.interp(h, indices, rho[:, i]) for i in range(rho.shape[1])]
                )
                for h in clock.HORIZONS
            ]
        )
        weights = select_weights(pay_rho, target[indices.index(20)], 20, clock.HORIZONS)
        other = slice(1 - j, None, 2)
        pay = stock_pay[other].mean(0) * varratio
        m = market_pay[other, :, 2].mean(0)
        old = 0.5 * (m + pay[:, 2])
        new = 0.5 * (m + (weights.T * pay).sum(1))
        origin = fixed @ old
        results.append(
            rm
            @ np.column_stack(
                (fixed @ new - origin, warped @ old - origin, warped @ new - origin)
            )
        )
    return np.mean(results, 0), diagnostic


def main():
    OUT.mkdir(exist_ok=True)
    signals = {}
    hist = []
    cal = []
    rows = {(k, h): [] for k in ARMS for h in (1, 2)}
    diag = {}
    for d in range(16):
        with np.load(clock.OUT / f"day_{d:02d}.npz") as f:
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
                bank, diag[str(d)] = operator(hist)
            with np.load(live.data.OUT / f"day_{d:02d}.npz") as f:
                m = f["state"]
            with np.load(tape.OUT / f"day_{d:02d}.npz") as f:
                q = f["reference"]
                dm = f["change"]
                fixed = f["directions"][:, 2]
            if d >= 8:
                q += np.load(precision.OUT / f"oof_{d:02d}.npy")[:, None]
                q += np.load(nonlinear.OUT / f"oof_{d:02d}.npy")[:, None]
            y = np.load(base.CACHE / f"day_{d:02d}_y.npy", mmap_mode="r")
            z = np.column_stack((dm, fixed, m @ bank))
            if d in (8, 12):
                w = {key: live.fit(cal, ARMS.index(key[0]), key[1]) for key in rows}
            if d >= 8:
                xx = np.column_stack((q.mean(1, dtype=float), z))
                for (k, h), values in rows.items():
                    update = xx @ w[k, h]
                    signals[d, k, h] = update.astype("float32")
                    dz = np.broadcast_to(update[:, None], q.shape[:2])
                    values.append(market.metric(q[..., h], y[..., h], dz))
            cal.append(live.stat(q, y, z))
        hist.append(v)
    report = {
        f"{k}_{h}": dict(
            folds=[market.gain(sum(v[:4])), market.gain(sum(v[4:]))],
            days=[market.gain(g) for g in v],
        )
        for (k, h), v in rows.items()
    }
    picks = {}
    for h in (1, 2):
        ok = [
            k
            for k in ARMS
            if min(report[f"{k}_{h}"]["folds"]) > 0
            and sum(i > 0 for i in report[f"{k}_{h}"]["days"]) >= 5
        ]
        if ok:
            picks[str(h)] = max(ok, key=lambda k: min(report[f"{k}_{h}"]["folds"]))
    (OUT / "rolling.json").write_text(
        json.dumps(dict(selected=picks, results=report, estimated_lags=diag), indent=2)
    )
    print(picks, {k: v["folds"] for k, v in report.items()}, diag, flush=True)
    if not picks:
        return
    for day in range(8, 16):
        n = next(len(z) for (d, _, _), z in signals.items() if d == day)
        changes = np.zeros((n, 3), np.float32)
        for h, k in picks.items():
            changes[:, int(h)] = signals[day, k, int(h)]
        np.save(OUT / f"oof_{day:02d}.npy", changes)
    w = np.zeros((8, 3))
    for h, k in picks.items():
        w[:, int(h)] = live.fit(cal, ARMS.index(k), int(h))
    bank, _ = operator(hist)
    np.savez(OUT / "weights.npz", bank=bank, w=w)
