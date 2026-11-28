"""Predict stock-relative future field values with stock-relative current and historical observations."""

import json
import numpy as np
import experiment_dense_full_state_field_path_v1 as full
import experiment_dense_recipient_clocks_v1 as relative

fine, fields, ref = full.fine, full.fields, full.ref
OUT = ref.base.ROOT / "artifacts/dense_relative_path_inputs_v1"
ARMS = ("current_noncore", "previous_core", "joint_context")


def train(days):
    rows = []
    for d in range(days):
        with (
            np.load(full.OUT / f"day_{d:02d}.npz") as f,
            np.load(fields.OUT / f"moments_{d:02d}.npz") as g,
        ):
            rows.append((f["grams"][0], f["cross"][0], g["cov"][0, 0], g["variance"]))
    gram, cross, cov, var = map(np.stack, zip(*rows))
    units = np.sqrt(cov.mean(0).diagonal()).clip(0.03)
    n = len(units) - len(fields.COLUMNS)
    predictor_scale = np.r_[units, units[:n]]
    future_scale = np.r_[units[n:], np.tile(units[:n], 3)]
    gram = gram / np.outer(predictor_scale, predictor_scale)
    cross = cross / np.outer(predictor_scale, future_scale)
    teacher = fine.news.train(range(days), stock_fraction=1.0, teacher_decoder=True)[
        "bank"
    ]
    # Undo the conversion to common-market payoff units used by the shared field teacher.
    teacher *= np.sqrt(var[:, 0].mean() / var[:, 1].mean())
    bank = []
    for j in (0, 1):
        g = gram[j::2].mean(0)
        rhs = cross[1 - j :: 2].mean(0) @ teacher

        def solve(indices):
            weights = np.zeros(len(g))
            xx = g[np.ix_(indices, indices)]
            weights[indices] = np.linalg.solve(
                xx + 0.22 * np.eye(len(xx)), rhs[indices]
            )
            return weights

        baseline = solve(np.arange(n))
        sets = (
            np.arange(len(units)),
            np.r_[0:n, len(units) : len(g)],
            np.arange(len(g)),
        )
        bank.append(
            np.column_stack([solve(ids) - baseline for ids in sets])
            / predictor_scale[:, None]
        )
    return np.mean(bank, 0)


def signal(raw, grid, code, bank):
    x = fields.parent.state(raw, grid, code)
    n = x.shape[-1]
    nf = len(fields.COLUMNS)
    out = np.zeros((*x.shape[:2], len(ARMS)))
    for a, b in fields.parent.sessions(grid):
        for t in np.array_split(np.arange(a + 40, b), 6):
            if not len(t):
                continue
            extra = np.tanh(
                0.65 * np.arcsinh(raw[t][:, :, fields.COLUMNS].astype(float))
            )
            extra -= extra.mean(1, keepdims=True)
            out[t] = (
                x[t] @ bank[:n] + extra @ bank[n : n + nf] + x[t - 20] @ bank[n + nf :]
            )
    return out - out.mean(1, keepdims=True)


def rolling_short_changes():
    import experiment_dense_count_weighted_quotes_v1 as short

    history = []
    updates = {}
    code = short.codec()
    for d in range(4, 16):

        def load(key):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{key}.npy", mmap_mode="r")

        raw, grid = load("x"), load("time").astype(int)
        forecast = np.load(short.TAPE / f"forecast_{d:02d}.npy")
        if d in (8, 12):
            weights = short.train(history, 0)
        if d >= 8:
            changes = short.apply(raw, grid, forecast, code, {"market": weights})
            updates[d] = changes["market"][:, 0, 0]
        history.append(short.record(raw, grid, forecast, load("y"), code))
    with np.load(short.OUT / "weights.npz") as release:
        np.testing.assert_allclose(
            short.train(history, 0)[:, 0], release["market"][:, 0], atol=1e-8, rtol=0
        )
    return updates


def main(
    out=OUT,
    arms=ARMS,
    train_bank=train,
    generate=signal,
    *,
    with_prediction=False,
    target=2,
    rolling_reference=None,
):
    out.mkdir(exist_ok=True)
    code = fields.parent.codec()

    def inputs(raw, grid, p, bank):
        if with_prediction:
            return generate(raw, grid, code, bank, p)
        return generate(raw, grid, code, bank)

    short_changes = rolling_short_changes() if target == 0 else {}
    cal = []
    groups = {k: [] for k in arms}
    for day in range(6, 16):
        if day in (6, 8, 12):
            bank = train_bank(day)

        def load(k):
            return np.load(ref.base.CACHE / f"day_{day:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = load("x"), load("time").astype(int), load("y")
        if rolling_reference is not None:
            p = np.load(rolling_reference / f"rolling_{day:02d}.npy")
        else:
            with np.load(ref.tape.OUT / f"day_{day:02d}.npz") as f:
                p = f["reference"]
            if day >= 8:
                for mod in (
                    ref.precision,
                    ref.nonlinear,
                    ref.clock,
                    fine.memory,
                    fine.news,
                    fine.path,
                    fine,
                ):
                    p += np.load(mod.OUT / f"oof_{day:02d}.npy")[:, None]
        if day in short_changes:
            p[..., 0] += short_changes[day][:, None]
        z = inputs(raw, grid, p, bank)
        if day in (8, 12):
            weights = [relative.fit(cal, j, target) for j in range(len(arms))]
        if day >= 8:
            xx = relative.features(p, z)
            for k, w in zip(arms, weights):
                groups[k].append(
                    ref.market.metric(p[..., target], y[..., target], xx @ w)
                )
        if day == 8:
            np.testing.assert_allclose(
                inputs(raw[:512], grid[:512], p[:512], bank), z[:512], rtol=0, atol=1e-7
            )
        cal.append(relative.stats(p, y, z))
    checks = {}
    for k, s in groups.items():
        folds = (s[:4], s[4:])
        checks[k] = dict(
            folds=[ref.market.gain(sum(g)) for g in folds],
            days=[ref.market.gain(x) for x in s],
            omit_one=[ref.market.gain(sum(g) - x) for g in folds for x in g],
        )
    passed = [
        k
        for k, v in checks.items()
        if min(v["omit_one"]) > 0
        and min(v["folds"]) > 0
        and sum(x > 0 for x in v["days"]) >= 5
    ]
    pick = max(passed, key=lambda k: min(checks[k]["folds"])) if passed else None
    (out / "rolling.json").write_text(
        json.dumps(dict(selected=pick, target=target, results=checks), indent=2)
    )
    print(pick, {k: v["folds"] for k, v in checks.items()}, flush=True)
    if pick:
        np.savez(
            out / "weights.npz",
            bank=train_bank(16),
            code=code,
            w=relative.fit(cal, arms.index(pick), target),
        )


if __name__ == "__main__":
    main()
