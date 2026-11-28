"""Average four complete partitions of the distal 40-tick payoff without double counting a boundary."""

import numpy as np
import experiment_dense_five_tick_field_curve_v1 as fine

fields, ref = fine.fields, fine.ref
OUT = ref.base.ROOT / "artifacts/dense_exact_cover_fields_v1"
ARMS = ("matched_duration", "subinterval_price", "matched_duration_and_price")
LAGS = tuple(range(-20, 60, 5))


def record(z, grid, core, y):
    cross = 0
    pay = 0
    count = 0
    ratio = ref.base.UNITS[0] / ref.base.UNITS[1]
    for a, b in fields.parent.sessions(grid):
        for t in np.array_split(np.arange(a + 40, b - 61, 8), 6):
            if not len(t):
                continue
            cur = z[t, ..., :core].reshape(-1, core).astype(float)
            width = z.shape[-1]
            cross += np.stack(
                [cur.T @ z[t + k].reshape(-1, width).astype(float) for k in LAGS]
            )
            hist = np.concatenate(
                (
                    z[t, ..., core:],
                    z[t, ..., :core],
                    z[t - 20, ..., :core],
                    z[t - 40, ..., :core],
                ),
                -1,
            )
            near = [ratio * y[t + k, ..., 0] for k in (0, 5, 10, 15)]
            full = y[t, ..., 1]
            # Two estimates per subinterval, using the prefix directly and subtracting the suffix.
            labels = np.stack(
                [0.5 * (sum(near[:j]) + full - sum(near[j:])) for j in (1, 2, 3)]
                + [full],
                -1,
            )
            pay += hist.reshape(len(cur), -1).astype(float).T @ labels.reshape(
                len(cur), 4
            )
            count += len(cur)
    return cross / count, pay / count


def collect():
    OUT.mkdir(exist_ok=True)
    code = fields.parent.codec()
    for d in range(16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = load("x"), load("time").astype(int), load("y")[..., :2]
        x, av = fields.parent.state(raw, grid, code, with_market=True)
        n = x.shape[-1]
        omitted = np.tanh(0.65 * np.arcsinh(raw[:, :, fields.COLUMNS].astype(float)))
        omitted -= omitted.mean(1, keepdims=True)
        z = np.concatenate((x, omitted.astype(np.float32)), -1)
        m = fields.book_inputs(raw, grid, ref.market.market_state(av, grid), {})
        common = y.mean(1, dtype=float)
        cs, ps = record(z, grid, n, y - common[:, None])
        cm, pm = record(m, grid, n, common)
        np.savez(
            OUT / f"day_{d:02d}.npz", cross=np.stack((cs, cm)), pay=np.stack((ps, pm))
        )
        print(f"Measured time partitions for day {d + 1}/16", flush=True)


def train(history):
    rows = []
    for d in range(len(history)):
        with (
            np.load(fields.OUT / f"moments_{d:02d}.npz") as f,
            np.load(OUT / f"day_{d:02d}.npz") as r,
        ):
            rows.append((f["cov"], f["variance"], r["cross"], r["pay"]))
    cov, var, cross, pay = map(np.stack, zip(*rows))
    units = np.sqrt(cov[:, :, 0].mean(0).diagonal(axis1=1, axis2=2)).clip(
        np.array([0.03, 0.006])[:, None]
    )
    n = units.shape[1] - len(fields.COLUMNS)
    in_unit = np.concatenate((units[:, n:], np.tile(units[:, :n], (1, 3))), 1)
    price_unit = np.array([np.sqrt(var[:, 1].mean() / var[:, 0].mean()), 1.0])
    cov = (cov / (units[:, :, None] * units[:, None, :])[:, None]).mean(1)
    cross = (cross / (units[:, :n, None] * units[:, None, :])[:, None]).mean(1)
    pay = (pay * (price_unit[:, None] / in_unit)[:, :, None]).mean(1)
    banks = []
    for side in (0, 1):
        k = cov[side::2].mean(0)
        lagged = cross[side::2].mean(0)
        prices = pay[1 - side :: 2].mean(0)

        def lag(a, b):
            return k[b - a] if b >= a else k[a - b].T

        past = (0, -1, -2)
        g = np.block([[lag(a, b)[:n, :n] for b in past] for a in past])
        to_field = np.vstack([lag(a, 0)[:n, n:] for a in past])
        pred = np.linalg.solve(g + 0.17 * np.eye(len(g)), to_field)
        residual = (
            k[0, n:, n:] - to_field.T @ pred - pred.T @ to_field + pred.T @ g @ pred
        )
        nf = len(fields.COLUMNS)
        decoder = np.linalg.solve(
            residual + 0.29 * np.eye(nf), prices[:nf] - pred.T @ prices[nf:]
        )

        def at(h):
            return lagged[LAGS.index(h)]

        response = {}
        for h in range(20, 60, 5):
            drift = (
                at(h)[:, n:] - np.hstack([at(h - j)[:, :n] for j in (0, 20, 40)]) @ pred
            )
            response[h] = np.linalg.solve(
                k[0, :n, :n] + 0.22 * np.eye(n), drift @ decoder
            )
        flat = sum(v[:, 3] for v in response.values()) / 8
        duration = np.zeros(n)
        priced = np.zeros(n)
        for offset in (0, 5, 10, 15):
            edges = [20, 40, 60] if offset == 0 else [20, 20 + offset, 40 + offset, 60]
            for start, end in zip(edges, edges[1:]):
                width = end - start
                duration += response[start][:, 3] * (width / 20) / 8
                priced += response[start][:, width // 5 - 1] / 8
        banks.append(
            np.column_stack((duration - flat, priced - duration, priced - flat))
            / units[1, :n, None]
        )
    return dict(bank=np.mean(banks, 0))


if __name__ == "__main__":
    collect()
    fine.memory.runner.run(
        OUT,
        ARMS,
        train,
        fine.views,
        prior_head=(fine.memory, fine.news, fine.path, fine),
        targets=(2,),
        keep_oof=True,
        check_leave_one_out=True,
    )
