"""Forecast the released five-tick field payoff from current noncore fields and earlier core observations."""

import numpy as np
import experiment_dense_five_tick_field_curve_v1 as fine

fields, ref = fine.fields, fine.ref
OUT = ref.base.ROOT / "artifacts/dense_full_state_field_path_v1"
ARMS = ("current_fields", "previous_core", "both_inputs")
prepare_inputs = fields.book_inputs


def statistics(z, grid, core):
    gram = 0
    cross = 0
    count = 0
    for a, b in fields.parent.sessions(grid):
        block = z[a:b]
        cumulative = np.empty_like(block, dtype=np.float64)
        for phase in range(5):
            np.cumsum(
                block[phase::5], axis=0, dtype=np.float64, out=cumulative[phase::5]
            )

        def average(end):
            value = cumulative[end].copy()
            after = end >= 40
            value[after] -= cumulative[end[after] - 40]
            return value / 8

        for t in np.array_split(np.arange(40, b - a - 61, 8), 6):
            if not len(t):
                continue
            # All features at the forecast origin are already available.
            x = np.concatenate((block[t], block[t - 20, ..., :core]), -1)
            x = x.reshape(-1, x.shape[-1]).astype(float)
            future = average(t + 55)
            y = np.concatenate(
                (
                    future[..., core:],
                    future[..., :core],
                    average(t + 35)[..., :core],
                    average(t + 15)[..., :core],
                ),
                -1,
            )
            gram += x.T @ x
            cross += x.T @ y.reshape(len(x), -1)
            count += len(x)
    return gram / count, cross / count


def collect():
    OUT.mkdir(exist_ok=True)
    code = fields.parent.codec()
    for d in range(16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid = load("x"), load("time").astype(int)
        x, av = fields.parent.state(raw, grid, code, with_market=True)
        core = x.shape[-1]
        extra = np.tanh(0.65 * np.arcsinh(raw[:, :, fields.COLUMNS].astype(float)))
        extra -= extra.mean(1, keepdims=True)
        z = np.concatenate((x, extra.astype(np.float32)), axis=-1)
        market = fields.book_inputs(raw, grid, ref.market.market_state(av, grid), {})
        st, st_cross = statistics(z, grid, core)
        mk, mk_cross = statistics(market, grid, core)
        np.savez(
            OUT / f"day_{d:02d}.npz",
            grams=np.stack((st, mk)),
            cross=np.stack((st_cross, mk_cross)),
        )
        print(f"Measured training day {d + 1}/16", flush=True)


def train(history):
    moments = []
    for d in range(len(history)):
        with (
            np.load(OUT / f"day_{d:02d}.npz") as f,
            np.load(fields.OUT / f"moments_{d:02d}.npz") as sf,
        ):
            moments.append((f["grams"], f["cross"], sf["cov"][:, 0]))
    gram, cross, original = map(np.stack, zip(*moments))
    units = np.sqrt(original.mean(0).diagonal(axis1=1, axis2=2)).clip(
        np.array([0.03, 0.006])[:, None]
    )
    n = units.shape[1] - len(fields.COLUMNS)
    inputs = np.concatenate((units, units[:, :n]), 1)
    outputs = np.concatenate((units[:, n:], np.tile(units[:, :n], (1, 3))), 1)
    gram = (gram / (inputs[:, :, None] * inputs[:, None, :])).mean((0, 1))
    cross = (cross / (inputs[:, :, None] * outputs[:, None, :])).mean((0, 1))
    decoder = fine.news.train(history, teacher_decoder=True)["bank"]
    rhs = cross @ decoder

    def fit(columns):
        k = gram[np.ix_(columns, columns)]
        w = np.zeros(len(gram))
        w[columns] = np.linalg.solve(k + 0.22 * np.eye(len(columns)), rhs[columns])
        return w

    present = np.arange(units.shape[1])
    core = np.arange(n)
    older = np.arange(len(present), len(gram))
    baseline = fit(core)
    bank = np.column_stack(
        [
            fit(cols) - baseline
            for cols in (present, np.r_[core, older], np.arange(len(gram)))
        ]
    )
    return dict(bank=bank / inputs[1, :, None])


def views(state, grid, op):
    bank = op["bank"]
    n = bank.shape[0] - state.shape[1]
    out = np.zeros((len(state), len(ARMS)))
    for a, b in fields.parent.sessions(grid):
        t = np.arange(a + 40, b)
        out[t] = (
            state[t] @ bank[: state.shape[1]]
            + state[t - 20, :n] @ bank[state.shape[1] :]
        )
    return out


if __name__ == "__main__":
    collect()
    fine.memory.runner.run(
        OUT,
        ARMS,
        train,
        views,
        prior_head=(fine.memory, fine.news, fine.path, fine),
        prepare_inputs=prepare_inputs,
        targets=(2,),
        keep_oof=True,
        check_leave_one_out=True,
    )
