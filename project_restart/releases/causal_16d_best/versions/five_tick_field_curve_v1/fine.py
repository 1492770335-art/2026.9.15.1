"""Resolve the 21-60 tick market payoff into eight shorter responses to omitted book innovations."""

import numpy as np
import paired
from paired import path, memory, news, ref, fields

OUT = ref.base.ROOT / "artifacts/dense_five_tick_field_curve_v1"
ARMS = ("short_payoff", "fine_path", "joint_quadrature")


def observe(z, grid, y, core):
    cross = 0
    pay = 0
    count = 0
    pay_count = 0
    for a, b in fields.parent.sessions(grid):
        if b - a <= 61:
            continue
        block = z[a:b].astype(float)
        # A trailing sum on each five-tick lattice gives all eight future samples without repeated matrix products.
        cumulative = block.copy()
        for phase in range(5):
            np.cumsum(block[phase::5], axis=0, out=cumulative[phase::5])

        def average(end):
            value = cumulative[end].copy()
            present = end >= 40
            value[present] -= cumulative[end[present] - 40]
            return value / 8

        for t in np.array_split(np.arange(20, b - a - 61, 8), 5):
            if not len(t):
                continue
            future = average(t + 55)
            target = np.concatenate(
                (
                    future[..., core:],
                    future[..., :core],
                    average(t + 35)[..., :core],
                    average(t + 15)[..., :core],
                ),
                -1,
            )
            now = block[t, ..., :core].reshape(-1, core)
            cross += now.T @ target.reshape(-1, target.shape[-1])
            count += len(now)
        for t in np.array_split(np.arange(40, b - a - 61, 8), 5):
            if not len(t):
                continue
            hist = np.concatenate(
                (
                    block[t, ..., core:],
                    block[t, ..., :core],
                    block[t - 20, ..., :core],
                    block[t - 40, ..., :core],
                ),
                -1,
            )
            labels = y[a + t].ravel().astype(float)
            pay += hist.reshape(len(labels), -1).T @ labels
            pay_count += len(labels)
    return cross / count, pay / pay_count


def collect():
    OUT.mkdir(exist_ok=True)
    code = fields.parent.codec()
    for day in range(16):
        load = lambda k: np.load(
            ref.base.CACHE / f"day_{day:02d}_{k}.npy", mmap_mode="r"
        )
        raw, grid, y = load("x"), load("time").astype(int), load("y")[..., 0]
        x, av = fields.parent.state(raw, grid, code, with_market=True)
        n = x.shape[-1]
        m = fields.book_inputs(raw, grid, ref.market.market_state(av, grid), {})
        omitted = np.tanh(0.65 * np.arcsinh(raw[:, :, fields.COLUMNS].astype(float)))
        omitted -= omitted.mean(1, keepdims=True)
        z = np.concatenate((x, omitted.astype(np.float32)), axis=-1)
        common = y.mean(1, dtype=float)
        bs, ps = observe(z, grid, y - common[:, None], n)
        bm, pm = observe(m, grid, common, n)
        np.savez(OUT / f"day_{day:02d}.npz", bs=bs, bm=bm, ps=ps, pm=pm)


def train(history):
    data = []
    for d in range(len(history)):
        with (
            np.load(fields.OUT / f"moments_{d:02d}.npz") as f,
            np.load(paired.OUT / f"day_{d:02d}.npz") as c,
            np.load(OUT / f"day_{d:02d}.npz") as a,
        ):
            data.append(
                (
                    f["cov"],
                    f["pay"],
                    f["variance"],
                    np.load(news.delayed.OUT / f"payoffs_{d:02d}.npy"),
                    c["stock"],
                    c["market"],
                    a["bs"],
                    a["bm"],
                    a["ps"],
                    a["pm"],
                )
            )
    grams, pays, var, delays, ls, lm, bs, bm, ps, pm = [np.stack(x) for x in zip(*data)]
    sd = np.sqrt(grams[:, 1, 0].mean(0).diagonal()).clip(0.006)
    ss = np.sqrt(grams[:, 0, 0].mean(0).diagonal()).clip(0.03)
    n = len(sd) - len(fields.COLUMNS)
    nf = len(sd) - n
    scale = np.sqrt(var[:, 1].mean() / var[:, 0].mean())
    cov = 0.5 * (grams[:, 0] / np.outer(ss, ss) + grams[:, 1] / np.outer(sd, sd))
    pay = 0.5 * (pays[:, 0] * scale / ss + pays[:, 1] / sd)
    past = 0.5 * (delays[:, 0, :, 0] * scale / ss[:n] + delays[:, 1, :, 0] / sd[:n])
    denom = lambda s: np.r_[s[n:], np.tile(s[:n], 3)]
    xs, xm = np.outer(ss[:n], denom(ss)), np.outer(sd[:n], denom(sd))
    fine = 0.5 * (bs / xs + bm / xm)
    coarse = 0.25 * (ls / xs + lm / xm)
    short = (
        0.5
        * (ps * scale / denom(ss) + pm / denom(sd))
        * (4 * ref.base.UNITS[0] / ref.base.UNITS[1])
    )
    banks = []
    for j in (0, 1):
        first, other = slice(j, None, 2), slice(1 - j, None, 2)
        k = cov[first].mean(0)

        def cross(a, b):
            return k[b - a] if b >= a else k[a - b].T

        times = (0, -1, -2)
        ctx = np.block([[cross(a, b)[:n, :n] for b in times] for a in times])
        field = np.concatenate([cross(t, 0)[:n, n:] for t in times])
        proj = np.linalg.solve(ctx + 0.17 * np.eye(len(ctx)), field)
        er = k[0, n:, n:] - field.T @ proj - proj.T @ field + proj.T @ ctx @ proj
        old_y = (
            pay[other, n:].mean(0)
            - proj.T @ np.r_[pay[other, :n].mean(0), past[other].mean(0).ravel()]
        )
        short_y = short[other].mean(0)
        read = np.linalg.solve(
            er + 0.29 * np.eye(nf),
            np.column_stack((old_y, short_y[:nf] - proj.T @ short_y[nf:])),
        )

        def bridge(rows):
            avg = rows[first].mean(0)
            return np.linalg.solve(
                k[0, :n, :n] + 0.22 * np.eye(n), avg[:, :nf] - avg[:, nf:] @ proj
            )

        current, later = bridge(coarse), bridge(fine)
        old = current @ read[:, 0]
        banks.append(
            np.column_stack(
                (
                    current @ read[:, 1] - old,
                    later @ read[:, 0] - old,
                    later @ read[:, 1] - old,
                )
            )
            / sd[:n, None]
        )
    return dict(bank=np.mean(banks, 0))


def views(m, grid, f):
    z = m @ f["bank"]
    for a, b in fields.parent.sessions(grid):
        z[a : min(a + 40, b)] = 0
    return z
