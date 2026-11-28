"""Filter a continuous latent price with rounded or temporarily stale observed prices."""
import numpy as np
from scipy.signal import lfilter
import experiment_dense_two_view_price_imputation_v1 as proxy

onset, rel, book, REFERENCE = proxy.onset, proxy.rel, proxy.book, proxy.REFERENCE
OUT = REFERENCE.parent / "dense_subtick_observation_filter_v1"
ARMS = ("rounded_price", "stale_price_mixture", "effect_of_staleness")
NODES = np.linspace(-.45, .45, 10)
F = proxy.FIELDS


def increments(raw):
    with np.load(onset.settled.GEO) as codec:
        k, coef, missing = float(codec["k"]), codec["coeff"][2, 1], codec["row"]
    valid = ~np.all(raw == missing, axis=2)
    delta = np.log1p(k * (raw[:, :, 0].astype(float) - float(np.float16(.00136)))) * coef / k
    return delta * valid, valid


def tick_size(step, prior):
    out = prior.copy()
    for stock, sample in enumerate(abs(step).T):
        moves = sample[sample > .003]
        if len(moves) >= 12:
            seed = np.quantile(moves, .10)
            multiples = np.rint(moves / seed)
            near = (multiples >= 1) & (abs(moves / seed - multiples) < .14)
            if near.sum() >= 8:
                out[stock] = np.median(moves[near] / multiples[near])
    return out.clip(.003)


def collect():
    OUT.mkdir(exist_ok=True)
    for day in range(16):
        raw = onset.settled.load(day, "x")
        grid = onset.settled.load(day, "time")
        step, valid = increments(raw)
        unit = tick_size(step, np.full(50, .04))
        # A multi-tick innovation rate is less sensitive to immediate bid/ask reversals.
        total, count = np.zeros(50), np.zeros(50)
        for a, b in rel.fields.parent.sessions(grid):
            windows = (b-a)//8
            r = step[a:a + 8*windows].reshape(windows, 8, 50).sum(1)
            ok = valid[a:a + 8*windows].reshape(windows, 8, 50).all(1)
            total += (r.clip(-3., 3.)**2 * ok).sum(0)
            count += ok.sum(0)
        np.save(OUT / f"gram_{day:02d}.npy", np.column_stack((unit, total/(8*count.clip(1)))))


def train(end):
    day_stats = np.stack([np.load(OUT / f"gram_{d:02d}.npy") for d in range(end)])
    prior = np.median(day_stats, axis=0)
    w = proxy.train(end)
    return np.concatenate((w, prior[:, None, :]), axis=1)


def filter_session(step, valid, drift, change, unit, variance):
    ticks, stocks = step.shape
    mean = np.zeros((2, stocks))
    var = np.ones_like(mean) * unit**2 / 12
    out = np.zeros((ticks, stocks, 2))
    gain = 1 - np.exp(-1/180)
    live = lfilter([gain], [1, gain-1], step**2 - variance, axis=0) + variance
    for t in range(ticks):
        if t and t % 128 == 0:
            unit = tick_size(step[max(0,t-512):t], unit)
        noise = np.maximum(.4*variance + .6*live[t], (.18*unit)**2)
        pred = mean + drift[t] - step[t]
        pv = var + noise
        cells = unit[:, None] * NODES
        dist = (cells[None] - pred[:, :, None])**2 / pv[:, :, None]
        logw = -.5*dist
        maximum = logw.max(2)
        w = np.exp(logw - maximum[:, :, None])
        mass = w.sum(2)
        w /= mass[:, :, None]
        observed = np.sum(w * cells, axis=2)
        obs_var = np.sum(w*(cells[None]-observed[:, :, None])**2, axis=2)
        # Stale and rounded prices have different likelihoods on a zero move.
        # The second filter can retain uncertainty if few book fields changed.
        pround = (np.exp(maximum[1])*mass[1]/len(NODES)*unit/np.sqrt(2*np.pi*pv[1])).clip(0, 1)
        prior = .04 + .62*np.exp(-.45*change[t])
        stale = prior / (prior + (1-prior)*pround)
        stale *= abs(step[t]) < .003
        mix = observed[1] + stale*(pred[1] - observed[1])
        obs_var[1] = ((1-stale)*(obs_var[1]+(observed[1]-mix)**2) +
                      stale*(pv[1]+(pred[1]-mix)**2))
        observed[1] = mix
        mean = np.where(valid[t], observed, 0).clip(-8*unit, 8*unit)
        var = np.where(valid[t], obs_var, unit**2/12).clip(1e-9, (12*unit)**2)
        out[t] = mean.T
    return out


def observations(raw, delta, valid, bank):
    w, priors = bank[:, :-1], bank[:, -1]
    view = np.arcsinh(raw[:, :, F].astype(float))
    book_change = np.diff(view, axis=0, prepend=view[:1])
    # Use only other stocks for the price channel, and only non-price fields for the book channel.
    common = (delta.sum(1, keepdims=True) - delta) / 49
    market_weights = w[:, 0, 0] - w[:, 1:51, 0].sum(1)
    other = delta @ w[:, 1:51, 0].T + common*market_weights
    own = np.einsum("tsf,sf->ts", book_change, w[:, 51:, 1])
    drift = (other + own) * .5
    return filter_session(delta, valid, drift, (book_change != 0).sum(2), priors[:, 0], priors[:, 1])


def design(raw, grid, code, pred, bank):
    step, valid = increments(raw)
    out = np.zeros((*pred.shape[:2], len(ARMS), 3))
    for a, b in rel.fields.parent.sessions(grid):
        z = observations(raw[a:b], step[a:b], valid[a:b], bank)
        out[a:b] = np.concatenate((z, z[:, :, 1:2]-z[:, :, :1]), axis=2)[:, :, :, None]
    return out - out.mean(1, keepdims=True)


def predict(raw, grid, p, params):
    return book.path.corrected(p, design(raw, grid, params["code"], p, params["bank"]), params)


if __name__ == "__main__":
    collect()
    onset.rolling(OUT, ARMS, train, design, REFERENCE)
