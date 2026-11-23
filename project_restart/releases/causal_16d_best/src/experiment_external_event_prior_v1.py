"""Compare internal and FI-2010 priors for reversals on a stock's price-change clock."""
import hashlib
import json
import zipfile

import numpy as np

import experiment_crossfit_decay_contrasts_v1 as parent
import experiment_fi2010_transfer_v1 as fi

OUT = fi.peer.ART / 'external_event_prior_v1'
HORIZONS = (1, 4, 12)
STATES = 128


def events(r, threshold):
    ix = np.flatnonzero(abs(r) > threshold)
    moves = r[ix]
    sign = np.sign(moves)
    event = np.arange(len(ix))
    code = np.zeros(len(event), dtype=int)
    for lag in range(1, 6):
        code = 2 * code + (sign == sign[(event-lag).clip(0)])
    # Compare consecutive inter-event durations and move magnitudes within the same stock.
    gap = np.diff(ix, prepend=0)
    previous = (event-1).clip(0)
    code = 2*code + (gap > gap[previous])
    code = 2*code + (abs(moves) > 1.1*abs(moves[previous]))
    return ix, sign, code


def observations(seq):
    _, sign, code = seq
    t = np.arange(5, len(code)-max(HORIZONS))
    count = np.bincount(code[t], minlength=STATES)
    sums = np.r_[0., sign.cumsum()]
    rhs = [np.bincount(code[t], weights=sign[t]*(sums[t+1+h]-sums[t+1])/np.sqrt(h),
                       minlength=STATES) for h in HORIZONS]
    return np.column_stack((count, *rhs))


def external(encode=None):
    manifest = json.loads((fi.OUT/'external_data.json').read_text())
    assert hashlib.sha256(fi.ARCHIVE.read_bytes()).hexdigest() == manifest['sha256']
    stats = []
    with zipfile.ZipFile(fi.ARCHIVE) as archive:
        for name in sorted({s['file'] for s in manifest['sequences']}):
            with archive.open(name) as data:
                ask = np.fromstring(data.readline(), sep=' ')
                data.readline()
                bid = np.fromstring(data.readline(), sep=' ')
            mid = np.log((ask+bid)/2)
            for s in [s for s in manifest['sequences'] if s['file'] == name]:
                prices = mid[s['first']:s['first']+s['rows']]
                r = np.diff(prices, prepend=prices[0])
                start, end = s['first'], s['first']+s['rows']
                stream = events(r,1e-9) if encode is None else encode(r,ask[start:end]-bid[start:end],1e-9)
                stats.append(observations(stream))
    return sum(stats)


def table(stat):
    count = stat[:, 0:1]
    center = stat[:, 1:].sum(0)/max(count.sum(), 1)
    return (stat[:, 1:]+48*center)/(count+48)


def session(d):
    r = parent.base.paths.book.rawday(d)[:, :, 0].astype(float)-.00136
    return tuple(events(col, .022) for col in r.T), len(r)


def design(streams, t, prior):
    columns = []
    for ix, sign, code in streams:
        if not len(ix):
            columns.append(np.zeros((len(t), 7)))
            continue
        last = np.searchsorted(ix, t, side='right')-1
        valid = last >= 5
        k = last.clip(0)
        past_gap = (ix[k]-ix[(k-5).clip(0)])/5
        reliability = 1/(1+(t-ix[k]).clip(0)/(1+past_gap))
        s = sign[k]*valid
        values = prior[code[k]]*s[:, None]
        columns.append(np.column_stack((s,values,values*reliability[:,None])))
    x = np.stack(columns, axis=1)
    return x-x.mean(1, keepdims=True)


def fit(days, prior):
    grams, rhs = [], []
    for d, (stream, length) in enumerate(days):
        t = np.arange(d%9, length-60, 9)
        x = design(stream, t, prior).reshape(-1,7)
        y = fi.peer.labels(d).reshape(-1,50,3)[t].astype(float)
        y -= y.mean(1, keepdims=True)
        grams.append(x.T@x/len(x))
        rhs.append(x.T@y.reshape(-1,3)/len(x))
    g = np.mean(grams,axis=0)
    return np.linalg.solve(g+.2*np.diag(g.diagonal().clip(.0001)),np.mean(rhs,axis=0))


def estimate(days, counts, external_counts):
    # Identical number of prior pseudo-observations in both arms. Only their source differs.
    inner = sum(counts)
    effective = min(inner[:,0].sum(),external_counts[:,0].sum())
    priors = [table(inner + s*(effective/s[:,0].sum())) for s in (inner,external_counts)]
    return [(p,fit(days,p)) for p in priors]


def predictions(stream, t, models):
    return [design(stream,t,p)@w for p,w in models]


def screen(out=OUT, source=external, read_session=session):
    ext = source()
    days, counts, folds, daily, individual = [], [], [], [], []
    for end in (8,10,12,14):
        for d in range(len(days),end):
            stream,n = read_session(d)
            days.append((stream,n))
            counts.append(sum(observations(seq) for seq in stream))
        models = estimate(days, counts, ext)
        groups = [[],[]]
        inputs = []
        for d in (end,end+1):
            stream,n = read_session(d)
            t = np.arange(d%4,n,4)
            a,b = predictions(stream,t,models)
            if d == end:
                short = tuple(tuple(x[seq[0]<213] for x in seq) for seq in stream)
                for pred, full in zip(predictions(short,t[t<213],models),(a,b)):
                    np.testing.assert_allclose(pred,full[t<213],rtol=0,atol=1.e-10)
            y = fi.peer.labels(d).reshape(-1,50,3)[t]
            inputs.append((a,b,y-y.mean(1,keepdims=True)))
            for rows, ref in zip(groups,(fi.peer.reference,parent.fixed.earlier.early.baseline)):
                p = ref(d).reshape(-1,50,3)[t]
                rows.append(tuple(v.reshape(-1,3) for v in (p,p+.2*(b-a),y)))
            daily.append([fi.peer.delta(rows[-1:]) for rows in groups])
        a,b,y = (np.concatenate(vals) for vals in zip(*inputs))
        individual.append([fi.ic(a,y),fi.ic(b,y)])
        folds.append([fi.peer.delta(rows) for rows in groups])
        print('window',end+1,folds[-1],flush=True)
    keep = (np.asarray(folds).min(axis=(0,1))>0)&((np.asarray(daily)[:,0]>0).sum(0)>=5)
    (out/'results.json').write_text(json.dumps(dict(external_events=int(ext[:,0].sum()),folds=folds,
           daily=daily,standalone_relative_ic=individual,retained=keep.tolist()),indent=2))
    if keep.any():
        for d in (14,15):
            stream,n = read_session(d)
            days.append((stream,n))
            counts.append(sum(observations(seq) for seq in stream))
        models = estimate(days,counts,ext)
        np.savez(out/'model.npz',priors=[p for p,w in models],weights=[w for p,w in models],mask=keep)
    return keep.any()


def load():
    incumbent = parent.load()
    with np.load(OUT/'model.npz') as file:
        models = list(zip(file['priors'],file['weights']))
        keep = file['mask']
    def predict(raw,upstream):
        q = incumbent(raw,upstream).copy()
        stream = [events(col.astype(float)-.00136,.022) for col in raw[:,:,0].T]
        a,b = predictions(stream,np.arange(len(raw)),models)
        q += .2*(b-a)*keep
        return q
    return predict


def replay():
    previous = json.loads((parent.OUT/'raw_replay.json').read_text())['pooled']['score']
    first = None
    for name in ('raw_replay','reload_verification'):
        result = parent.base.peer.walk.replay(load())
        if first is not None:
            np.testing.assert_allclose(result['pooled']['candidate'], first['pooled']['candidate'],atol=1e-10,rtol=0)
            for d in result['daily']:
                assert abs(first['daily'][d]['score']-result['daily'][d]['score'])<1e-10
        (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(name,result['pooled']['score'],flush=True)
        if result['pooled']['score'] <= previous:
            break
        first = result


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    if screen():
        replay()
