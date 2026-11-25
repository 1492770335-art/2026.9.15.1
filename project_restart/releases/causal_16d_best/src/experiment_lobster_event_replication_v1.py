"""Replicate the FI-2010 price-state transfer on independently sourced US equity quotes."""
import hashlib
import json
import zipfile
import numpy as np

import experiment_external_event_prior_v1 as trial

OUT = trial.fi.peer.ART/'lobster_event_replication_v1'
SYMBOLS = ('AAPL','AMZN','GOOG','INTC','MSFT')


def source():
    fi = trial.external()
    learned = trial.table(fi)
    marginal = fi[:,1:].sum(0)/fi[:,0].sum()
    records = []
    counts = []
    for symbol in SYMBOLS:
        path = trial.OUT/'lobster_samples'/f'{symbol}.zip'
        with zipfile.ZipFile(path) as z:
            name, = [k for k in z.namelist() if k.endswith('_orderbook_1.csv')]
            with z.open(name) as csv:
                ask,ask_size,bid,bid_size = np.loadtxt(csv,delimiter=',',unpack=True)
        valid = (ask>=bid)&(bid>0)&(ask_size>0)&(bid_size>0)&(ask<1.e9)
        runs = np.split(np.arange(len(bid)),np.flatnonzero(valid[:-1] != valid[1:])+1)
        delta = np.zeros(3)
        events = 0
        for idx in runs:
            if not valid[idx[0]] or len(idx)<20:
                continue
            mid = np.log(.5*(ask[idx]+bid[idx]))
            seq = trial.events(np.diff(mid,prepend=mid[0]),1.e-9)
            counts.append(trial.observations(seq))
            _,sign,code = seq
            t = np.arange(5,len(sign)-max(trial.HORIZONS))
            sums = np.r_[0.,sign.cumsum()]
            truth = np.column_stack([sign[t]*(sums[t+1+h]-sums[t+1])/np.sqrt(h) for h in trial.HORIZONS])
            delta += (((truth-marginal)**2-(truth-learned[code[t]])**2).sum(0))
            events += len(t)
        records.append(dict(symbol=symbol,book_rows=len(bid),events=events,
                            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            fi_mse_improvement=(delta/max(1,events)).tolist()))
    out = sum(counts)
    report = dict(date='2012-06-21',level=1,source='https://php.lobsterdata.com/info/DataSamples.php',
                  horizons=trial.HORIZONS,sequences=records,events=int(out[:,0].sum()))
    (OUT/'source_replication.json').write_text(json.dumps(report,indent=2))
    # Match the FI-2010 pseudo-observation count for a comparison of source markets.
    return out*min(1.,fi[:,0].sum()/out[:,0].sum())


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    print('replicated',trial.screen(OUT,source,trial.session))
