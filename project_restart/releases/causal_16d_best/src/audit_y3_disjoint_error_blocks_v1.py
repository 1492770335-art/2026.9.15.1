"""Measure repeatability of stock-relative offset and slope errors using nonoverlapping block outcomes in days 9-16."""
import json
import numpy as np
import y3_intraday_replication_references as history

base = history.base
WIDTH = 180
LAGS = (1,2,4,7)


def profile(p,y):
    facets = []
    for k in (0,1):
        ref = p[:,:,2]-k*base.RATIO*p[:,:,1]
        truth = y[:,:,2]-k*base.RATIO*y[:,:,1]
        ref = ref-ref.mean(1,keepdims=True)
        truth = truth-truth.mean(1,keepdims=True)
        # Descriptive scaling only; no resulting daily slope is used to produce a prediction.
        beta = (ref*truth).sum(dtype=float)/np.square(ref).sum(dtype=float)
        residual = truth/beta.clip(.35,2.8)-ref
        expo = np.tanh(ref/np.sqrt(.01+np.square(ref).mean(1,keepdims=True)))
        facets.append(np.stack((residual,expo*residual),2))
    values = np.stack(facets,3)
    panels = []
    for lo,hi in base.segments(len(p)):
        bins = [values[t:t+WIDTH-60:4].mean(0) for t in range(lo,hi-WIDTH+1,WIDTH)]
        v = np.stack(bins)
        panels.append(v-v.mean(1,keepdims=True))
    return panels


def corr(pairs):
    a,b = [np.concatenate(col,axis=0).reshape(-1,2,2).astype(float) for col in zip(*pairs)]
    return ((a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))).tolist()


def main():
    banks=[]
    for d in range(8,16):
        raw=history.trial.clock.book.rawday(d)
        ps=history.references(d,raw)
        y=history.trial.clock.labels(d).reshape(-1,50,3)
        banks.append([profile(p,y) for p in ps])
    result=[]
    for block in (banks[:4],banks[4:]):
        report=[]
        for ref in (0,1):
            metrics={}
            for lag in LAGS:
                metrics[str(lag)]=corr([(v[:-lag],v[lag:]) for session in block for v in session[ref]])
            metrics['morning_afternoon']=corr([(s[ref][0].mean(0,keepdims=True),s[ref][1].mean(0,keepdims=True)) for s in block])
            metrics['other_day_same_clock']=corr([(s[ref][j],u[ref][j]) for i,s in enumerate(block) for u in block[i+1:] for j in (0,1)])
            report.append(metrics)
        result.append(report)
    dest=history.incumbent.OUT/'disjoint_error_blocks.json'
    dest.write_text(json.dumps(dict(days=[[9,12],[13,16]],lags_in_blocks=LAGS,block_ticks=WIDTH,
        origins_per_block=30,order=['offset, slope exposure','Y3, far'],correlations=result),indent=2))
    print(json.dumps([v[0] for v in result]),flush=True)


if __name__=='__main__':
    main()
