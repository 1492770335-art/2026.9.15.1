"""Learn distal returns from book changes between price-forecast-matched observations of the same stock."""
import json
import numpy as np
from scipy.signal import lfilter
from scipy.spatial.distance import cdist
import experiment_dense_quote_renewal_microprice_v1 as source
from experiment_dense_shared_horizon_calibration_v1 import score

onset, rel, book, REFERENCE = source.onset, source.rel, source.book, source.REFERENCE
OUT = REFERENCE.parent / "dense_matched_distal_panels_v1"
ARMS = ("levels", "matched_times", "matched_time_and_stock_contrasts")
PAIRS = [(41,42),(43,44),(50,51),(52,53),(54,55),(56,57),(79,80)]
SIGNED = [0,11,26,34,36,38,64,74]
C = 3


def ratio():
    with np.load(onset.settled.GEO) as f:
        return float(f["coeff"][2,1]/f["coeff"][1,1])


def centered(x):
    return x - x.mean(1,keepdims=True)


def smooth(x, span):
    a = -np.expm1(-1./span)
    return lfilter([a],[1,a-1],x,axis=0)


def panels(raw, grid, p):
    left, right = map(list, zip(*PAIRS))
    for a,b in rel.fields.parent.sessions(grid):
        data = raw[a:b]
        pressure = np.tanh(.65*np.arcsinh(data[:,:,left].astype(float))) - np.tanh(.65*np.arcsinh(data[:,:,right].astype(float)))
        flow = np.tanh(.65*np.arcsinh(data[:,:,SIGNED].astype(float)))
        state = np.concatenate((pressure,flow),axis=2)
        short, medium, slow = [smooth(state,n) for n in (6,30,110)]
        forecasts = centered(p[a:b].astype(float))
        change = medium-slow
        persistent = np.tanh(3*short*slow)
        x = np.concatenate((forecasts,short,change,slow,slow*persistent,
                            change*np.tanh(2*forecasts[...,1:2]*medium),
                            slow*np.tanh(3*np.abs(short.mean(1,keepdims=True)))),axis=2)
        yield a,b,centered(x),np.concatenate((forecasts,medium[:,:,:3],slow[:,:,-1:]),axis=2)


def load_p(d):
    if d >= 6:
        return np.load(REFERENCE / f"rolling_{d:02d}.npy")
    return onset.settled.head.ranks.band.pairs.net.future.forecast(d)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = load_p(d)
        target = centered(y[...,2].astype(float)-ratio()*y[...,1])
        grams = None
        for a,b,x,keys in panels(raw,grid,p):
            t = np.arange(20+d%7,b-a-60,7)
            if len(t)<25:
                continue
            yx = np.concatenate((x[t],target[a+t,:,None]),axis=2)
            z = yx.reshape(-1,yx.shape[-1])
            if grams is None:
                grams = np.zeros((3,z.shape[1],z.shape[1]))
            grams[0] += z.T@z / 50
            keys = keys[t]
            keys = keys/(keys.std((0,1))+.04)
            for stock in range(50):
                # Compare the same stock at distinct times with similar issued forecasts
                # and liquidity. Both outcomes come from the fitting date.
                distance = cdist(keys[:,stock],keys[:,stock],"sqeuclidean")
                distance[abs(t[:,None]-t)<85] = np.inf
                matched = distance.argmin(1)
                keep = np.isfinite(distance[np.arange(len(t)),matched])
                u,v = np.flatnonzero(keep),matched[keep]
                rows = yx[u,stock]-yx[v,stock]
                grams[1] += rows.T@rows/100
                # Difference again against a separate stock matched on forecasts at both issue times.
                dist = ((keys[u,:,:3]-keys[u,stock:stock+1,:3])**2 +
                        (keys[v,:,:3]-keys[v,stock:stock+1,:3])**2).sum(2)
                dist[:,stock] = np.inf
                other = dist.argmin(1)
                contrast = rows-(yx[u,other]-yx[v,other])
                grams[2] += contrast.T@contrast/200
        np.save(OUT/f"gram_{d:02d}.npy",grams)


def train(end):
    gs = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    result = []
    for rows in gs.transpose(1,0,2,3):
        gram = rows.mean(0)
        w = len(gram)-1
        sd = np.sqrt(gram.diagonal()[:w]).clip(.001)
        cov = gram[:w,:w] / np.outer(sd,sd)
        rhs = rows[:,:w,-1]/sd
        eig,rot = np.linalg.eigh(cov)
        cross = rhs@rot
        mean = cross.mean(0)
        noise = cross.var(0)/len(rows)
        coeff = mean*mean**2/(mean**2 + noise + 1e-10)/(eig.clip(0)+.40)
        result.append((rot@coeff)/sd)
    return np.stack(result,axis=1)


def signals(raw,grid,p,bank):
    out = np.zeros((*p.shape[:2],bank.shape[1]))
    for a,b,x,_ in panels(raw,grid,p):
        out[a:b] = x@bank
    return out


def calibration(g):
    pooled = np.sum(g,axis=0)
    cov = pooled[1:,1:]-np.outer(pooled[0,1:],pooled[0,1:])/pooled[0,0]
    return np.array([cov[0,2]/cov[0,0],cov[1,3]/cov[1,1]])


def phase_stats(p,y):
    z = np.column_stack((np.ones(p.shape[0]*p.shape[1]),p[...,2].ravel(),p[...,1].ravel(),
                         y[...,2].ravel(),y[...,1].ravel())).astype(float)
    return z.T@z


def predict(raw, grid, p, params):
    z = signals(raw,grid,p,params["bank"])
    q = p.copy()
    q[...,2] += (rel.relative.features(p,z) @ params["w"]).astype(q.dtype)
    return q


def rolling(model):
    out, arms = model.OUT, model.ARMS
    stats,cal = [],[]
    results = {arm:[[],[]] for arm in arms}
    # Only Y3 changes. Y1's market-only correction cancels in centered forecast inputs.
    for d in range(6,16):
        if d in (6,8,12):
            bank = model.train(d)
        p = np.load(model.REFERENCE / f"rolling_{d:02d}.npy")
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        z = model.signals(raw,grid,p,bank)
        if d==8:
            np.testing.assert_allclose(z[:512],model.signals(raw[:512],grid[:512],p[:512],bank),atol=1e-8,rtol=0)
        if d in (8,12):
            weights = [rel.relative.fit(stats,j,2) for j in range(len(arms))]
            beta3,beta2 = calibration(cal)
        if d>=8:
            x = rel.relative.features(p,z)
            prior = centered(beta3*p[...,2].astype(float)-ratio()*beta2*p[...,1])
            actual = centered(y[...,2].astype(float)-ratio()*y[...,1])
            for j,arm in enumerate(arms):
                shift = x@weights[j]
                results[arm][0].append(rel.ref.market.metric(p[...,2],y[...,2],shift))
                results[arm][1].append(rel.ref.market.metric(prior,actual,centered(shift)*beta3))
        stats.append(rel.relative.stats(p,y,z))
        cal.append(phase_stats(p,y))
    report = {k:dict(full=score(full),distal=score(distal)) for k,(full,distal) in results.items()}
    passed = [arm for arm,r in report.items() if min(r["full"]["folds"]+r["full"]["omit_one"]+r["distal"]["folds"])>0
              and sum(x>0 for x in r["full"]["days"])>=5]
    pick = max(passed,key=lambda k:min(report[k]["full"]["folds"])) if passed else None
    (out/"rolling.json").write_text(json.dumps(dict(selected={"2":pick} if pick else {},results=report),indent=2))
    if pick:
        np.savez(out/"weights.npz",bank=model.train(16),w=rel.relative.fit(stats,arms.index(pick),2))
    print(pick,{k:[v["full"]["folds"],v["distal"]["folds"]] for k,v in report.items()},flush=True)


if __name__=="__main__":
    import sys
    collect()
    rolling(sys.modules[__name__])
