"""Read the original fixed forecasts with heteroskedastic training losses for individual stocks' Y3."""
import json

import numpy as np
from scipy.signal import lfilter

from experiment_dense_decoded_counts_v1 import base,TAPE,codec,decode,released,sessions,metric,gain,summary
from experiment_dense_session_slope_v1 import GEO

OUT = base.ROOT/"artifacts/dense_robust_distal_readout_v1"
ARMS = ("ordinary","past_risk","pseudo_huber","loss_difference")


def smooth(x,tau):
    alpha = 1-np.exp(-1/tau)
    return lfilter([alpha],[1,alpha-1],x-x[:1],axis=0)+x[:1]


def views(raw,grid,p,code):
    with np.load(GEO) as f:k,unit = float(f["k"]),float(f["coeff"][2,1])
    parts = []
    for a,b in sessions(grid):
        s = raw[a:b]
        cash = np.log1p(k*(s[:,:,0].astype(float)-.00136))*unit/k
        move = cash-cash.mean(1,keepdims=True)
        # Estimate target-scale dispersion from observable one-step moves at each stock.
        risk = np.sqrt(.10+48*(.7*smooth(move**2,128)+.3*smooth(move**2,18)))
        risk = np.clip(risk,.35,3.5)
        f = p[a:b].astype(float)
        x = (f-f.mean(1,keepdims=True))*base.UNITS
        band = np.diff(x,prepend=0.,axis=2)
        std = np.sqrt(np.square(band).mean(1,keepdims=True))+.045
        rank = np.tanh(band/std)
        book = s[:,:,[26,38,41,42,54,55]].astype(float)
        book = np.tanh(.6*(book[:,:,::2]-book[:,:,1::2]))
        c = np.rint(decode(s[:,:,77:81],code[:4])).clip(0,512)
        q = np.stack((c[:,:,0]-c[:,:,2],c[:,:,1]-c[:,:,3]),2).clip(0)
        rem = np.tanh((q[:,:,0]-q[:,:,1])/np.sqrt(1+q.sum(2)))
        press = np.concatenate((book,rem[:,:,None]),2)
        v = np.concatenate((x,rank,rank*np.tanh(np.log(risk))[:,:,None],rank*np.roll(rank,1,axis=2)**2,
                            press,smooth(press,16),press*rank[:,:,2:3]**2),2)
        parts.append((v-v.mean(1,keepdims=True),risk))
    return tuple(np.concatenate(cols) for cols in zip(*parts))


def stats(x,r,p,y):
    p = p[...,2].astype(float);y = y[...,2].astype(float)
    p -= p.mean(1,keepdims=True);y -= y.mean(1,keepdims=True)
    beta = (np.sum(p*y)/np.sum(p*p)).clip(.35,3.)
    residual = y/beta-p
    med = np.median(np.abs(residual))+.02
    rel = (r/np.median(r)).clip(.4,2.8)
    # All policies see the same stocks/targets. Only gradient weighting differs.
    weights = [np.ones_like(r),1/np.sqrt(.3+rel**2),1/np.sqrt(1+(residual/(2.8*med*rel))**2)]
    out = []
    for weight in weights:
        weight = (weight/weight.mean()).ravel()
        xx = x.reshape(-1,x.shape[-1])
        v = (xx.T*weight)@xx/len(xx)
        grad = xx.T@(weight*residual.ravel())/len(xx)
        out.append((v,grad))
    return out,float(np.square(p).mean())


def train(hist):
    def reg(g,c):return g+c*np.diag(g.diagonal().clip(1e-7))
    g0 = np.mean([s[0][0][0] for s in hist],0)
    # Protect the original short, medium and long stock portfolios.
    proj = np.eye(len(g0))[:,3:].copy()
    proj[:3] = -np.linalg.solve(reg(g0[:3,:3],.01),g0[:3,3:])
    result = {}
    for i,key in enumerate(ARMS[:3]):
        h = proj.T@np.mean([s[0][i][0] for s in hist],0)@proj
        grad = np.stack([s[0][i][1]@proj for s in hist])
        avg = grad.mean(0)
        avg *= abs(avg)/(abs(avg)+grad.std(0)/np.sqrt(len(grad))+1e-6)
        result[key] = proj@np.linalg.solve(reg(h,1),avg)
    result[ARMS[3]] = result["pseudo_huber"]-result["ordinary"]
    for key,w in result.items():
        risk = np.sqrt(max(w@g0@w,1e-12)/np.mean([s[1] for s in hist]))
        result[key] = w*min(1,.045/risk)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    code, hist = codec(), []
    rows = {k:[] for k in ARMS}
    for d in range(4,16):
        raw = np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid = np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p = np.load(TAPE/f"forecast_{d:02d}.npy")
        y = np.load(base.CACHE/f"day_{d:02d}_y.npy")
        x,r = views(raw,grid,p,code)
        if d in (8,12):w = train(hist)
        if d>=8:
            for key in ARMS:
                rows[key].append(metric(p[...,2],y[...,2],x@w[key]))
            if d==8:
                short = views(raw[:120],grid[:120],p[:120],code)
                np.testing.assert_array_equal(short[0],x[:120])
        hist.append(stats(x[::3],r[::3],p[::3],y[::3]))
    report = {k:dict(folds=[gain(sum(v[:4])),gain(sum(v[4:]))],days=[gain(t) for t in v]) for k,v in rows.items()}
    ok = [k for k,v in report.items() if min(v["folds"])>0 and sum(z>0 for z in v["days"])>=5]
    pick = max(ok,key=lambda k:min(report[k]["folds"])) if ok else None
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=pick,results=report),indent=2))
    print({k:v["folds"] for k,v in report.items()},pick,flush=True)
    if not pick:return
    np.savez(OUT/"weights.npz",code=code,w=train(hist)[pick])
    outer = []
    for d in range(16,20):
        raw = np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        grid = np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
        p = released(d,raw)
        q = p.copy()
        with np.load(OUT/"weights.npz") as f:
            x,_ = views(raw,grid,p,f["code"])
            q[:,:,2] += x@f["w"]
        outer.append((p,np.load(base.CACHE/f"day_{d:02d}_y.npy"),q))
    result = summary(outer)
    assert result["rows"]==936250 and abs(result["baseline_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=result,daily=[summary([r]) for r in outer]),indent=2))
    print(result,flush=True)


if __name__=="__main__":main()
