"""Infer stock corrections from matched-pair spreads with similar original near-term forecasts."""
import json

import numpy as np

from experiment_dense_robust_distal_readout_v1 import smooth
from experiment_dense_decoded_counts_v1 import base, TAPE, codec, decode, released, sessions, metric, gain, summary

OUT = base.ROOT/"artifacts/dense_near_matched_spreads_v1"
ARMS = ("whole", "distal", "terminal")
NEIGHBORS = 4


def sources(raw, grid, code):
    chunks = []
    for a,b in sessions(grid):
        s = raw[a:b]
        price = np.tanh(s[:,:,0].astype(float)-.00136)
        book = s[:,:,[26,38,41,42,54,55]].astype(float)
        imb = np.tanh(.65*(book[:,:,::2]-book[:,:,1::2]))
        counts = np.rint(decode(s[:,:,77:81],code[:4])).clip(0,512)
        left,right,u,v = np.moveaxis(counts,2,0)
        def direction(l,r):return np.tanh((l-r)/np.sqrt(1+l+r+.025*(l+r)**2))
        levels = np.stack((direction(u,v),direction((left-u).clip(0),(right-v).clip(0)),price),2)
        src = np.concatenate((imb,levels),2)
        fast = smooth(src,8)
        slow = smooth(src,50)
        chunks.append(np.concatenate((src-fast,fast-slow,slow),2).astype("float32"))
    return np.concatenate(chunks)


def pair_readout(p,src,wrong=False):
    # Near-term coordinates define the graph. The two original near forecasts
    # stay attached to their stocks when constructing spread and outcome features.
    curve = p*base.UNITS
    curve = np.diff(curve,prepend=0.,axis=2)
    curve -= curve.mean(1,keepdims=True)
    curve /= (.04+curve.std(1,keepdims=True))
    keys = np.tanh(curve[:,:,:2]).astype("float32")
    if wrong:keys = np.roll(keys,17,axis=1)
    dist = ((keys[:,:,None]-keys[:,None])**2).sum(3)
    stocks = np.arange(50)
    dist[:,stocks,stocks] = np.inf
    idx = np.argpartition(dist,NEIGHBORS-1,axis=2)[:,:,:NEIGHBORS]
    t = np.arange(len(p))[:,None,None]
    receiver = stocks[None,:,None]
    s = src[:,:,None]-src[t,idx]
    # Conditional pair interactions are antisymmetric when source and recipient swap.
    far = np.tanh(curve[:,:,2:3])
    spread = (far[:,:,None]-far[t,idx])
    local = np.tanh(src[:,:,None]+src[t,idx])
    z = np.concatenate((s,s*np.abs(spread),spread*np.abs(s),s*local*far[:,:,None]*far[t,idx]),3)
    z = z.astype("float32")
    strength = 1/(.12+np.sqrt(dist[t,receiver,idx])).clip(.12,10)
    strength /= strength.sum(2,keepdims=True)
    # Symmetrize the directed k-neighbor graph and its signed edge messages.
    graph = np.zeros((len(p),50,50),np.float32)
    graph[t,receiver,idx] = strength
    graph = graph+graph.transpose(0,2,1)
    divergence = (strength[:,:,:,None]*z).sum(2)
    for k in range(NEIGHBORS):
        np.add.at(divergence,(t[...,0],idx[:,:,k]),-strength[:,:,k,None]*z[:,:,k])
    denom = graph.sum(2,keepdims=True)+1.5
    value = divergence/denom
    # Anchoring at zero limits how much a small or weakly connected cluster can move.
    for _ in range(4):
        value = (divergence+graph@value)/denom
    return value-value.mean(1,keepdims=True)


def build(p,src,wrong):
    return np.concatenate([pair_readout(p[i:i+32],src[i:i+32],wrong) for i in range(0,len(p),32)])


def stat(x,p,y,t,grid,coef):
    n = len(t)
    # Only the training targets use an already-aligned future 20-tick official label.
    # y(t+40, 20) is on a slightly different price denominator; it is a separate auxiliary target.
    q = p[t].astype(float)
    q -= q.mean(1,keepdims=True)
    truth = np.stack((y[t,:,2],y[t,:,2]-base.UNITS[1]*y[t,:,1],
                      base.UNITS[1]*(y[t+40,:,1]-coef[1,0])),2).astype(float)
    truth -= truth.mean(1,keepdims=True)
    z = np.concatenate((q,x,truth),2).reshape(n*50,-1)
    return z.T@z/len(z)


def fit(hist,arm):
    mats = np.stack(hist)
    g = mats.mean(0)
    n = len(g)-3
    proj = np.eye(n)[:,3:]
    def reg(q,c):return q+c*np.diag(q.diagonal().clip(1e-7))
    proj[:3] = -np.linalg.solve(reg(g[:3,:3],.015),g[:3,3:n])
    h = proj.T@g[:n,:n]@proj
    slope = (mats[:,2,n]/mats[:,2,2]).clip(.3,3.)
    idx = ARMS.index(arm)
    # For partial payoffs, first remove what the three existing forecasts already explain.
    if idx:
        target = mats[:,:n,n+idx]-mats[:,:n,:3]@np.linalg.solve(reg(g[:3,:3],.02),g[:3,n+idx])
        raw = target/slope[:,None]
    else:
        raw = mats[:,:n,n]/slope[:,None]-mats[:,:n,2]
    grad = raw@proj
    rhs = grad.mean(0)
    rhs *= abs(rhs)/(abs(rhs)+grad.std(0)/np.sqrt(len(hist))+1e-6)
    w = proj@np.linalg.solve(reg(h,.95),rhs)
    risk = np.sqrt(max(w@g[:n,:n]@w,1e-12)/g[2,2])
    return w*min(1,.055/risk)


def apply(p,x,w):
    delta = p@w[:3]+x@w[3:]
    return delta-delta.mean(1,keepdims=True)


def main():
    from experiment_dense_session_slope_v1 import GEO
    with np.load(GEO) as g:co = g["coeff"]
    OUT.mkdir(exist_ok=True)
    code = codec()
    hist = [[],[]]
    panels = {(arm,shift):[] for arm in ARMS for shift in (0,1)}
    for d in range(4,16):
        load = lambda key:np.load(base.CACHE/f"day_{d:02d}_{key}.npy",mmap_mode="r")
        raw,grid,y = load("x"),load("time").astype(int),load("y")
        p = np.load(TAPE/f"forecast_{d:02d}.npy")
        src = sources(raw,grid,code)
        train = np.concatenate([np.arange(a+(d%4),b-60,4) for a,b in sessions(grid)])
        if d in (8,12):
            w = {(k,s):fit(hist[s],k) for k,s in panels}
        for s in (0,1):
            if d>=8:
                x = build(p,src,s)
                for arm in ARMS:
                    panels[arm,s].append(metric(p[...,2],y[...,2],apply(p,x,w[arm,s])))
                if d==8:
                    n = 219
                    xp = build(p[:n],sources(raw[:n],grid[:n],code),s)
                    np.testing.assert_allclose(x[:n],xp,rtol=0,atol=2e-6)
                selected = x[train]
            else:
                selected = build(p[train],src[train],s)
            hist[s].append(stat(selected,p,y,train,grid,co))
        print("completed training date",d+1,flush=True)
    report = {}
    for (arm,s),rows in panels.items():
        report[f"{arm}_{s}"] = dict(folds=[gain(sum(z)) for z in (rows[:4],rows[4:])],days=[gain(z) for z in rows])
    passing = []
    for arm in ARMS:
        true,control = report[f"{arm}_0"],report[f"{arm}_1"]
        if min(true["folds"])>0 and sum(i>0 for i in true["days"])>=5 and all(a>b for a,b in zip(true["folds"],control["folds"])):
            passing.append(arm)
    pick = max(passing,key=lambda arm:min(report[f"{arm}_0"]["folds"])) if passing else None
    (OUT/"rolling.json").write_text(json.dumps(dict(results=report,selected=pick,control="rotate stock identities only when matching near curves"),indent=2))
    print(pick,{k:v["folds"] for k,v in report.items()},flush=True)
    if not pick:return
    np.savez(OUT/"weights.npz",code=code,w=fit(hist[0],pick))
    outer = []
    for d in range(16,20):
        load = lambda key:np.load(base.CACHE/f"day_{d:02d}_{key}.npy",mmap_mode="r")
        raw,grid = load("x"),load("time").astype(int)
        p = released(d,raw)
        with np.load(OUT/"weights.npz") as f:
            src = sources(raw,grid,f["code"])
            q = p.copy();q[:,:,2] += apply(p,build(p,src,False),f["w"])
        outer.append((p,load("y"),q))
    result = summary(outer)
    assert result["rows"]==936250 and abs(result["baseline_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=result,daily=[summary([r]) for r in outer]),indent=2))
    print(result,flush=True)


if __name__=="__main__":main()
