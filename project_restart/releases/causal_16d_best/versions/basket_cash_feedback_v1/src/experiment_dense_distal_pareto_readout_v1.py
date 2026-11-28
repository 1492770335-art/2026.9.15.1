"""Optimize the official full-horizon correlation and the distal correlation jointly on preceding dates."""
import json
import numpy as np
from scipy.optimize import minimize
import experiment_dense_matched_distal_panels_v1 as panel

onset, rel, REFERENCE = panel.onset, panel.rel, panel.REFERENCE
OUT = REFERENCE.parent / "dense_distal_pareto_readout_v1"
ARMS = ("joint_correlation", "date_balanced_joint", "near_payoff_neutral")
# The final five columns hold full P, medium P, full Y, medium Y, and the intercept.
EXTRA = 5


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = panel.load_p(d)
        gr = None
        for a,b,x,_ in panel.panels(raw,grid,p):
            t = np.arange(d%3,b-a,3)
            obs = np.concatenate((x[t], p[a+t,:,2:3],p[a+t,:,1:2],
                                  y[a+t,:,2:3],y[a+t,:,1:2],np.ones((*x[t].shape[:2],1))),axis=2)
            means = obs.mean(1,keepdims=True)
            def gram(z):
                z = z.reshape(-1,z.shape[-1])
                return z.T@z
            part = np.stack((gram(obs),gram(obs-means)))
            gr = part if gr is None else gr+part
        np.save(OUT/f"gram_{d:02d}.npy",gr/gr[0,-1,-1])


def covariances(end, directory):
    g = np.stack([np.load(directory/f"gram_{d:02d}.npy") for d in range(4,end)])
    full = g[:,0]
    full -= full[:,:,-1,None]*full[:,None,-1,:]
    return g[:,:,:-1,:-1]


def fit_bank(g):
    width = g.shape[-1]-4
    pooled = g[:,0].mean(0)
    # Whiten visible information after removing exposure to the incumbent outputs.
    tr = np.eye(width)[:,3:]
    m = pooled[:3,:3]
    tr[:3] = -np.linalg.solve(m+.005*np.diag(m.diagonal()),pooled[:3,3:width])
    cov = tr.T@pooled[:width,:width]@tr
    sd = np.sqrt(cov.diagonal().clip(1e-8))
    ev,u = np.linalg.eigh(cov/np.outer(sd,sd))
    tr = tr@(u/(sd[:,None]*np.sqrt(ev.clip(0)+.55)))
    eyes = np.eye(width+4)
    b = pooled[-4,-2]/pooled[-4,-4]
    e = pooled[-3,-1]/pooled[-3,-3]
    predictions = (eyes[-4], b*eyes[-4]-panel.ratio()*e*eyes[-3])
    outcomes = (eyes[-2], eyes[-2]-panel.ratio()*eyes[-1])
    # Linearize Pearson correlation at the incumbent in units of added forecast RMS.
    scale = np.sqrt(pooled[-4,-4])
    gradients = []
    for kind,(p,y,amp) in enumerate(zip(predictions,outcomes,(scale,b*scale))):
        mat = g[:,kind]
        vp = p@mat@p
        vy = y@mat@y
        slope = ((p@mat@y)/vp)
        resid = y-p*slope[:,None]
        cross = (mat[:,:width]@resid[:,:,None])[:,:,0]@tr
        gradients.append(cross*(amp/np.sqrt(vp*vy))[:,None])
    a,c = gradients
    # Give the two tasks equal leverage at initialization without changing the sign of either gradient.
    units = np.maximum([np.linalg.norm(a.mean(0)),np.linalg.norm(c.mean(0))],1e-8)
    a,c = a/units[0],c/units[1]
    near = (g[:,1,:width,-1]@tr)
    near /= np.linalg.norm(near,axis=1,keepdims=True).clip(1e-7)
    directions = []
    for name in ARMS:
        aa,cc = a,c
        if name=="near_payoff_neutral":
            q = np.linalg.svd(near,full_matrices=False)[2]
            aa=a-(a@q.T)@q
            cc=c-(c@q.T)@q
        vectors = np.stack((aa.mean(0),cc.mean(0)))
        if name=="date_balanced_joint":
            # Include each training half as an additional constraint. A positive score
            # on one fitting period alone cannot dominate this update.
            vectors=np.stack([rows[k::2].mean(0) for rows in (aa,cc) for k in (0,1)])
        # Closest convex combination of gradients to zero gives their common
        # ascent direction. If they conflict perfectly the update vanishes.
        gram=vectors@vectors.T
        n=len(gram)
        result=minimize(lambda w:.5*w@gram@w,np.ones(n)/n,jac=lambda w:gram@w,
                        bounds=[(0,1)]*n,constraints=[dict(type="eq",fun=lambda w:w.sum()-1,
                        jac=lambda w:np.ones(n))],method="SLSQP",options={"ftol":1e-13})
        if not result.success:
            raise RuntimeError(result.message)
        direction=(result.x@vectors)
        dw=tr@direction
        rms=np.sqrt(max(0.,dw@pooled[:width,:width]@dw))
        if rms>1e-10:
            dw/=rms
        else:
            dw[:]=0
        # Estimate curvature using the exact pooled correlations; size is chosen
        # only on the history and shrunk to account for estimation noise.
        candidates=np.array([0.,.01,.025,.05,.085,.14])
        improvements=[]
        for risk in candidates:
            correction=np.r_[risk*scale*dw,np.zeros(4)]
            deltas=[]
            for kind,(pr,yy,amp) in enumerate(zip(predictions,outcomes,(1.,b))):
                mat=g[:,kind].mean(0)
                def corr(v):
                    return (v@mat@yy)/np.sqrt((v@mat@v)*(yy@mat@yy))
                deltas.append((corr(pr+amp*correction)-corr(pr))/units[kind])
            improvements.append(min(deltas))
        directions.append(.5*scale*candidates[np.argmax(improvements)]*dw)
    return np.stack(directions,axis=1)


def train(end):
    return fit_bank(covariances(end,OUT))


def signals(raw,grid,p,bank):
    return panel.signals(raw,grid,p,bank)


def predict(raw,grid,p,params):
    q=p.copy()
    q[...,2]+=signals(raw,grid,p,params["bank"])[:,:,int(params["selected"])].astype(q.dtype)
    return q


def rolling(module):
    out,arms=module.OUT,module.ARMS
    results={arm:[[],[],[]] for arm in arms}
    cal=[panel.phase_stats(panel.load_p(d),onset.settled.load(d,"y")) for d in range(4,8)]
    for d in range(8,16):
        if d in (8,12):
            bank=module.train(d)
            beta3,beta2=panel.calibration(cal)
        p=panel.load_p(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        z=module.signals(raw,grid,p,bank)
        if d in (8,12):
            for n in (223,2501):
                short=module.signals(raw[:n],grid[:n],p[:n],bank)
                np.testing.assert_allclose(z[:n],short,atol=2e-7,rtol=0)
        near=panel.centered(beta2*p[...,1].astype(float)*panel.ratio())
        actual_near=panel.centered(y[...,1].astype(float)*panel.ratio())
        prior=panel.centered(beta3*p[...,2].astype(float))-near
        actual=panel.centered(y[...,2].astype(float))-actual_near
        for j,k in enumerate(arms):
            shift=z[...,j]
            for store,(pred,target,change) in zip(results[k],((p[...,2],y[...,2],shift),
                     (prior,actual,beta3*shift),(near,actual_near,beta3*shift))):
                store.append(rel.ref.market.metric(pred,target,change))
        cal.append(panel.phase_stats(p,y))
    report={k:{n:panel.score(v) for n,v in zip(("full","distal","near_if_applied"),rows)} for k,rows in results.items()}
    passed=[arm for arm,r in report.items() if min(r["full"]["folds"]+r["full"]["omit_one"]+r["distal"]["folds"])>0
              and sum(x>0 for x in r["full"]["days"])>=5]
    pick=max(passed,key=lambda k:min(report[k]["full"]["folds"])) if passed else None
    (out/"rolling.json").write_text(json.dumps(dict(selected={"2":pick} if pick else {},results=report),indent=2))
    if pick:
        np.savez(out/"weights.npz",bank=module.train(16),selected=np.array(arms.index(pick)))
    print(pick,{k:[v["full"]["folds"],v["distal"]["folds"]] for k,v in report.items()},flush=True)


if __name__=="__main__":
    import sys
    collect()
    rolling(sys.modules[__name__])
