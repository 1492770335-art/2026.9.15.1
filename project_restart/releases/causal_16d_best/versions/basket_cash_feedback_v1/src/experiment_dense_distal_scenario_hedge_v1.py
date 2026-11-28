"""Neutralize a distal portfolio against several independently estimated near-return scenarios at each issuance."""
import numpy as np
import experiment_dense_distal_pareto_readout_v1 as pareto

panel,onset,rel,REFERENCE = pareto.panel,pareto.onset,pareto.rel,pareto.REFERENCE
OUT = REFERENCE.parent / "dense_distal_scenario_hedge_v1"
ARMS = ("issuer_forecasts", "near_return_scenarios", "near_scenarios_and_volatility")
WIDTH = 93
# These channels are independent of the incumbent even on days without a rolling forecast.
KEEP = np.r_[3:63,78:93]
GROUPS = 4
HEDGES = 2*GROUPS


def solve(g, rhs, penalty):
    sd = np.sqrt(g.diagonal().clip(1e-8))
    return np.linalg.solve(g/np.outer(sd,sd)+penalty*np.eye(len(g)), rhs/sd[:,None])/sd[:,None]


def build_near():
    for d in range(16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        # All scenario inputs are price/book data, so earlier dates can be used.
        dummy=np.zeros((*raw.shape[:2],3),np.float32)
        side=len(KEEP)+2
        gram=np.zeros((side,side))
        for a,b,x,_ in panel.panels(raw,grid,dummy):
            t=np.arange(d%5,b-a-20,5)
            z=np.concatenate((x[t][:,:,KEEP],panel.centered(y[a+t,:,:2].astype(float))),axis=2)
            rows=z.reshape(-1,side)
            gram+=rows.T@rows
        np.save(OUT/f"near_{d:02d}.npy",gram)


def scenarios(end):
    rows=[np.load(OUT/f"near_{i:02d}.npy") for i in range(end)]
    bank=np.zeros((WIDTH,HEDGES))
    for k in range(GROUPS):
        m=sum(rows[k::GROUPS])
        bank[KEEP,2*k:2*k+2]=solve(m[:-2,:-2],m[:-2,-2:],.6)
    return bank


def neutral(x, exposures):
    # Projection uses only the 50 stocks at the current time.
    ex=panel.centered(exposures)
    ex=ex/np.sqrt((ex*ex).mean(1,keepdims=True)+1e-8)
    m=ex.swapaxes(1,2)@ex
    rhs=ex.swapaxes(1,2)@x
    return x-ex@np.linalg.solve(m+.25*np.eye(m.shape[-1]),rhs)


def bases(raw,grid,p,hedges,stride,offset):
    for a,b,x,_ in panel.panels(raw,grid,p):
        vol=np.sqrt(.002+panel.smooth(np.tanh(raw[a:b,:,0].astype(float))**2,45))
        times=np.arange(offset,b-a,stride)
        for start in range(0,len(times),128):
            t=times[start:start+128]
            u=x[t]
            near=u@hedges
            levels=u[:,:,:3]
            # The dispersion across date-specific fits describes uncertainty in the immediate price response.
            all_near=np.concatenate((levels,near),axis=2)
            risk=vol[t]-vol[t].mean(1,keepdims=True)
            stressed=near.reshape(len(t),50,GROUPS,2).mean(2)*risk[:,:,None]
            yield a+t, (neutral(u,levels),neutral(u,all_near),neutral(u,np.concatenate((all_near,stressed),axis=2)))


def collect():
    OUT.mkdir(exist_ok=True)
    build_near()
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=panel.load_p(d)
        far=panel.centered(y[...,2].astype(float)-panel.ratio()*y[...,1])
        full=panel.centered(y[...,2].astype(float))
        gram=np.zeros((len(ARMS),WIDTH+2,WIDTH+2))
        for t,views in bases(raw,grid,p,scenarios(d),4,d%4):
            for j,u in enumerate(views):
                data=np.concatenate((u,far[t,:,None],full[t,:,None]),axis=2).reshape(-1,WIDTH+2)
                gram[j]+=data.T@data
        np.save(OUT/f"gram_{d:02d}.npy",gram/len(p))


def train(end):
    rows=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    weights=[]
    for gs in rows.transpose(1,0,2,3):
        g=gs.mean(0)
        # Learn the part of the far-return covariance with independently replicating full-return payoff.
        sd=np.sqrt(g.diagonal()[:WIDTH].clip(1e-9))
        ev,rot=np.linalg.eigh(g[:WIDTH,:WIDTH]/np.outer(sd,sd))
        axis=rot/(sd[:,None]*np.sqrt(ev.clip(0)+.8))
        terms=[]
        for i in (0,1):
            far=axis.T@gs[i::2,:WIDTH,-2].mean(0)
            full=axis.T@gs[1-i::2,:WIDTH,-1].mean(0)
            terms.append(np.where(far*full>0,np.sign(far)*np.minimum(abs(far),abs(full)),0))
        weights.append(axis@np.mean(terms,0))
    return np.column_stack((scenarios(end),*weights))


def signals(raw,grid,p,bank):
    out=np.empty((*p.shape[:2],len(ARMS)))
    for t,views in bases(raw,grid,p,bank[:,:HEDGES],1,0):
        for j,x in enumerate(views):
            out[t,:,j]=x@bank[:,HEDGES+j]
    return out


def predict(raw,grid,p,params):
    z=signals(raw,grid,p,params["bank"])
    q=p.copy()
    q[...,2]+=(rel.relative.features(p,z)@params["w"]).astype(q.dtype)
    return q


if __name__=="__main__":
    import sys
    collect()
    panel.rolling(sys.modules[__name__])
