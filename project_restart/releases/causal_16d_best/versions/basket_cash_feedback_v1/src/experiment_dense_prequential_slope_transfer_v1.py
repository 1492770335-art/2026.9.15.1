"""Learn which early-session forecast calibration errors persist into later stock-relative returns."""
import numpy as np
import experiment_dense_event_stock_payoffs_v1 as parent

onset,rel,REFERENCE = parent.onset,parent.rel,parent.REFERENCE
OUT = REFERENCE.parent/"dense_prequential_slope_transfer_v1"
ARMS = ("forecast_geometry","settled_slope_transfer","settled_after_geometry")


def frame(p):
    x = p.astype(float)-p.mean(1,keepdims=True,dtype=float)
    sd = np.sqrt(np.square(x).mean(1)).clip(.03)
    xn = (x/sd[:,None]).clip(-5,5)
    g = (xn.transpose(0,2,1)@xn)/50
    geom = np.column_stack((np.log(sd),g[:,0,1],g[:,0,2],g[:,1,2]))
    # A response direction orthogonal to the date's long forecast is less confounded by changes in units.
    x = np.stack((xn[:,:,0]-xn[:,:,2]*g[:,None,0,2],xn[:,:,1]-xn[:,:,2]*g[:,None,1,2],xn[:,:,2]),-1)
    return x,geom


def panels(raw,grid,p):
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    for a,b in rel.fields.parent.sessions(grid):
        n = b-a
        x,geom = frame(p[a:b])
        cash = np.log1p(k*(raw[a:b,:,0].astype(float)-.00136)).cumsum(0)
        gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
        ctx = np.zeros((n,9))
        history = np.zeros((3,3))
        count = np.zeros(3)
        for due in range(5,n,5):
            for j,h in enumerate((5,20,60)):
                origin = due-h
                if origin<0:
                    continue
                valid = gaps[due+1]==gaps[origin]
                norm = valid.sum().clip(1)
                returns = np.expm1(cash[due]-cash[origin])*coef[j,1]/k
                pred = p[a+origin,:,j]
                r = returns-(returns*valid).sum()/norm
                q = pred-(pred*valid).sum()/norm
                error = (r-q).clip(-3,3)*valid
                grad = (error@x[origin])/norm
                decay = np.exp(-5/480)
                history[j] = history[j]*decay+grad
                count[j] = count[j]*decay+1
            ctx[due:min(due+5,n)] = (history/(count[:,None]+12)).ravel()
        # Every feature is known at the forecast origin; fitting targets are attached separately.
        yield a,b,x,geom,ctx


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        v = p[...,2]-p[...,2].mean(1,keepdims=True)
        label = y[...,2]-y[...,2].mean(1,keepdims=True)
        beta = ((v*label).sum()/np.square(v).sum()).clip(.3,3)
        gram = 0.
        for a,b,x,g,h in panels(raw,grid,p):
            ts = np.arange(d%5,b-a,5)
            ctx = np.column_stack((np.ones(len(ts)),g[ts],h[ts]))
            z = (x[ts,:,:,None]*ctx[:,None,None,:]).reshape(len(ts),50,-1)
            z = np.concatenate((z,(label[a+ts]/beta-v[a+ts])[...,None]),axis=-1).reshape(-1,z.shape[-1]+1)
            gram += z.T@z
        np.save(OUT/f"gram_{d:02d}.npy",gram/len(p))


def train(end):
    rows = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    n = len(rows[0])-1
    g = rows.mean(0)
    w = []
    for group in (np.arange(1,7),np.arange(7,16),np.arange(7,16)):
        control = np.arange(7) if len(w)==2 else np.array([0])
        control = (16*np.arange(3)[:,None]+control).ravel()
        ix = (16*np.arange(3)[:,None]+group).ravel()
        basis = np.eye(n)[:,ix]
        c = g[np.ix_(control,control)]
        basis[control] = -np.linalg.solve(c+.035*np.diag(c.diagonal().clip(1e-7)),g[np.ix_(control,ix)])
        cov = basis.T@g[:n,:n]@basis
        sd = np.sqrt(cov.diagonal().clip(1e-6))
        val,u = np.linalg.eigh(cov/np.outer(sd,sd))
        axes = basis@(u/(sd[:,None]*np.sqrt(val.clip(0)+.65)))
        per = rows[:,:n,-1]@axes
        mu = per.mean(0)
        w.append(axes@(mu*mu**2/(mu**2+per.var(0,ddof=1)/len(rows)+1.e-8)))
    np.save(OUT/f"fit_{end}.npy",np.stack(w,axis=-1).reshape(3,16,3))
    return np.array([end])


def design(raw,grid,code,p,bank):
    w = np.load(OUT/f"fit_{int(bank[0])}.npy")
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b,x,g,h in panels(raw,grid,p):
        ctx = np.column_stack((np.ones(b-a),g,h))
        slopes = np.einsum("tc,fca->tfa",ctx,w)
        pred = x@slopes
        z[a:b,:,:,2] = pred-pred.mean(1,keepdims=True)
    return z


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    q[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
