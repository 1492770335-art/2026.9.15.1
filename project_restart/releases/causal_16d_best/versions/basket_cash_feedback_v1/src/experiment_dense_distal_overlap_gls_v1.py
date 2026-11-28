"""Whiten overlapping distant-return labels and the contemporaneous regressors in both time and stock dimensions."""
import numpy as np
import experiment_dense_distal_scenario_hedge_v1 as hedge

panel,onset,rel,REFERENCE=hedge.panel,hedge.onset,hedge.rel,hedge.REFERENCE
OUT=REFERENCE.parent/"dense_distal_overlap_gls_v1"
ARMS=("scalar_time_whitening","stock_time_whitening","innovation_covariance_whitening")
LAGS=(4,12,20)
W=hedge.WIDTH


def label_moments():
    for d in range(16):
        y,grid=[onset.settled.load(d,k) for k in ("y","time")]
        fy=panel.centered(y[...,2].astype(float)-panel.ratio()*y[...,1])
        gram=np.zeros((200,200))
        for a,b in rel.fields.parent.sessions(grid):
            t=np.arange(a+20+d%4,b-60,4)
            z=np.concatenate([fy[t-k] for k in (0,*LAGS)],axis=1)
            gram+=z.T@z
        np.save(OUT/f"lag_{d:02d}.npy",gram)


def noise_model(end):
    g=sum(np.load(OUT/f"lag_{d:02d}.npy") for d in range(end))
    g/=np.trace(g)/200
    lag=list(LAGS)
    c=np.array([[1-abs(i-j)/40 for j in lag] for i in lag])
    w=np.linalg.solve(c+.05*np.eye(3),1-np.array(lag)/40)
    scalar=np.concatenate([z*np.eye(50) for z in w])
    cov=g[50:,50:]
    # Empirical lag covariances can represent cross-stock lead/lag; regularize to independent overlapping windows.
    matrix=scalar+np.linalg.solve(cov+.45*np.eye(len(cov)),g[50:,:50]-cov@scalar)
    q=np.vstack((np.eye(50),-matrix))
    err=q.T@g@q
    ev,rot=np.linalg.eigh(err)
    root=(rot/np.sqrt(ev.clip(0)+.5*np.trace(err)/50))@rot.T
    return scalar.reshape(3,50,50),matrix.reshape(3,50,50),root


def inputs(raw,grid,p):
    for a,b,x,_ in panel.panels(raw,grid,p):
        for i in range(0,len(x),128):
            sl=slice(i,i+128)
            x[sl]=hedge.neutral(x[sl],x[sl,:,:3].copy())
        yield a,b,x


def collect():
    OUT.mkdir(exist_ok=True)
    label_moments()
    for d in range(4,16):
        scalar,matrix,root=noise_model(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=panel.load_p(d)
        far=panel.centered(y[...,2].astype(float)-panel.ratio()*y[...,1])
        grams=np.zeros((3,W+1,W+1))
        for a,b,x in inputs(raw,grid,p):
            t=np.arange(20+d%4,b-a-60,4)
            if not len(t):
                continue
            z=np.concatenate((x,far[a:b,:,None]),axis=2)
            for u in np.array_split(t,(len(t)+63)//64):
                present=z[u]
                for j,params in enumerate((scalar,matrix)):
                    filtered=present-sum(m.T@z[u-k] for m,k in zip(params,LAGS))
                    zz=filtered.reshape(-1,W+1)
                    grams[j]+=zz.T@zz
                    if j==1:
                        zz=(root@filtered).reshape(-1,W+1)
                        grams[2]+=zz.T@zz
        np.save(OUT/f"gram_{d:02d}.npy",grams/len(p))


def train(end):
    grams=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    bank=[]
    for samples in grams.transpose(1,0,2,3):
        g=samples.mean(0)
        sd=np.sqrt(g.diagonal()[:W].clip(1e-6))
        ev,u=np.linalg.eigh(g[:W,:W]/np.outer(sd,sd))
        axes=u/(sd[:,None]*np.sqrt(ev.clip(0)+1.))
        rhs=samples[:,:W,-1]@axes
        avg=rhs.mean(0)
        coef=avg*avg**2/(avg**2+rhs.var(0)/len(rhs)+1e-8)
        bank.append(axes@coef)
    return np.stack(bank,axis=1)


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],len(ARMS)))
    for a,b,x in inputs(raw,grid,p):
        out[a:b]=x@bank
    return out


def predict(raw,grid,p,params):
    out=p.copy()
    z=signals(raw,grid,p,params["bank"])
    out[...,2]+=(rel.relative.features(p,z)@params["w"]).astype(out.dtype)
    return out


if __name__=="__main__":
    import sys
    collect()
    panel.rolling(sys.modules[__name__])
