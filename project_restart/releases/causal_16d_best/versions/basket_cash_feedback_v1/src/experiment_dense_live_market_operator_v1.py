"""Update the market-state transition using only the book states that have already arrived."""
import json
import numpy as np
import prepare_dense_market_operator_moments_v1 as data
import experiment_dense_transfer_precision_v1 as precision

market=data.market
tape=data.tape
base=data.base
OUT=base.ROOT/"artifacts/dense_live_market_operator_v1"
ARMS=("forward_update","symmetric_update","independent_blocks")


def model(st,mk):
    g=np.mean(mk,0);s=np.mean(st,0);n=(len(g)-1)//3
    sd=np.sqrt(g.diagonal()[:n]).clip(.006);ss=np.sqrt(s.diagonal()[:n]).clip(.03)
    ratio=sd/ss
    h=.5*(g[:n,:n]+s[:n,:n]*np.outer(ratio,ratio))/np.outer(sd,sd)
    val,u=np.linalg.eigh(h+.25*np.eye(n));root=(u/np.sqrt(val))/sd[:,None]
    lag=.5*(g[:n,n:2*n]+g[:n,2*n:-1]+(s[:n,n:2*n]+s[:n,2*n:-1])*np.outer(ratio,ratio))
    op=root.T@lag@root
    eig,axis=np.linalg.eigh(.5*(op+op.T))
    reward=.5*root.T@(g[:n,-1]+s[:n,-1]*ratio*np.sqrt(g[-1,-1]/s[-1,-1]))
    pick=np.argsort(abs(eig*(axis.T@reward)))[-8:]
    z=root@np.column_stack((reward,axis[:,pick]))
    scale=np.sqrt(np.diag(z.T@g[:n,:n]@z)).clip(.001)
    z=z/scale
    gram=z.T@g[:n,:n]@z
    tc=z.T@(g[:n,n:2*n]+g[:n,2*n:-1])@z/2
    return z,gram,tc


def solve(g,xy):
    return np.linalg.solve(g+.22*np.eye(len(g)),xy)


def views(m,grid,parameters):
    z,g0,t0=parameters
    raw=m@z
    orig=solve(g0,t0[:,0]);sym=solve(g0,.5*(t0[:,0]+t0[0]))
    result=[]
    for a,b in market.parent.sessions(grid):
        u=raw[a:b].clip(-6,6);n=len(u)
        mat=np.zeros((n,3))
        gs=np.zeros((2,len(g0),len(g0)));ts=np.zeros_like(gs);mass=np.zeros(2)
        for start in range(0,n,64):
            if start:
                gs*=np.exp(-64/640);ts*=np.exp(-64/640);mass*=np.exp(-64/640)
                idx=(start//64)%2
                for lag in (20,40):
                    t=np.arange(max(lag,start-64),start,2)
                    past=u[t-lag];now=u[t]
                    gs[idx]+=(past.T@past)*.5
                    ts[idx]+=(past.T@now)*.5
                    mass[idx]+=.5*len(t)
            def read(g,t,weight,N):
                # Pseudo-observations use the same state coordinates as the unfolding session.
                denom=weight+N
                gg=(weight*g0+g)/denom
                c=(weight*t0+t)/denom
                return solve(gg,c[:,0])-orig,solve(gg,.5*(c[:,0]+c[0]))-sym
            f,s=read(gs.sum(0),ts.sum(0),170.,mass.sum())
            left=read(gs[0],ts[0],85.,mass[0])[1]
            right=read(gs[1],ts[1],85.,mass[1])[1]
            sl=slice(start,min(start+64,n));p=u[sl]
            l=p@left;r=p@right
            stable=np.where(l*r>0,np.sign(l)*np.minimum(abs(l),abs(r)),0.)
            mat[sl]=np.column_stack((p@f,p@s,stable))
        result.append(mat)
    return np.concatenate(result)


def stat(q,y,x):
    mat=np.column_stack((q.mean(1,dtype=float),x,y.mean(1,dtype=float)))
    mat-=mat.mean(0)
    return mat.T@mat/len(mat)


def fit(hist,j,h):
    c=np.stack(hist);g=c.mean(0);n=len(g)-3;idx=5+j
    ids=np.arange(5)
    v=np.zeros(n);v[idx]=1.
    hh=g[np.ix_(ids,ids)]
    v[ids]=-np.linalg.solve(hh+.05*np.diag(hh.diagonal().clip(1e-8)),g[ids,idx])
    slope=(c[:,h,n+h]/c[:,h,h]).clip(.25,4.)
    grad=(c[:,:n,n+h]/slope[:,None]-c[:,:n,h])@v
    mu=grad.mean();mu*=abs(mu)/(abs(mu)+grad.std()/np.sqrt(len(c))+1e-7)
    w=v*mu/(2.*(v@g[:n,:n]@v)+1e-8)
    risk=np.sqrt(w@g[:n,:n]@w/g[h,h])
    return w*min(1,.035/max(risk,1e-8))


def main():
    OUT.mkdir(exist_ok=True)
    st=[];mk=[];hist=[];rows={(k,h):[] for k in ARMS for h in (1,2)}
    for d in range(16):
        with np.load(data.OUT/f"day_{d:02d}.npz") as f:s,g,m=f["stock"],f["market"],f["state"]
        if d>=6:
            grid=np.load(base.CACHE/f"day_{d:02d}_time.npy").astype(int)
            y=np.load(base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r")
            if d in (6,8,12):params=model(st,mk)
            with np.load(tape.OUT/f"day_{d:02d}.npz") as f:q=f["reference"];dm=f["change"];fixed=f["directions"][:,2]
            if d>=8:q+=np.load(precision.OUT/f"oof_{d:02d}.npy")[:,None]
            z=views(m,grid,params);x=np.column_stack((dm,fixed,z))
            if d in (8,12):weights={k:fit(hist,ARMS.index(k[0]),k[1]) for k in rows}
            if d>=8:
                xx=np.column_stack((q.mean(1,dtype=float),x))
                for (k,h),v in rows.items():
                    delta=np.broadcast_to((xx@weights[k,h])[:,None],q.shape[:2])
                    v.append(market.metric(q[...,h],y[...,h],delta))
                if d==8:
                    n=2531;np.testing.assert_allclose(z[:n],views(m[:n],grid[:n],params),atol=1e-7,rtol=0)
            hist.append(stat(q,y,x))
        st.append(s);mk.append(g)
    report={f"{k}_{h}":dict(folds=[market.gain(sum(v[:4])),market.gain(sum(v[4:]))],days=[market.gain(g) for g in v]) for (k,h),v in rows.items()}
    picks={}
    for h in (1,2):
        ok=[k for k in ARMS if min(report[f"{k}_{h}"]["folds"])>0 and sum(v>0 for v in report[f"{k}_{h}"]["days"])>=5]
        if ok:picks[str(h)]=max(ok,key=lambda k:min(report[f"{k}_{h}"]["folds"]))
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=picks,results=report),indent=2));print(picks,{k:v["folds"] for k,v in report.items()},flush=True)
    if not picks:return
    w=np.zeros((8,3))
    for h,k in picks.items():w[:,int(h)]=fit(hist,ARMS.index(k),int(h))
    basis,gram,lag=model(st,mk)
    np.savez(OUT/"weights.npz",basis=basis,gram=gram,lag=lag,w=w)
    outer=[]
    for d in range(16,20):
        grid,r,y=tape.outer(d)
        raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        with np.load(market.parent.OUT/"weights.npz") as f:
            _,av=market.parent.state(raw,grid,f["code"],with_market=True)
        m=market.market_state(av,grid);q=r["reference"].copy()
        with np.load(precision.OUT/"weights.npz") as f:
            xx=np.column_stack((q.mean(1,dtype=float),r["change"],r["directions"][:,2],m@f["bank"]))
            q+=(xx@f["w"]).astype(q.dtype)[:,None]
        with np.load(OUT/"weights.npz") as f:
            z=views(m,grid,(f["basis"],f["gram"],f["lag"]))
            xx=np.column_stack((q.mean(1,dtype=float),r["change"],r["directions"][:,2],z))
            pred=q+(xx@f["w"]).astype(q.dtype)[:,None]
        outer.append((q,y,pred))
    score=market.summary(outer)
    assert score["rows"]==936250 and abs(score["baseline_score"]-json.loads((precision.OUT/"outer.json").read_text())["pooled"]["score"])<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=score,daily=[market.summary([r]) for r in outer]),indent=2));print(score,flush=True)


if __name__=="__main__":main()
