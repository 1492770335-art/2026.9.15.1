"""Read donor stock payoffs at horizons matching a receiver's historical feature persistence."""
import json
import numpy as np
import experiment_dense_market_horizon_evidence_v1 as ref

base=ref.base
market=ref.market
parent=market.parent
OUT=base.ROOT/"artifacts/dense_recipient_clocks_v1"
ARMS=("recipient_clock","matched_donors","own_payoff_clock")
LAGS=np.array(ref.clock.clock.LAGS)
HORIZONS=np.array(ref.clock.clock.HORIZONS)


def basis():
    rows=[]
    for d in range(4):
        with np.load(ref.clock.clock.OUT/f"day_{d:02d}.npz") as f:rows.append(f["stock"])
    cov=np.mean(rows,0)
    sd=np.sqrt(cov[0].diagonal()).clip(.03)
    eig,u=np.linalg.eigh(cov[0]/np.outer(sd,sd)+.2*np.eye(len(sd)))
    root=(u/np.sqrt(eig))/sd[:,None]
    slow=root.T@cov[3:5].mean(0)@root
    _,rotate=np.linalg.eigh(.5*(slow+slow.T))
    return root@rotate


def record(x,grid,y,transform):
    K=transform.shape[1]
    powers=np.zeros((len(LAGS),50,K));payoffs=np.zeros((50,K,len(HORIZONS)))
    target_var=np.zeros((50,len(HORIZONS)));n=0
    for a,b in parent.sessions(grid):
        t=np.arange(a,b-80,6)
        if not len(t):continue
        current=x[t].astype(float)@transform
        for i,h in enumerate(LAGS):
            later=x[t+h].astype(float)@transform
            powers[i]+=(current*later).sum(0)
        f,m=base.UNITS[:2]
        ys=np.stack((f*y[t,:,0],f*(y[t,:,0]+y[t+5,:,0]),m*y[t,:,1],
                     m*(y[t,:,1]+y[t+20,:,1]),y[t,:,2]),2).astype(float)
        ys-=ys.mean(1,keepdims=True)
        payoffs+=np.einsum("tsi,tsh->sih",current,ys,optimize=True)
        target_var+=(ys**2).sum(0);n+=len(t)
    return powers/n,payoffs/n,target_var/n


def weights(history,transform):
    P,B,V=(np.stack(z) for z in zip(*history))
    bank=[];ranges=[]
    for j in (0,1):
        cov=P[j::2].mean(0);others=slice(1-j,None,2)
        # Borrow from other instruments. Distinguish the recipient's dynamics from its own realized payoffs.
        peer=(cov.sum(1,keepdims=True)-cov)/49
        stock=.75*cov+.25*peer
        source=peer/peer[:1].clip(.02);target=stock/stock[:1].clip(.02)
        raw=B[others].mean(0);var=V[others].mean(0)
        stock_sd=np.sqrt(stock[0]).clip(.05)
        pscore=raw/stock_sd[:,:,None]/np.sqrt(var[:,None,:]).clip(.09)
        vals=[]
        for s in range(50):
            ids=np.arange(50)!=s
            delta=target[3,s]-source[3,s]
            def interpolate(curve):
                return np.stack([np.interp(HORIZONS,LAGS,v) for v in curve.T],axis=1)
            src=interpolate(source[:,s])
            match=ref.clock.select_weights(src,target[3,s],20,HORIZONS)
            avg=ref.clock.select_weights(src,source[3,s],20,HORIZONS)
            # Identify donors using the whole state decay profile, independent of realized returns.
            dist=np.mean((target[2:5,ids]-target[2:5,s,None])**2,axis=(0,2))
            affinity=np.exp(-dist/np.median(dist).clip(.0005))
            affinity/=affinity.sum().clip(1e-8)
            allpay=pscore[ids].mean(0)
            similar=np.einsum("s,sij->ij",affinity,pscore[ids])-allpay
            own=pscore[s]-allpay
            speed=target[3:5,s].mean(0).clip(-.5,1)
            value=np.stack((((match-avg).T*allpay).sum(1),(match.T*similar).sum(1),
                            ((match-avg).T*own).sum(1)),1)
            vals.append(transform@(value*(speed*np.sqrt(var[s,2])/stock_sd[s])[:,None]))
            ranges.append(float(np.sqrt(np.mean(delta**2))))
        bank.append(vals)
    return np.mean(bank,0),float(np.mean(ranges))


def design(x,bank):
    z=np.einsum("tsi,sik->tsk",x,bank,optimize=True);return z-z.mean(1,keepdims=True)


def features(p,z):
    pp=p-p.mean(1,keepdims=True)
    return np.concatenate((pp,z),2)


def stats(p,y,z):
    v=np.concatenate((features(p,z),y-y.mean(1,keepdims=True)),2)
    xx=v.reshape(-1,v.shape[2]).astype(float)
    return xx.T@xx/len(xx)


def fit(cal,j,h):
    c=np.stack(cal);g=c.mean(0);n=len(g)-3
    v=np.zeros(n);v[3+j]=1
    v[:3]=-np.linalg.solve(g[:3,:3]+.025*np.diag(g.diagonal()[:3]),g[:3,3+j])
    slope=(c[:,h,n+h]/c[:,h,h]).clip(.3,3.)
    grad=(c[:,:n,n+h]/slope[:,None]-c[:,:n,h])@v
    mu=grad.mean();mu*=abs(mu)/(abs(mu)+grad.std()/np.sqrt(len(c))+1e-7)
    w=v*mu/(1.6*(v@g[:n,:n]@v)+1e-8)
    risk=np.sqrt(w@g[:n,:n]@w/g[h,h])
    return w*min(1,.05/max(risk,1e-8))


def main():
    OUT.mkdir(exist_ok=True)
    transform=basis();code=parent.codec();history=[];cal=[];rows={(k,h):[] for k in ARMS for h in (1,2)};diag={}
    for d in range(16):
        load=lambda k:np.load(base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y")
        x=parent.state(raw,grid,code)
        if d>=6:
            if d in (6,8,12):bank,diag[d]=weights(history,transform)
            with np.load(ref.tape.OUT/f"day_{d:02d}.npz") as f:q=f["reference"]
            if d>=8:
                for mod in (ref.precision,ref.nonlinear,ref.clock):q+=np.load(mod.OUT/f"oof_{d:02d}.npy")[:,None]
            z=design(x,bank)
            if d in (8,12):w={key:fit(cal,ARMS.index(key[0]),key[1]) for key in rows}
            if d>=8:
                xx=features(q,z)
                for (k,h),v in rows.items():v.append(market.metric(q[...,h],y[...,h],xx@w[k,h]))
                if d==8:
                    ix=2591
                    np.testing.assert_allclose(z[:ix],design(parent.state(raw[:ix],grid[:ix],code),bank),atol=1e-7,rtol=0)
            cal.append(stats(q,y,z))
        history.append(record(x,grid,y,transform))
    report={f"{k}_{h}":dict(folds=[market.gain(sum(v[:4])),market.gain(sum(v[4:]))],days=[market.gain(g) for g in v]) for (k,h),v in rows.items()}
    picks={}
    for h in (1,2):
        ok=[k for k in ARMS if min(report[f"{k}_{h}"]["folds"])>0 and sum(i>0 for i in report[f"{k}_{h}"]["days"])>=5]
        if ok:picks[str(h)]=max(ok,key=lambda k:min(report[f"{k}_{h}"]["folds"]))
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=picks,results=report,stock_clock_rms=diag),indent=2));print(picks,{k:v["folds"] for k,v in report.items()},diag,flush=True)
    if not picks:return
    w=np.zeros((6,3))
    for h,k in picks.items():w[:,int(h)]=fit(cal,ARMS.index(k),int(h))
    bank,_=weights(history,transform);np.savez(OUT/"weights.npz",bank=bank,w=w)
    outer=[]
    for d in range(16,20):
        grid,r,y=ref.tape.outer(d);raw=np.load(base.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r")
        x,av=parent.state(raw,grid,code,with_market=True)
        q,_=ref.latest(r["reference"],r,market.market_state(av,grid))
        with np.load(OUT/"weights.npz") as f:pred=q+(features(q,design(x,f["bank"]))@f["w"]).astype(q.dtype)
        outer.append((q,y,pred))
    score=market.summary(outer)
    assert score["rows"]==936250 and abs(score["baseline_score"]-json.loads((ref.clock.OUT/"outer.json").read_text())["pooled"]["score"])<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=score,daily=[market.summary([r]) for r in outer]),indent=2));print(score,flush=True)


if __name__=="__main__":main()
