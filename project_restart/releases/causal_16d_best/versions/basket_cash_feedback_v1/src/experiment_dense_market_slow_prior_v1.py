"""Regularize a market-wide slow book operator with stock-relative dynamics from the same training dates."""
import json
import numpy as np
import experiment_dense_slow_payoff_modes_v1 as parent

base,TAPE,released,metric,gain,summary=parent.base,parent.TAPE,parent.released,parent.metric,parent.gain,parent.summary
OUT=base.ROOT/"artifacts/dense_market_slow_prior_v1"
ARMS=("market_only","pooled_dynamics","stock_payoff_prior")


def market_state(m,grid):
    chunks=[]
    for a,b in parent.sessions(grid):
        block=m[a:b]
        # Remove the opening offset using a past-only local reference.
        chunks.append(block-parent.smooth(block,320))
    return np.concatenate(chunks)


def record(x,grid,y):
    total=0;count=0
    for a,b in parent.sessions(grid):
        t=np.arange(a,b-60,2)
        pay=base.UNITS[1]*y[t,:,1].mean(1,dtype=float)
        v=np.column_stack((x[t],x[t+20],x[t+40],pay))
        total+=v.T@v;count+=len(v)
    return total/count


def operators(stock,market):
    g=np.mean(market,0);s=np.mean(stock,0);n=(len(g)-1)//3
    sd=np.sqrt(g.diagonal()[:n]).clip(.006);ss=np.sqrt(s.diagonal()[:n]).clip(.03)
    target_ratio=np.sqrt(g[-1,-1]/s[-1,-1])
    match=np.r_[np.tile(sd/ss,3),target_ratio]
    # Express the stock-relative prior in the units of the observed market state.
    translated=[m*np.outer(match,match) for m in stock]
    def extract(seq):
        parts=[]
        common=np.mean(seq,0)
        cov=common[:n,:n]/np.outer(sd,sd)
        val,v=np.linalg.eigh(cov+.25*np.eye(n))
        root=((v/np.sqrt(val))@v.T)/sd[:,None]
        for parity in (0,1):
            trans=np.mean(seq[parity::2],0)
            weights=[]
            for h in (1,2):
                op=root.T@trans[:n,n*h:n*(h+1)]@root
                weights.append(.5*(op+op.T))
            operator=root@sum(weights)@root.T
            parts.append((operator,np.mean(seq[1-parity::2],0)[:n,-1]))
        return parts
    market_ops=extract(market)
    mixed_ops=extract([.5*(a+b) for a,b in zip(market,translated)])
    rows=[]
    for j in range(2):
        a,pay=market_ops[j]
        b,_=mixed_ops[j]
        prior=np.mean(translated[1-j::2],0)[:n,-1]
        rows.append(np.stack((a@pay,b@pay,b@(.5*pay+.5*prior)),1))
    return np.mean(rows,0)


def stats(p,y,output):
    market=p.mean(1,dtype=float)
    mat=np.column_stack((market,output,y[...,2].mean(1,dtype=float)))
    mat-=mat.mean(0)
    return mat.T@mat/len(mat)


def fit(hist,arm):
    c=np.stack(hist);g=c.mean(0);col=3+arm
    ids=np.arange(3)
    # Compare stock-regularized transfers to the standalone market readout.
    if arm:ids=np.arange(4)
    v=np.zeros(6);v[col]=1.
    h=g[np.ix_(ids,ids)]
    v[ids]=-np.linalg.solve(h+.06*np.diag(h.diagonal().clip(1e-7)),g[ids,col])
    slope=(c[:,2,-1]/c[:,2,2]).clip(.25,4)
    grad=(c[:,:-1,-1]/slope[:,None]-c[:,:-1,2])@v
    mu=grad.mean()
    mu*=abs(mu)/(abs(mu)+grad.std()/np.sqrt(len(grad))+1e-7)
    w=v*mu/(2.2*(v@g[:-1,:-1]@v)+1e-7)
    risk=np.sqrt(w@g[:-1,:-1]@w/g[2,2])
    return w*min(1,.05/max(risk,1e-8))


def main():
    OUT.mkdir(exist_ok=True)
    code=parent.codec();stock=[];market=[];pcal=[];cal=[];rows={k:[] for k in ARMS}
    for d in range(16):
        load=lambda k:np.load(base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y")
        x,av=parent.state(raw,grid,code,with_market=True)
        m=market_state(av,grid)
        if d>=4:
            if d%4==0:bank=operators(stock,market);pb=parent.components(stock)
            if d in (8,12):weights={k:fit(cal,j) for j,k in enumerate(ARMS)};pw=parent.fit(pcal,0)
            p=np.load(TAPE/f"forecast_{d:02d}.npy")
            z=m@bank;ref=x@pb
            if d>=8:
                prior=p[...,2]+np.concatenate((p-p.mean(1,keepdims=True),ref),2)@pw
                xx=np.column_stack((p.mean(1,dtype=float),z))
                for k,w in weights.items():
                    delta=np.broadcast_to((xx@w)[:,None],prior.shape)
                    rows[k].append(metric(prior,y[...,2],delta))
                if d==8:
                    n=2503
                    _,short=parent.state(raw[:n],grid[:n],code,with_market=True)
                    np.testing.assert_allclose(z[:n],market_state(short,grid[:n])@bank,atol=1e-7,rtol=0)
            cal.append(stats(p,y,z))
            pcal.append(parent.stats(p,y,ref))
        stock.append(parent.moment(x,grid,y))
        market.append(record(m,grid,y))
    report={k:dict(folds=[gain(sum(v[:4])),gain(sum(v[4:]))],days=[gain(s) for s in v]) for k,v in rows.items()}
    ok=[k for k in ARMS if min(report[k]["folds"])>0 and sum(v>0 for v in report[k]["days"])>=5]
    pick=max(ok,key=lambda k:min(report[k]["folds"])) if ok else None
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=pick,results=report),indent=2))
    print(pick,{k:v["folds"] for k,v in report.items()},flush=True)
    if not pick:return
    np.savez(OUT/"weights.npz",bank=operators(stock,market),w=fit(cal,ARMS.index(pick)))
    outer=[]
    for d in range(16,20):
        load=lambda k:np.load(base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid=load("x"),load("time").astype(int)
        p=released(d,raw)
        with np.load(parent.OUT/"weights.npz") as f,np.load(OUT/"weights.npz") as w:
            x,av=parent.state(raw,grid,f["code"],with_market=True)
            z=x@f["bank"]
            old=p.copy();old[...,2]+=np.concatenate((p-p.mean(1,keepdims=True),z),2)@f["w"]
            delta=np.column_stack((p.mean(1,dtype=float),market_state(av,grid)@w["bank"]))@w["w"]
            q=old.copy();q[...,2]+=delta[:,None]
        outer.append((old,load("y"),q))
    result=summary(outer)
    assert result["rows"]==936250 and abs(result["baseline_score"]-json.loads((parent.OUT/"outer.json").read_text())["pooled"]["score"])<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=result,daily=[summary([r]) for r in outer]),indent=2))
    print(result,flush=True)


if __name__=="__main__":main()
