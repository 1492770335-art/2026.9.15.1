"""Transport a learned 20-tick payoff through empirical book transitions fitted on disjoint dates."""
import json
import numpy as np

from experiment_dense_near_matched_spreads_v1 import sources
from experiment_dense_decoded_counts_v1 import base,TAPE,codec,released,sessions,metric,gain,summary
from experiment_dense_robust_distal_readout_v1 import smooth

OUT=base.ROOT/"artifacts/dense_slow_payoff_modes_v1"
ARMS=("reversible","directed","additional_memory")


def state(raw,grid,code,*,with_market=False):
    src=sources(raw,grid,code)
    chunks=[]
    market=[]
    for a,b in sessions(grid):
        s=raw[a:b]
        fields=np.tanh(.65*np.arcsinh(s[:,:,[7,10,11,14,21,23,25,32,39,40,47,50,60,64,67,72,74,75,76]].astype(float)))
        flags=s[:,:,83:].astype(float)
        flags=np.tanh(.5*(flags-flags.mean(1,keepdims=True)))
        book=src[a:b].reshape(b-a,50,3,6).sum(2)
        # Interactions express price pressure under different depth and print activity.
        mark=book*np.abs(book[:,:,-1:])
        x=np.concatenate((src[a:b],fields,smooth(fields,24),flags,smooth(flags,24),mark),2)
        average=x.mean(1,keepdims=True)
        if with_market:market.append(average[:,0])
        x-=average
        chunks.append(x.astype("float32"))
    local=np.concatenate(chunks)
    return (local,np.concatenate(market)) if with_market else local


def moment(x,grid,y):
    grams=[]
    for a,b in sessions(grid):
        ix=np.arange(a,b-60,5)
        if not len(ix):continue
        pay=y[ix,:,1].astype(float)*base.UNITS[1]
        pay-=pay.mean(1,keepdims=True)
        m=np.concatenate((x[ix],x[ix+20],x[ix+40],pay[:,:,None]),2)
        z=m.reshape(-1,m.shape[-1]);grams.append(z.T@z)
    return sum(grams)/sum(len(np.arange(a,b-60,5))*50 for a,b in sessions(grid))


def components(hist):
    g=np.mean(hist,0)
    n=(len(g)-1)//3
    sd=np.sqrt(g.diagonal()[:n]).clip(.03)
    cov=g[:n,:n]/np.outer(sd,sd)
    val,vec=np.linalg.eigh(cov+.12*np.eye(n))
    root=(vec/np.sqrt(val))@vec.T
    root=root/sd[:,None]
    ops=[];load=[]
    for subset in (hist[::2],hist[1::2]):
        sub=np.mean(subset,0)
        ops.append([root.T@sub[:n,i*n:(i+1)*n]@root for i in (1,2)])
        load.append(root.T@sub[:n,-1])
    bank=[]
    for (k20,k40),other,pay in ((ops[0],ops[1],load[1]),(ops[1],ops[0],load[0])):
        # A state with persistent temporal correlation need not carry directional information.
        # Split off antisymmetric transition and the part not captured by two 20-tick steps.
        symmetric=lambda k:(k+k.T)*.5
        even=(symmetric(k20)+symmetric(k40))@pay
        oriented=(k20+k40)@pay-even
        missed=(k40-k20@other[0])@pay
        bank.append(root@np.stack((even,oriented,missed),1))
    return np.mean(bank,0)


def stats(p,y,z):
    a=p.astype(float);a-=a.mean(1,keepdims=True)
    b=y[...,2].astype(float);b-=b.mean(1,keepdims=True)
    x=np.concatenate((a,z,b[:,:,None]),2)
    x=x.reshape(-1,x.shape[2])
    return x.T@x/len(x)


def fit(hist,j):
    c=np.stack(hist)
    g=c.mean(0)
    col=3+j
    v=np.zeros(6);v[col]=1.
    ids=np.arange(col)
    gram=g[np.ix_(ids,ids)]
    v[ids]=-np.linalg.solve(gram+.035*np.diag(gram.diagonal().clip(1e-8)),g[ids,col])
    n=len(v)
    dev=(c[:,:n,-1]/(c[:,2,-1]/c[:,2,2]).clip(.35,3.)[:,None]-c[:,:n,2])@v
    mean=dev.mean()
    mean*=abs(mean)/(abs(mean)+dev.std()/np.sqrt(len(hist))+1e-7)
    coef=mean/(1.5*(v@g[:n,:n]@v)+1e-8)
    risk=abs(coef)*np.sqrt((v@g[:n,:n]@v)/g[2,2])
    return v*coef*min(1,.05/max(risk,1e-8))


def main():
    OUT.mkdir(exist_ok=True)
    code=codec()
    operators=[];cal=[];rows={k:[] for k in ARMS}
    for d in range(16):
        load=lambda s:np.load(base.CACHE/f"day_{d:02d}_{s}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y")
        x=state(raw,grid,code)
        if d>=4:
            if d%4==0:bank=components(operators)
            if d in (8,12):weights={k:fit(cal,j) for j,k in enumerate(ARMS)}
            p=np.load(TAPE/f"forecast_{d:02d}.npy")
            z=x@bank
            if d>=8:
                source=np.concatenate((p-p.mean(1,keepdims=True),z),2)
                for k in ARMS:rows[k].append(metric(p[...,2],y[...,2],source@weights[k]))
                if d==8:
                    n=2531
                    short=state(raw[:n],grid[:n],code)@bank
                    np.testing.assert_allclose(short,z[:n],rtol=0,atol=1e-7)
            cal.append(stats(p,y,z))
        operators.append(moment(x,grid,y))
    report={k:dict(folds=[gain(sum(v[:4])),gain(sum(v[4:]))],days=[gain(g) for g in v]) for k,v in rows.items()}
    ok=[k for k in ARMS if min(report[k]["folds"])>0 and sum(v>0 for v in report[k]["days"])>=5]
    best=max(ok,key=lambda k:min(report[k]["folds"])) if ok else None
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=best,results=report),indent=2))
    print(best,{k:v["folds"] for k,v in report.items()},flush=True)
    if not best:return
    np.savez(OUT/"weights.npz",code=code,bank=components(operators),w=fit(cal,ARMS.index(best)))
    outer=[]
    for d in range(16,20):
        load=lambda s:np.load(base.CACHE/f"day_{d:02d}_{s}.npy",mmap_mode="r")
        raw,grid=load("x"),load("time").astype(int)
        p=released(d,raw)
        with np.load(OUT/"weights.npz") as f:
            z=state(raw,grid,f["code"])@f["bank"]
            q=p.copy();q[...,2]+=np.concatenate((p-p.mean(1,keepdims=True),z),2)@f["w"]
        outer.append((p,load("y"),q))
    result=summary(outer)
    assert result["rows"]==936250 and abs(result["baseline_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=result,daily=[summary([r]) for r in outer]),indent=2))
    print(result,flush=True)


if __name__=="__main__":main()
