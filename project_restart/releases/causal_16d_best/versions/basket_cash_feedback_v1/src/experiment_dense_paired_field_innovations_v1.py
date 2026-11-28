"""Estimate the small innovation signal with true aligned history/future pairs rather than stationary lag substitution."""
import numpy as np
import experiment_dense_field_news_history_v1 as path

fields,news,memory,ref = path.fields,path.news,path.memory,path.ref
OUT = ref.base.ROOT/"artifacts/dense_paired_field_innovations_v1"
ARMS = ("aligned_pairs","independent_dynamics","held_out_projection")


def sample(z,grid,core):
    total = 0
    count = 0
    for a,b in fields.parent.sessions(grid):
        for t in np.array_split(np.arange(a+20,b-61,8),6):
            if not len(t):continue
            # The same instruments and forecast origins enter all five terms of the cancellation.
            target = np.concatenate((z[t+20,...,core:]+z[t+40,...,core:],
                z[t+20,...,:core]+z[t+40,...,:core],z[t,...,:core]+z[t+20,...,:core],
                z[t-20,...,:core]+z[t,...,:core]),axis=-1)
            current = z[t,...,:core].reshape(-1,core).astype(float)
            total += current.T@target.reshape(-1,target.shape[-1]).astype(float)
            count += len(current)
    return total/count


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        load = lambda k: np.load(ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid = load("x"),load("time").astype(int)
        x,av = fields.parent.state(raw,grid,fields.parent.codec(),with_market=True)
        core = x.shape[-1]
        m = fields.book_inputs(raw,grid,ref.market.market_state(av,grid),{})
        omitted=np.tanh(.65*np.arcsinh(raw[:,:,fields.COLUMNS].astype(float)))
        omitted-=omitted.mean(1,keepdims=True)
        z=np.concatenate((x,omitted.astype(np.float32)),axis=-1)
        np.savez(OUT/f"day_{d:02d}.npz",stock=sample(z,grid,core),market=sample(m,grid,core))


def train(history):
    data=[]
    for d in range(len(history)):
        with np.load(fields.OUT/f"moments_{d:02d}.npz") as f,np.load(OUT/f"day_{d:02d}.npz") as r:
            data.append((f["cov"],f["pay"],f["variance"],np.load(news.delayed.OUT/f"payoffs_{d:02d}.npy"),r["stock"],r["market"]))
    grams,payoffs,var,delays,bs,bm=(np.stack(v) for v in zip(*data))
    sd=np.sqrt(grams[:,1,0].mean(0).diagonal()).clip(.006)
    ss=np.sqrt(grams[:,0,0].mean(0).diagonal()).clip(.03)
    n=len(sd)-len(fields.COLUMNS)
    scale=np.sqrt(var[:,1].mean()/var[:,0].mean())
    cov=.5*(grams[:,0]/np.outer(ss,ss)+grams[:,1]/np.outer(sd,sd))
    pay=.5*(payoffs[:,0]*scale/ss+payoffs[:,1]/sd)
    past=.5*(delays[:,0,:,0]*scale/ss[:n]+delays[:,1,:,0]/sd[:n])
    def scaled(o,units):
        return o/np.outer(units[:n],np.r_[units[n:],np.tile(units[:n],3)])
    paired=.5*(scaled(bs,ss)+scaled(bm,sd))
    nf=len(sd)-n
    coefficients=[]
    for fold in (0,1):
        first,second=slice(fold,None,2),slice(1-fold,None,2)
        k=cov[first].mean(0)
        cross=lambda a,b:k[b-a] if b>=a else k[a-b].T
        dates=(0,-1,-2)
        gram=np.block([[cross(a,b)[:n,:n] for b in dates] for a in dates])
        field=np.concatenate([cross(a,0)[:n,n:] for a in dates])
        proj=np.linalg.solve(gram+.17*np.eye(len(gram)),field)
        er=k[0,n:,n:]-field.T@proj-proj.T@field+proj.T@gram@proj
        grad=pay[second,n:].mean(0)-proj.T@np.r_[pay[second,:n].mean(0),past[second].mean(0).ravel()]
        reward=np.linalg.solve(er+.29*np.eye(nf),grad)
        original=sum(cross(0,h)[:n,n:]-np.concatenate([cross(0,h+t)[:n,:n] for t in dates],axis=1)@proj for h in (1,2))
        q=k[0,:n,:n]+.22*np.eye(n)
        def realized(period):
            block=paired[period].mean(0)
            return np.linalg.solve(q,(block[:,:nf]-block[:,nf:]@proj)@reward)
        baseline=np.linalg.solve(q,original@reward)
        matched,independent=realized(first),realized(second)
        coefficients.append(np.column_stack((matched-baseline,independent-matched,independent-baseline))/sd[:n,None])
    return dict(bank=np.mean(coefficients,axis=0))


def views(m,grid,model):
    z=m@model["bank"]
    for a,b in fields.parent.sessions(grid):
        z[a:min(a+40,b)]=0.
    return z


if __name__=="__main__":
    collect()
    memory.runner.run(OUT,ARMS,train,views,prior_head=(memory,news,path),keep_oof=True)
