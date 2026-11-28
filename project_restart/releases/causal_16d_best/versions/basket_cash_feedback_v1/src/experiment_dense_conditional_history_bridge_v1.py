"""Learn return-relevant transitions between book histories after conditioning both sides on the current quote."""
from functools import lru_cache

import numpy as np

import experiment_dense_reconstructed_opening_history_v1 as init
from prepare_dense_opening_history_posterior_reference_v1 import OUT as REFERENCE

onset,rel,book = init.onset,init.rel,init.book
OUT = REFERENCE.parent / "dense_conditional_history_bridge_v1"
ARMS = ("past_payoff","future_quote_bridge","past_after_future_control")
BANDS = ((1,5),(6,20),(21,60))
RANK = 10
N = RANK*len(BANDS)


def quote_windows(total,field,t,sign):
    rows = []
    for lo,hi in BANDS:
        a,b = (t+lo,t+hi) if sign==1 else (t-hi,t-lo)
        rows.append((total[b+1]-total[a])/(hi-lo+1)-field[t])
    return np.concatenate(rows,axis=2)


def summaries(raw,grid,stride,offset,both):
    for a,b in rel.fields.parent.sessions(grid):
        field = np.arcsinh(raw[a:b][:,:,init.F].astype(np.float32))
        total = np.concatenate((np.zeros_like(field[:1]),field.cumsum(0,dtype=float)))
        times = np.arange(60+offset,b-a-(60 if both else 0),stride)
        for i in range(0,len(times),80):
            t = times[i:i+80]
            phi = init.inputs(raw[a+t]).astype(np.float32)
            past = quote_windows(total,field,t,-1)
            future = quote_windows(total,field,t,1) if both else None
            yield a+t,phi,past,future


def learn_encoder():
    cov,cross,yy = 0.,0.,0.
    count = 0
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        for _,phi,past,future in summaries(raw,grid,10,d,True):
            x = phi.reshape(-1,phi.shape[-1]).astype(float)
            y = np.concatenate((past,future),axis=2).reshape(len(x),-1)
            cov += x.T@x
            cross += x.T@y
            yy += y.T@y
            count += len(x)
    cov,cross,yy = cov/count,cross/count,yy/count
    w = np.linalg.solve(cov+.09*np.diag(cov.diagonal().clip(.003)),cross)
    residual = yy-w.T@cross-cross.T@w+w.T@cov@w
    f = len(init.F)
    pooled = sum(residual[i*f:(i+1)*f,i*f:(i+1)*f] for i in range(6))/6
    scale = np.sqrt(pooled.diagonal().clip(.02))
    eig,vec = np.linalg.eigh(pooled/np.outer(scale,scale))
    basis = vec[:,-RANK:]/scale[:,None]/np.sqrt(eig[-RANK:].clip(.03))
    proj = np.kron(np.eye(3),basis)
    left,right = np.split(w,2,axis=1)
    np.savez(OUT/"encoder.npz",basis=proj,wp=left@proj,wf=right@proj)
    encoder.cache_clear()


@lru_cache(maxsize=1)
def encoder():
    with np.load(OUT/"encoder.npz") as f:
        return tuple(f[k].astype(np.float32) for k in ("basis","wp","wf"))


def encode(phi,window,basis,weight):
    x = window.astype(np.float32)@basis-phi@weight
    return (x-x.mean(1,keepdims=True)).clip(-5,5)


def collect():
    OUT.mkdir(exist_ok=True)
    learn_encoder()
    basis,wp,wf = encoder()
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6
             else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        cur = p-p.mean(1,keepdims=True)
        label = y-y.mean(1,keepdims=True)
        scale = ((cur*label).sum((0,1))/(cur*cur).sum((0,1))).clip(.3,3)
        err = label/scale-cur
        gram = np.zeros((2*N+6,2*N+6))
        count = 0
        for t,phi,past,future in summaries(raw,grid,8,d%8,True):
            x,v = encode(phi,past,basis,wp),encode(phi,future,basis,wf)
            z = np.concatenate((x,v,cur[t],err[t]),axis=2).reshape(-1,2*N+6).astype(float)
            gram += z.T@z
            count += len(z)
        np.save(OUT/f"gram_{d:02d}.npy",gram/count)


def ridge(cov):
    return cov+.55*np.diag(cov.diagonal().clip(.005))


def train(end):
    rows = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    g = rows.mean(0)
    # Remove the baseline forecast directions before fitting both halves of the bridge.
    q = np.arange(2*N,2*N+3)
    tr = np.eye(2*N+6)
    tr[np.ix_(q,np.arange(2*N))] = -np.linalg.solve(ridge(g[np.ix_(q,q)]),g[q,:2*N])
    rows = tr.T@rows@tr
    w = np.zeros((N,len(ARMS),3))
    for parity in (0,1):
        dyn,ret = (rows[j::2].mean(0) for j in (parity,1-parity))
        transition = np.linalg.solve(ridge(dyn[:N,:N]),dyn[:N,N:2*N])
        payoff = np.linalg.solve(ridge(ret[N:2*N,N:2*N]),ret[N:2*N,-3:])
        direct = np.linalg.solve(ridge(ret[:N,:N]),ret[:N,-3:])
        w[:,0] += direct*.5
        w[:,1] += (transition@payoff)*.5
        w[:,2] += (direct-transition@payoff)*.5
    return np.concatenate((w,np.einsum("ij,jah->iah",tr[q,:N],w)))


def design(raw,grid,code,p,bank):
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    basis,wp,_ = encoder()
    for t,phi,past,_ in summaries(raw,grid,1,0,False):
        x = encode(phi,past,basis,wp)
        current = p[t]-p[t].mean(1,keepdims=True)
        v = np.concatenate((x,current),axis=2)@bank.reshape(N+3,-1)
        out[t] = v.reshape(len(t),50,len(ARMS),3)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
