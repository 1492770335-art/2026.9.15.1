"""Update the price of book information with partial and fully settled returns sharing one forecast expiry."""
import numpy as np
from scipy.signal import lfilter
import experiment_dense_projected_future_surprises_v1 as proj
from experiment_dense_matured_value_evidence_v1 import GEO,check_template

ref,rel=proj.ref,proj.rel
OUT=ref.base.ROOT/"artifacts/dense_value_maturity_updates_v1"
ARMS=("settled","marked_same_expiry","pending_only")
USES_FORECAST=True


def train(days):
    w=proj.train(days)
    scales=[];risk=[]
    for d in range(4,days):
        p=proj.future.forecast(d).astype(float)
        y=np.load(ref.base.CACHE/f"day_{d:02d}_y.npy",mmap_mode="r").astype(float)
        y-=y.mean(1,keepdims=True);p-=p.mean(1,keepdims=True)
        scales.append(((p*y).mean((0,1))/(p**2).mean((0,1))).clip(.2,4))
        risk.append(np.mean(y[:,:,2]**2))
    np.savez(OUT/f"params_{days}.npz",w=w,slope=np.mean(scales,0),risk=np.sqrt(np.mean(risk)))
    return np.array([days])


def views(raw,grid,code,bank,p):
    with np.load(OUT/f"params_{int(bank[0])}.npz") as f:w,price,risk=f["w"],f["slope"],float(f["risk"])
    signal=proj.views(raw,grid,code,w)
    q=p.astype(float)-p.mean(1,keepdims=True,dtype=float)
    gram=q.transpose(0,2,1)@q
    load=np.linalg.solve(gram+.03*np.eye(3)*gram.diagonal(axis1=1,axis2=2)[:,None].clip(.005),
                         q.transpose(0,2,1)@signal)
    signal-=q@load
    signal/=np.sqrt(np.mean(signal**2,1,keepdims=True)+.04)
    with np.load(GEO) as f:k,co,row=float(f["k"]),f["coeff"][2,1],f["row"]
    out=np.zeros((*q.shape[:2],3))
    for a,b in rel.fields.parent.sessions(grid):
        z=signal[a:b];n=len(z)
        block=raw[a:b]
        path=np.vstack((np.zeros(50),np.cumsum(np.log1p(k*(block[:,:,0].astype(float)-.00136)),0)))
        gaps=np.vstack((np.zeros(50,int),np.cumsum(np.all(block==row,axis=2),0)))
        stop=np.arange(0,n,5)
        if not len(stop):continue
        step_decay=np.exp(-5/240.)
        def ema(arr):return lfilter([1-step_decay],[1,-step_decay],arr,axis=0)
        updates=[]
        for horizon in (20,40,60):
            end=stop[stop>=horizon];origin=end-horizon
            marginal=np.zeros((len(stop),3));size=np.zeros_like(marginal)
            ok=gaps[end+1]==gaps[origin]
            mass=ok.sum(1).clip(1)
            observed=co/k*np.expm1(path[end+1]-path[origin+1])
            remaining=np.zeros_like(observed)
            if horizon<60:
                remaining=ref.base.UNITS[1]*q[a+end,:,1]*price[1]
                if horizon==20:remaining=.5*(remaining+q[a+end,:,2]*price[2])
            # At each stop the price path and the shorter forecasts have already been observed.
            error=observed+remaining-q[a+origin,:,2]*price[2]
            error=(error-(error*ok).sum(1,keepdims=True)/mass[:,None])/risk
            e=np.clip(error,-4,4)*ok
            marginal[horizon//5:]=np.einsum("tih,ti->th",z[origin],e)/mass[:,None]
            size[horizon//5:]=np.einsum("tih,ti->th",z[origin]**2,ok)/mass[:,None]
            updates.append((ema(marginal),ema(size)))
        full,den=updates[2]
        pending=np.zeros_like(full);pending_den=np.zeros_like(full)
        for (estimate,exposure),h in zip(updates[:2],(20,40)):
            width=4 # four five-tick updates until the next vintage supersedes this one
            def recent(arr):
                tail=arr.copy()
                tail[width:]-=step_decay**width*arr[:-width]
                return tail
            age=np.exp(-h/240.)
            pending+=age*recent(estimate)
            pending_den+=age*recent(exposure)
        settled=np.exp(-60/240.)
        rates=(full/(.5+den),(settled*full+pending)/(.5+settled*den+pending_den),
               pending/(.15+pending_den))
        for j,r in enumerate(rates):
            out[a:b,:,j]=np.einsum("tih,th->ti",z,r[np.arange(n)//5].clip(-1,1))/3
    return out-out.mean(1,keepdims=True)


if __name__=="__main__":
    check_template()
    rel.main(OUT,ARMS,train,views,with_prediction=True)
