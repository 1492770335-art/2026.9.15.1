"""Learn which calibration changes persist across time scales and the intraday session boundary."""
import numpy as np
import experiment_dense_forecast_strength_transfer_v1 as prev
from prepare_dense_forecast_strength_reference_v1 import OUT as REFERENCE

onset,rel = prev.onset,prev.rel
OUT = REFERENCE.parent/"dense_session_skill_memory_v1"
ARMS = ("recent_above_slow","previous_session_prior","confirmed_skill")
RANGES = prev.RANGES


def sessions(raw,grid,p):
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    timescales = np.array([120.,720.,2400.])
    decay = np.exp(-5./timescales)
    carry = np.zeros(9)
    for a,b in rel.fields.parent.sessions(grid):
        x,_ = prev.slope.frame(p[a:b])
        n = b-a
        h = np.zeros((3,3,3))
        count = np.zeros((3,3))
        cash = np.log1p(k*(raw[a:b,:,0].astype(float)-.00136)).cumsum(0)
        gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
        state = np.zeros((n,3,9))
        # Only past, fully observed intervals contribute to the morning prior.
        state[:,1] = carry[None]*np.exp(-np.arange(n)[:,None]/800)
        for due in range(5,n,5):
            for j,hor in enumerate((5,20,60)):
                t = due-hor
                if t<0:
                    continue
                ok = gaps[due+1]==gaps[t]
                qty = ok.sum().clip(1)
                error = np.expm1(cash[due]-cash[t])*coef[j,1]/k-p[a+t,:,j]
                error = (error-(error*ok).sum()/qty).clip(-3,3)*ok
                grad = (error@x[t])/qty
                h[:,j] = h[:,j]*decay[:,None]+grad
                count[:,j] = count[:,j]*decay+1
            fast,slow,total = (h/(count[:,:,None]+12)).reshape(3,9)
            sl = slice(due,min(due+5,n))
            state[sl,0] = fast-slow
            # The slower prior must have the same sign before a fast update is trusted.
            prior = .5*(slow+state[due,1])
            state[sl,2] = np.where(fast*prior>0,np.sign(fast)*np.minimum(abs(fast),abs(prior)),0.)
        carry = (h[-1]/(count[-1,:,None]+12)).ravel()
        yield a,b,x,state


def feature(t,x,state):
    ctx = np.column_stack((np.ones(len(t)),state[t].reshape(len(t),-1)))
    # fitter expects each group of 9 temporal observations adjacent across feature directions.
    interactions = np.einsum("tsf,tc->tscf",x[t],ctx).reshape(len(t),50,84)
    return interactions-interactions.mean(1,keepdims=True)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        v = p-p.mean(1,keepdims=True)
        label = y-y.mean(1,keepdims=True)
        scale = ((v*label).sum((0,1))/np.square(v).sum((0,1))).clip(.3,3)
        residual = label/scale-v
        gram = rhs = 0.
        for a,b,x,state in sessions(raw,grid,p):
            ticks = np.arange(d%5,b-a,5)
            for t in np.array_split(ticks,max(1,(len(ticks)+95)//96)):
                f = feature(t,x,state).reshape(-1,84)
                gram += f.T@f
                rhs += f.T@residual[a+t].reshape(-1,3)
        np.savez(OUT/f"day_{d:02d}.npz",g=gram/len(p),r=rhs/len(p))


def train(end):
    return prev.fitter.fit(end,OUT,RANGES)


def design(raw,grid,code,p,bank):
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b,x,state in sessions(raw,grid,p):
        for t in np.array_split(np.arange(b-a),(b-a+95)//96):
            pred = feature(t,x,state)@bank.reshape(len(bank),-1)
            z[a+t] = pred.reshape(len(t),50,len(ARMS),3)
    return z


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    for j in range(3):
        q[...,j] += (rel.relative.features(p,z[...,j])@params["w"][:,j]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
