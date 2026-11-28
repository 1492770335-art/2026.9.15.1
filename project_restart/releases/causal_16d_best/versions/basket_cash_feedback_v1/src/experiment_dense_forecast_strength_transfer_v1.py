"""Estimate separate forecast calibration from settled strong and weak cross-sectional signals."""
import numpy as np
import experiment_dense_prequential_slope_transfer_v1 as slope
from prepare_dense_prequential_slope_reference_v1 import OUT as REFERENCE
import experiment_dense_pressure_onset_order_v1 as fitter

onset,rel = slope.onset,slope.rel
OUT = REFERENCE.parent/"dense_forecast_strength_transfer_v1"
ARMS = ("all_signal_strengths","strong_vs_weak_origins","matched_receiver_strength")
RANGES = ((3,30),(30,57),(57,84))


def sessions(raw,grid,p):
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    for a,b in rel.fields.parent.sessions(grid):
        n = b-a
        x,_ = slope.frame(p[a:b])
        centered = p[a:b]-p[a:b].mean(1,keepdims=True)
        level = np.abs(centered)/np.sqrt(np.square(centered).mean(1,keepdims=True)).clip(.03)
        strength = np.tanh(1.5*(level-.8))
        strength -= strength.mean(1,keepdims=True)
        cash = np.log1p(k*(raw[a:b,:,0].astype(float)-.00136)).cumsum(0)
        gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
        hist = np.zeros((2,3,3))
        counts = np.zeros(3)
        states = np.zeros((n,2,9))
        decay = np.exp(-5/480.)
        for due in range(5,n,5):
            for j,h in enumerate((5,20,60)):
                t = due-h
                if t<0:
                    continue
                ok = gaps[due+1]==gaps[t]
                qty = ok.sum().clip(1)
                r = np.expm1(cash[due]-cash[t])*coef[j,1]/k-p[a+t,:,j]
                r = (r-(r*ok).sum()/qty).clip(-3,3)*ok
                grad = x[t]*r[:,None]
                hist[:,j] = decay*hist[:,j]+np.stack((grad.sum(0),strength[t,:,j]@grad))/qty
                counts[j] = decay*counts[j]+1.
            states[due:min(due+5,n)] = (hist/(counts[None,:,None]+12)).reshape(2,9)
        yield a,b,x,strength,states


def features(t,x,strength,states):
    basic = x[t]
    all_context,contrast = (states[t,j,None,None,:] for j in range(2))
    values = [basic]
    for interaction in (basic[...,None]*all_context,basic[...,None]*contrast,
                        (basic*strength[t])[...,None]*contrast):
        values.append(interaction.reshape(len(t),50,-1))
    v = np.concatenate(values,axis=-1)
    return v-v.mean(1,keepdims=True)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        relative = p-p.mean(1,keepdims=True)
        label = y-y.mean(1,keepdims=True)
        scale = ((relative*label).sum((0,1))/np.square(relative).sum((0,1))).clip(.3,3)
        residual = label/scale-relative
        gram = rhs = 0.
        for a,b,*state in sessions(raw,grid,p):
            ticks = np.arange(d%5,b-a,5)
            for ts in np.array_split(ticks,max(1,(len(ticks)+95)//96)):
                v = features(ts,*state).reshape(-1,84)
                gram += v.T@v
                rhs += v.T@residual[a+ts].reshape(-1,3)
        np.savez(OUT/f"day_{d:02d}.npz",g=gram/len(p),r=rhs/len(p))


def train(end):
    return fitter.fit(end,OUT,RANGES)


def design(raw,grid,code,p,bank):
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b,*state in sessions(raw,grid,p):
        for t in np.array_split(np.arange(b-a),(b-a+95)//96):
            pred = features(t,*state)@bank.reshape(len(bank),-1)
            z[a+t] = pred.reshape(len(t),50,len(ARMS),3)
    return z


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
