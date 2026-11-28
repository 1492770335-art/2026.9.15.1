"""Calibrate forecasts from the paths of finished intervals, distinguishing reversals from continuations."""
import numpy as np
import experiment_dense_session_skill_memory_v1 as memory

prev,onset,rel,REFERENCE = memory.prev,memory.onset,memory.rel,memory.REFERENCE
OUT = REFERENCE.parent/"dense_path_skill_calibration_v1"
ARMS = ("early_vs_late_move","persistent_vs_reversing","current_trend_match")


def sessions(raw,grid,p):
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    for a,b in rel.fields.parent.sessions(grid):
        x,_ = prev.slope.frame(p[a:b])
        n = b-a
        cash = np.log1p(k*(raw[a:b,:,0].astype(float)-.00136)).cumsum(0)
        gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
        states = np.zeros((n,2,9))
        count = np.zeros(3)
        hist = np.zeros((2,3,3))
        trend = np.zeros_like(x)
        decay = np.exp(-5/480.)
        for j,h in enumerate((5,20,60)):
            ticks = np.arange(h,n)
            first = cash[ticks-h//2]-cash[ticks-h]
            last = cash[ticks]-cash[ticks-h//2]
            valid = gaps[ticks+1]==gaps[ticks-h]
            trend[ticks,:,j] = valid*2*first*last/(first**2+last**2+(k/coef[j,1]*.10)**2)
        for due in range(5,n,5):
            for j,h in enumerate((5,20,60)):
                t = due-h
                if t<0:
                    continue
                ok = gaps[due+1]==gaps[t]
                norm = ok.sum().clip(1)
                earlier = cash[due-h//2]-cash[t]
                later = cash[due]-cash[due-h//2]
                err = np.expm1(earlier+later)*coef[j,1]/k-p[a+t,:,j]
                def center(v):
                    return (v-(v*ok).sum()/norm).clip(-3,3)*ok
                tilt = center((later-(h//2/h)*(earlier+later))*coef[j,1]/k)
                confirmed = center(trend[due,:,j])*center(err)
                hist[:,j] = decay*hist[:,j]+np.stack((tilt,confirmed))@x[t]/norm
                count[j] = count[j]*decay+1
            states[due:min(due+5,n)] = (hist/(count[None,:,None]+12)).reshape(2,9)
        yield a,b,x,trend,states


def features(t,x,trend,states):
    basic = x[t]
    parts = [basic]
    for value,context in ((basic,states[t,0]),(basic,states[t,1]),(basic*trend[t],states[t,1])):
        parts.append((value[...,None]*context[:,None,None]).reshape(len(t),50,-1))
    v = np.concatenate(parts,axis=-1)
    return v-v.mean(1,keepdims=True)


def collect_for(out,reference,read_states,read_features):
    out.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(reference/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        q = p-p.mean(1,keepdims=True)
        label = y-y.mean(1,keepdims=True)
        scale = ((q*label).sum((0,1))/np.square(q).sum((0,1))).clip(.3,3)
        err = label/scale-q
        gram = rhs = 0.
        for a,b,*state in read_states(raw,grid,p):
            ticks = np.arange(d%5,b-a,5)
            for t in np.array_split(ticks,max(1,(len(ticks)+95)//96)):
                z = read_features(t,*state)
                z = z.reshape(-1,z.shape[-1])
                gram += z.T@z
                rhs += z.T@err[a+t].reshape(-1,3)
        np.savez(out/f"day_{d:02d}.npz",g=gram/len(p),r=rhs/len(p))


def collect():
    collect_for(OUT,REFERENCE,sessions,features)


def train(end):
    return prev.fitter.fit(end,OUT,prev.RANGES)


def read(raw,grid,p,bank,read_states,read_features):
    z = np.zeros((*p.shape[:2],bank.shape[1],3))
    for a,b,*state in read_states(raw,grid,p):
        for t in np.array_split(np.arange(b-a),(b-a+95)//96):
            pred = read_features(t,*state)@bank.reshape(len(bank),-1)
            z[a+t] = pred.reshape(len(t),50,bank.shape[1],3)
    return z


def design(raw,grid,code,p,bank):
    return read(raw,grid,p,bank,sessions,features)


def corrected(p,z,params):
    q = p.copy()
    for j in range(3):
        q[...,j] += (rel.relative.features(p,z[...,j])@params["w"][:,j]).astype(q.dtype)
    return q


def predict(raw,grid,p,params):
    return corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
