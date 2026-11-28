"""Predict horizon errors from the relative onset of pressure in five book channels."""
import numpy as np
import experiment_dense_near_matched_spreads_v1 as book
import experiment_dense_settled_error_gru_v1 as settled
from experiment_dense_rank_risk_calibration_v1 import main as rolling
from prepare_dense_live_shared_rank_reference_v1 import OUT as REFERENCE

rel = settled.rel
OUT = REFERENCE.parent / "dense_pressure_onset_order_v1"
ARMS = ("duration_given_strength", "relative_onset_order", "price_between_onsets")
LEFT, RIGHT = np.triu_indices(5, 1)


def sessions(raw,grid,code):
    src = book.sources(raw,grid,code)
    for a,b in rel.fields.parent.sessions(grid):
        pressure = sum(src[a:b,:,j:j+5] for j in (0,6,12)).astype(float)
        pressure = book.smooth(pressure,5)
        t = np.arange(b-a)[:,None,None]
        last_up = np.maximum.accumulate(np.where(pressure>.15,t,-1),axis=0)
        last_down = np.maximum.accumulate(np.where(pressure<-.15,t,-1),axis=0)
        direction = np.sign(last_up-last_down)
        changed = np.concatenate((np.ones_like(direction[:1],bool),direction[1:]!=direction[:-1]))
        onset = np.maximum.accumulate(np.where(changed,t,0),axis=0)
        move = raw[a:b,:,0].astype(float)-.00136
        level = move.cumsum(0)
        yield a,b,pressure,direction,onset,level


def features(t,pressure,direction,onset,level,p):
    v = pressure[t]
    sign = direction[t]
    start = onset[t]
    age = t[:,None,None]-start
    strength = v*sign
    weight = np.tanh(2*strength.clip(0))*sign
    base = np.concatenate((p[t]-p[t].mean(1,keepdims=True),v,v*abs(v),sign,
                           weight[:,:,LEFT]*abs(weight[:,:,RIGHT]),weight[:,:,RIGHT]*abs(weight[:,:,LEFT])),axis=2)
    dur = np.log1p(age)
    duration = np.concatenate((weight*dur/(1+dur), sign*np.exp(-age/8), weight*np.exp(-age/32)),axis=2)
    both = np.sqrt(np.abs(weight[:,:,LEFT]*weight[:,:,RIGHT]))
    lead = np.tanh((dur[:,:,LEFT]-dur[:,:,RIGHT])*.7)
    same = sign[:,:,LEFT]*sign[:,:,RIGHT]>0
    marked = both*sign[:,:,LEFT]
    order = np.concatenate((marked*lead*same,marked*lead*(~same),marked*abs(lead)*same),axis=2)
    at_entry = level[start,np.arange(50)[None,:,None]]
    between = np.tanh((at_entry[:,:,RIGHT]-at_entry[:,:,LEFT])/np.sqrt(1+np.abs(age[:,:,LEFT]-age[:,:,RIGHT])))
    reaction = np.concatenate((both*between*same,both*between*(~same),both*between*lead),axis=2)
    x = np.concatenate((base,duration,order,reaction),axis=2)
    return x-x.mean(1,keepdims=True)


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4,16):
        raw,grid,y = [settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6
             else settled.head.ranks.band.pairs.net.future.forecast(d))
        q = p-p.mean(1,keepdims=True,dtype=float)
        target = y-y.mean(1,keepdims=True,dtype=float)
        slope = ((q*target).sum((0,1))/(q*q).sum((0,1))).clip(.2,4)
        error = target/slope-q
        gram = cross = 0
        n = 0
        for a,b,*state in sessions(raw,grid,code):
            t = np.arange(24+d%5,b-a-60,5)
            for ix in np.array_split(t,5):
                z = features(ix,*state,p[a:b])
                x = z.reshape(-1,z.shape[-1])
                gram += x.T@x
                cross += x.T@error[a+ix].reshape(-1,3)
                n += len(x)
        np.savez(OUT/f"day_{d:02d}.npz",g=gram/n,r=cross/n)


def fit(end, directory, ranges):
    rows = []
    for d in range(4,end):
        with np.load(directory/f"day_{d:02d}.npz") as f:
            rows.append((f["g"],f["r"]))
    gs,rs = [np.stack(col) for col in zip(*rows)]
    gram = gs.mean(0)
    arms = []
    for prior,stop in ranges:
        v = np.eye(len(gram))[:,prior:stop]
        before = gram[:prior,:prior]
        v[:prior] = -np.linalg.solve(before+.08*np.diag(before.diagonal().clip(.0001)),gram[:prior,prior:stop])
        g = v.T@gram@v
        unit = np.sqrt(g.diagonal()).clip(.005)
        eig,rot = np.linalg.eigh(g/np.outer(unit,unit))
        mode = v@(rot/(unit[:,None]*np.sqrt(eig.clip(0)+.6)))
        r = mode.T@rs
        mean = r.mean(0)
        noise = r.var(0,ddof=1)/len(r)
        w = mode@(mean*mean**2/(mean**2+noise+1e-8))
        arms.append(w/np.sqrt(np.sum(w*(gram@w),axis=0)).clip(1e-5))
    return np.stack(arms,axis=1)


def train(end):
    return fit(end, OUT, ((38,53),(53,83),(83,113)))


def design(raw,grid,code,p,bank):
    z = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b,*state in sessions(raw,grid,code):
        for t in np.array_split(np.arange(b-a),max(1,(b-a+127)//128)):
            x = features(t,*state,p[a:b])
            z[a+t] = (x@bank.reshape(len(bank),-1)).reshape(len(t),50,len(ARMS),3)
    return z


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[:,:,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    rolling(OUT,ARMS,train,design,REFERENCE)
