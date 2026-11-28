"""Predict the midpoint by integrating the time, size, and state of successive observable quote changes."""
import numpy as np
import experiment_dense_subtick_observation_filter_v1 as filt

onset, rel, book, REFERENCE = filt.onset, filt.rel, filt.book, filt.REFERENCE
OUT = REFERENCE.parent / "dense_quote_renewal_microprice_v1"
ARMS = ("first_quote_change", "subsequent_quote_changes", "joint_wait_and_state")
BID = [41, 43, 50, 52, 54, 56, 79]
ASK = [42, 44, 51, 53, 55, 57, 80]
H = (5, 20, 60)
N = 5*2*2*3*2
DURATIONS = np.array([1,2,3,4,5,6,8,10,13,16,20,25,32,40,50,60])
L = len(DURATIONS)


def depth(raw):
    b = np.arcsinh(raw[:, :, BID].astype(np.float32))
    a = np.arcsinh(raw[:, :, ASK].astype(np.float32))
    return (b-a).clip(-5,5)


def book_axis():
    g = np.zeros((8,8))
    for day in range(4):
        raw, grid = [onset.settled.load(day,key) for key in ("x","time")]
        step, valid = filt.increments(raw)
        for a,b in rel.fields.parent.sessions(grid):
            t = np.arange(a+day%3,b-3,3)
            price = step[a:b].cumsum(0)
            future = price[t-a+3]-price[t-a]
            x = depth(raw[t])
            keep = valid[t] & valid[t+3]
            z = np.concatenate((x,future[:,:,None]),axis=2)[keep]
            g += z.astype(float).T @ z
    w = filt.proxy.previous.base.own.solve(g[:-1,:-1],g[:-1,-1],.6)
    return w / np.sqrt((w @ g[:-1,:-1] @ w)/sum(
        len(onset.settled.load(i,"time"))*50/3 for i in range(4)))


def state(raw, step, valid, axis):
    n = len(step)
    t = np.arange(n)
    spread_field = raw[:,:,39].astype(float)
    # The narrowest spread is encoded by the minimum observed on the four calibration dates.
    wide = spread_field > axis[-1]
    pressure = depth(raw) @ axis[:-1]
    imbalance = np.digitize(pressure, [-1.05,-.30,.30,1.05])
    moved = abs(step) > .003
    origin = np.maximum.accumulate(np.where(moved|~valid,t[:,None],0),axis=0)
    sign = step[origin,np.arange(50)] > 0
    age = t[:,None]-origin
    price = step.cumsum(0)
    move = price - price[(t-5).clip(0)]
    peer = (move.sum(1,keepdims=True)-move)/49
    market = np.digitize(peer,[-.045,.045])
    ids = ((((imbalance*2+wide)*2+sign)*3+market)*2+(age>=3)).astype(int)
    return ids, moved


def collect():
    OUT.mkdir(exist_ok=True)
    low = np.median([onset.settled.load(d,"x")[:,:,39].min() for d in range(4)])
    axis = np.r_[book_axis(),low+.001]
    np.save(OUT/"axis.npy",axis)
    for day in range(16):
        raw, grid = [onset.settled.load(day,key) for key in ("x","time")]
        step, valid = filt.increments(raw)
        counts = np.zeros((2,N))
        transitions = np.zeros((2,L,N,N),np.float32)
        reward = np.zeros((2,L,N))
        for a,b in rel.fields.parent.sessions(grid):
            delta, obs = step[a:b], valid[a:b]
            ids, moved = state(raw[a:b],delta,obs,axis)
            tick = np.arange(b-a)
            first = np.minimum.accumulate(np.where(moved,tick[:,None],b-a)[::-1],axis=0)[::-1]
            missing = (~obs).cumsum(0)
            for kind in (0,1):
                # At an arbitrary issue time and immediately after a move the remaining waits differ.
                t,s = np.where((obs if kind==0 else obs&moved)[:-60])
                good = ((t+s+day)%2==0) & (missing[t+60,s]==missing[t,s])
                t,s = t[good],s[good]
                src = ids[t,s]
                counts[kind] += np.bincount(src,minlength=N)
                stop = first[t+1,s]
                ok = stop<=t+60
                stop,t,s,src = (u[ok] for u in (stop,t,s,src))
                delay = np.searchsorted(DURATIONS,stop-t)
                key = delay*N+src
                jump = delta[stop,s]
                reward[kind] += np.bincount(key,weights=jump,minlength=L*N).reshape(L,N)
                transitions[kind] += np.bincount(key*N+ids[stop,s],minlength=L*N*N).reshape(L,N,N)
        np.savez_compressed(OUT/f"day_{day:02d}.npz",counts=counts,transitions=transitions,reward=reward)


def train(end):
    axis = np.load(OUT/"axis.npy")
    counts = np.zeros((2,N))
    trans = np.zeros((2,L,N,N))
    reward = np.zeros((2,L,N))
    for d in range(end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            counts += f["counts"]
            trans += f["transitions"]
            reward += f["reward"]
    for kind in (0,1):
        # Smooth across peer context and age for the same spread, side and imbalance.
        group = np.arange(N)//6
        ct = counts[kind].reshape(-1,6).sum(1)[group]
        for arr in (trans[kind], reward[kind]):
            rhs = arr.reshape(L,N//6,6,*arr.shape[2:]).sum(2)[:,group]
            ex = (1,)*(arr.ndim-2)
            den = ct.reshape(1,N,*ex).clip(1)
            arr += 100*rhs/den
            arr /= (counts[kind]+100).reshape(1,N,*ex)
    def values(kernel):
        v = np.zeros((61,N))
        initial = np.zeros_like(v)
        immediate = np.zeros_like(v)
        for h in range(1,61):
            ok = DURATIONS<=h
            for kind, target in ((0,initial),(1,v)):
                r = reward[kind,ok].sum(0)
                continuation = np.einsum("lij,lj->i",kernel[kind,ok],v[h-DURATIONS[ok]])
                target[h] = r + continuation
                if kind==0:
                    immediate[h] = r
        return initial[list(H)].T,immediate[list(H)].T
    full,first = values(trans)
    # Negative-control dynamics retain both marginals but disconnect the waiting time from the new state.
    marginal = trans.sum(1)
    marginal /= marginal.sum(2,keepdims=True).clip(1e-9)
    independent,_ = values(trans.sum(3,keepdims=True)*marginal[:,None])
    table = np.stack((first,full-first,full-independent),axis=1)
    return np.r_[axis,table.ravel()]


def design(raw,grid,code,p,bank):
    axis, table = bank[:8],bank[8:].reshape(N,3,3)
    step,valid = filt.increments(raw)
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b in rel.fields.parent.sessions(grid):
        ids,_ = state(raw[a:b],step[a:b],valid[a:b],axis)
        out[a:b] = table[ids]*valid[a:b,:,None,None]
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
