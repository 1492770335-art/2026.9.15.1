"""Compare order-book updates on price plateaus with equal-count windows ending before the plateau."""
import numpy as np
import torch

import experiment_dense_dual_clock_response_v1 as dual
from prepare_dense_dual_clock_response_reference_v1 import OUT as REFERENCE

base,event,onset,rel = dual.base,dual.event,dual.onset,dual.rel
OUT = REFERENCE.parent/"dense_between_moves_v1"
ARMS = ("earlier_matched_windows","within_price_plateaus","late_vs_early_updates")
# The first state is the open plateau, followed by four cohorts of completed ones.
BANDS = ((0,1),(1,2),(2,4),(4,8),(8,16))


def windows(raw,e):
    t = np.arange(len(raw))[:,None]
    stocks = np.arange(50)
    move = abs(raw[...,0].astype(float)-.00136)>.001
    update = np.vstack((np.zeros((1,50),bool),np.any(raw[1:,:,1:]!=raw[:-1,:,1:],axis=-1)))
    with np.load(onset.settled.GEO) as f:
        present = ~np.all(raw==f["row"],axis=-1)
    update &= present & np.vstack((np.zeros((1,50),bool),present[:-1]))
    # A missing observation invalidates the active spell until a new price change arrives.
    origin = np.maximum.accumulate(np.where(move,t,0),axis=0)
    invalid = np.maximum.accumulate(np.where(~present,t,-1),axis=0)
    count = update.cumsum(0)
    which = np.zeros((len(raw)+1,50),np.int32)
    tick,asset = np.where(update)
    which[count[tick,asset],asset] = tick
    sums = np.concatenate((np.zeros_like(e[:1],dtype=float),np.cumsum(e*update[...,None],axis=0,dtype=float)))
    start = count[origin,stocks]
    n = count-start
    good = (n>0)&(start>=n)&(origin>invalid)&(origin>0)
    def total(last):
        # Zero updates correspond to the empty prefix even if frame zero is nonzero.
        at = which[last,stocks]
        return sums[np.where(last>0,at+1,0),stocks]
    end_value = total(count)
    start_value = total(start)
    current = (end_value-start_value)/n.clip(1)[...,None]
    prev = (start_value-total((start-n).clip(0)))/n.clip(1)[...,None]
    midpoint = start+n//2
    middle = total(midpoint)
    early = (middle-start_value)/(midpoint-start).clip(1)[...,None]
    late = (end_value-middle)/(count-midpoint).clip(1)[...,None]
    # Insist that matched history also starts after the last missing observation.
    good &= (which[(start-n+1).clip(0,len(raw)),stocks]>invalid)
    return tuple((part*mask[...,None]).astype(np.float32) for part,mask in
                 ((prev,good),(current-prev,good),(late-early,good&(n>=2))))


def states(raw,grid,p):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    for a,b in rel.fields.parent.sessions(grid):
        section = raw[a:b]
        values = windows(section,base.innovations(section,device))
        count,ids,_ = event.clock(section)
        gaps = np.vstack((np.zeros(50,int),np.all(section==missing,axis=-1).cumsum(0)))
        yield a,b,(values,count,ids),gaps,coef,k


def features(t,state,p):
    values,count,ids = state
    stocks = np.arange(50)[None,:,None]
    num = count[t,:,None]-np.arange(15)
    last = ids[num.clip(0),stocks]-1
    result = [p[t]]
    for v in values:
        ended = v[last.clip(0),stocks]*((num>0)&(last>=0))[...,None]
        all_spells = np.concatenate((v[t,:,:,None].transpose(0,1,3,2),ended),axis=2)
        result.extend(all_spells[:,:,lo:hi].mean(2) for lo,hi in BANDS)
    return np.concatenate(result,axis=-1)


def collect_at(directory,reference,read_states,read_features):
    directory.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,key) for key in ("x","time","y")]
        p = (np.load(reference/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        gram = 0.
        n = 0
        for a,b,state,gaps,coef,k in read_states(raw,grid,p):
            ticks = np.arange(80,b-a-60)
            ix = np.arange(len(ticks))
            sample = ticks[ix%8==(ix//24+d)%8]
            for start in range(0,len(sample),48):
                t = sample[start:start+48]
                valid = gaps[t+61]==gaps[t]
                x = read_features(t,state,p[a:b])
                z = np.concatenate((x,base.target(y[a:b],t,coef,k)[...,1:]),axis=-1)
                z -= (z*valid[...,None]).sum(1,keepdims=True)/valid.sum(1)[:,None,None].clip(1)
                u = torch.as_tensor(z[valid],device=device,dtype=torch.float32)
                gram += (u.T@u).cpu().numpy().astype(float)
                n += len(u)
        np.save(directory/f"gram_{d:02d}.npy",gram/n)


def collect():
    collect_at(OUT,REFERENCE,states,features)


def train(end):
    return dual.train(end,OUT)


def read(raw,grid,p,bank,directory,read_states,read_features):
    with np.load(directory/f"fit_{int(bank[0])}.npz") as f:
        w = f["bank"].sum(-1)
    z = np.zeros((*p.shape[:2],w.shape[-1],3))
    for a,b,state,*_ in read_states(raw,grid,p):
        for start in range(0,b-a,48):
            t = np.arange(start,min(start+48,b-a))
            vals = read_features(t,state,p[a:b])@w
            z[a+t,:,:,2] = vals-vals.mean(1,keepdims=True)
    return z


def design(raw,grid,code,p,bank):
    return read(raw,grid,p,bank,OUT,states,features)


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    q[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
