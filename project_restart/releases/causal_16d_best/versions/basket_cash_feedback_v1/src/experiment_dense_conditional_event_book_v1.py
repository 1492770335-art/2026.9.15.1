"""Predict the book at each price event from its precursor and mark, then separate the fresh surprise."""
import json
import numpy as np
import torch
import experiment_dense_between_moves_v1 as spells

base,event,onset,rel,REFERENCE = spells.base,spells.event,spells.onset,spells.rel,spells.REFERENCE
OUT = REFERENCE.parent/"dense_conditional_event_book_v1"
ARMS = ("precursor","explained_event_update","unexplained_event_update")


def at_events(data,e,missing):
    step = data[...,0].astype(float)-.00136
    obs = ~np.all(data==missing,axis=-1)
    t,s = np.where((abs(step)>.001)&obs)
    prev = np.maximum.accumulate(np.where((abs(step)>.001)&obs,np.arange(len(data))[:,None],-1),axis=0)
    last = prev[(t-1).clip(0),s]
    gaps = np.vstack((np.zeros(50,int),(~obs).cumsum(0)))
    ok = (t>=6)&(last>=0)&(gaps[t+1,s]==gaps[np.minimum(t-6,last).clip(0),s])
    t,s,last = t[ok],s[ok],last[ok]
    lag = np.arange(1,6)
    before = e[t-1,s]
    avg = e[t[:,None]-lag,s[:,None]].mean(1)
    mark = np.tanh(step[t,s])
    duration = np.tanh(np.log1p(t-last)/2)
    rev = np.sign(step[last,s])*np.sign(mark)
    a = np.column_stack((np.ones(len(t)),before,avg,e[last,s],duration,rev))
    b = np.column_stack((mark,abs(mark),mark*duration,mark*rev,mark[:,None]*before,
                         abs(mark[:,None])*before,mark[:,None]*avg))
    return t,s,before,np.column_stack((a,b)),a.shape[1]


def solve(g,n):
    x = g[:n,:n]
    return np.linalg.solve(x+.08*np.diag(x.diagonal().clip(1e-6)),g[:n,-base.DIM:])


def dictionary():
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(onset.settled.GEO) as f:
        missing = f["row"]
    train = 0.
    audit = []
    for d in range(8):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        errors = np.zeros(4)
        for a,b in rel.fields.parent.sessions(grid):
            data = raw[a:b]
            e = base.innovations(data,device)
            t,s,before,x,cut = at_events(data,e,missing)
            delta = e[t,s]-before
            if d<4:
                z = np.column_stack((x,delta))
                train += z.T@z
            else:
                errors += [np.square(delta).sum(),np.square(delta-x[:,:cut]@prior).sum(),
                           np.square(delta-x@full).sum(),delta.size]
        if d==3:
            prior,full = solve(train,cut),solve(train,x.shape[1])
            np.savez(OUT/"book.npz",prior=prior,full=full)
        if d>=4:
            audit.append((errors[:3]/errors[3]).tolist())
    (OUT/"update_mse.json").write_text(json.dumps(dict(models=["copy","pre_event","including_mark"],daily=audit),indent=2))


def states(raw,grid,p):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    with np.load(OUT/"book.npz") as f:
        weights = f["full"]
    for a,b in rel.fields.parent.sessions(grid):
        data = raw[a:b]
        e = base.innovations(data,device)
        t,s,before,x,_ = at_events(data,e,missing)
        explained = x@weights
        marks = np.zeros((*e.shape[:2],3*base.DIM),np.float32)
        marks[t,s] = np.column_stack((before,explained,e[t,s]-before-explained))
        count,ids,_ = event.clock(data)
        gaps = np.vstack((np.zeros(50,int),np.all(data==missing,axis=-1).cumsum(0)))
        yield a,b,(marks,count,ids),gaps,coef,k


def features(t,state,p):
    marks,count,ids = state
    stock = np.arange(50)[None,:,None]
    num = count[t,:,None]-np.arange(event.TAPS)
    values = marks[ids[num.clip(0),stock],stock]*(num>0)[...,None]
    bands = np.stack([values[:,:,lo:hi].mean(2) for lo,hi in spells.dual.BANDS],axis=2)
    return np.concatenate((p[t],*(v.reshape(len(t),50,-1) for v in np.split(bands,3,axis=-1))),axis=-1)


def collect():
    dictionary()
    spells.collect_at(OUT,REFERENCE,states,features)


def train(end):
    return spells.dual.train(end,OUT)


def design(raw,grid,code,p,bank):
    return spells.read(raw,grid,p,bank,OUT,states,features)


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    q[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
