"""Estimate delayed field revisions from currently observed, asynchronously changing companion fields."""
from functools import lru_cache

import numpy as np

import experiment_dense_conditional_history_bridge_v1 as bridge

onset,rel,init,book,REFERENCE = bridge.onset,bridge.rel,bridge.init,bridge.book,bridge.REFERENCE
OUT = REFERENCE.parent/"dense_hidden_refresh_interpolation_v1"
ARMS = ("artificially_hidden_updates","anticipated_refresh","agreed_update")
FIELDS = np.array([5,10,14,24,39,40,41,42,52,53,54,55])
INPUTS = 11
S = np.arange(50)[None,:,None,None]


def learn_companions():
    gram = np.zeros((83,83))
    for d in range(4):
        raw = onset.settled.load(d,"x")
        t = np.arange(4+d,len(raw),7)
        change = np.arcsinh(raw[t,:,:83].astype(np.float32))-np.arcsinh(raw[t-4,:,:83].astype(np.float32))
        x = change.reshape(-1,83).astype(float)
        gram += x.T@x
    sd = np.sqrt(gram.diagonal().clip(.001))
    corr = np.abs(gram/np.outer(sd,sd))[FIELDS]
    corr[corr>.97] = 0
    np.save(OUT/"companions.npy",np.argsort(corr,axis=1)[:,-6:])
    companions.cache_clear()


@lru_cache(maxsize=1)
def companions():
    return np.load(OUT/"companions.npy")


def sessions(raw,grid):
    for a,b in rel.fields.parent.sessions(grid):
        view = np.arcsinh(raw[a:b,:,:83].astype(np.float32))
        y = view[:,:,FIELDS]
        change = np.vstack((np.ones_like(y[:1],bool),y[1:]!=y[:-1]))
        last = np.maximum.accumulate(np.where(change,np.arange(b-a)[:,None,None],0),axis=0)
        yield a,b,view,y,last


def rows(view,y,t,origin):
    keys = companions()[None,None,:,:]
    idx = t[:,None,None,None]
    current = view[idx,S,keys]
    prev = view[origin[...,None],S,keys]
    fields = np.arange(len(FIELDS))[None,None,:]
    names = np.arange(50)[None,:,None]
    held = y[origin,names,fields]
    age = t[:,None,None]-origin
    mean = y.mean(1)
    market = mean[t,None,:]-mean[origin,fields]
    features = np.concatenate((np.ones((*age.shape,1),np.float32),held[...,None],
                               np.log1p(age[...,None])/3,market[...,None],
                               np.tanh(held[...,None])*market[...,None],current-prev),axis=-1)
    return features,held,age


def collect():
    OUT.mkdir(exist_ok=True)
    learn_companions()
    for day in range(16):
        raw,grid = [onset.settled.load(day,k) for k in ("x","time")]
        grams = np.zeros((2,len(FIELDS),INPUTS+1,INPUTS+1))
        count = np.zeros((2,len(FIELDS)))
        for a,b,v,y,last in sessions(raw,grid):
            total = np.concatenate((np.zeros_like(y[:1]),y.cumsum(0,dtype=float)))
            times = np.arange(18+day%8,b-a-8,8)
            for t in np.array_split(times,(len(times)+63)//64):
                lag = np.array([2,6,16])[(t//8+day)%3]
                artificial = np.broadcast_to((t-lag)[:,None,None],last[t].shape)
                for j,origin in enumerate((artificial,last[t])):
                    x,held,age = rows(v,y,t,origin)
                    target = (y[t] if j==0 else (total[t+9]-total[t+1])/8)-held
                    ok = age>=2
                    z = np.concatenate((x,target[...,None]),axis=-1)*ok[...,None]
                    z = z.transpose(2,0,1,3).reshape(len(FIELDS),-1,INPUTS+1)
                    grams[j] += z.swapaxes(1,2)@z
                    count[j] += ok.sum((0,1))
        np.save(OUT/f"gram_{day:02d}.npy",grams/count.clip(1)[:,:,None,None])


def train(end):
    g = np.mean([np.load(OUT/f"gram_{d:02d}.npy") for d in range(end)],axis=0)
    cov = g[:,:,:-1,:-1]
    diag = np.diagonal(cov,axis1=2,axis2=3).clip(.001)
    return np.linalg.solve(cov+.16*diag[...,None]*np.eye(INPUTS),g[:,:,:-1,-1:])


def design(raw,grid,code,p,bank):
    updates = np.zeros((len(raw),50,len(FIELDS),3),np.float32)
    for a,b,v,y,last in sessions(raw,grid):
        for i in range(0,b-a,64):
            t = np.arange(i,min(b-a,i+64))
            x,_,age = rows(v,y,t,last[t])
            ds = np.einsum("tsfd,jfd->tsfj",x,bank[...,0]).clip(-1,1)*(age[...,None]>=2)
            agree = (ds[...,0]*ds[...,1]>0)*np.sign(ds[...,0])*np.minimum(abs(ds[...,0]),abs(ds[...,1]))
            updates[a+t] = np.concatenate((ds,agree[...,None]),axis=3)
    accum = np.concatenate((np.zeros_like(updates[:1]),updates.cumsum(0,dtype=float)))
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    positions = np.array([init.F.index(int(j)) for j in FIELDS])
    for a,b in rel.fields.parent.sessions(grid):
        all_t = np.arange(a,b)
        x,vol = book.phase.spline.features(raw,all_t)
        for i in range(0,b-a,64):
            sl = slice(i,i+64)
            t = all_t[sl]
            base = init.response(x[sl],vol[sl],t)
            for arm in range(3):
                copy = x[sl].copy()
                dx = updates[t,:,:,arm]
                copy[:,:,FIELDS] += dx
                for j,h in enumerate((5,24,120)):
                    left = (t+1-h).clip(0)
                    avg = (accum[t+1,:,:,arm]-accum[left,:,:,arm])/(t+1-left)[:,None,None]
                    cols = 100+j*len(init.F)+positions
                    copy[:,:,cols] += dx-avg
                delta = init.response(copy,vol[sl],t)-base
                out[t,:,arm] = delta-delta.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
