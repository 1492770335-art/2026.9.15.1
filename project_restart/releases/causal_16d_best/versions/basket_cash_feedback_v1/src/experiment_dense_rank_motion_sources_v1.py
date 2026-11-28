"""Attribute rank changes of all 100 fields to motion of the stock itself or of the other 49 names."""
import json
import numpy as np
import torch

import experiment_dense_joint_block_paths_v1 as paths

onset, rel, REFERENCE = paths.onset,paths.rel,paths.REFERENCE
OUT = REFERENCE.parent/"dense_rank_motion_sources_v1"
ARMS = ("current_rank","combined_rank_change","source_of_rank_change")
GROUPS = ((0,1),(2,3,4),(5,6,7))


def features(raw,now,old,device):
    def tensor(t):
        return torch.as_tensor(np.asarray(raw[t]),device=device,dtype=torch.float32).transpose(1,2).contiguous()
    a,b = tensor(now),tensor(old)
    a_sorted,b_sorted = a.sort(-1).values,b.sort(-1).values
    def rank(query,data,sorted_data):
        left = torch.searchsorted(sorted_data,query)
        right = torch.searchsorted(sorted_data,query,right=True)
        # Exclude the receiving stock even when its own previous value differs.
        return (left+right-50-(query-data).sign())/49
    latest,former = rank(a,a,a_sorted),rank(b,b,b_sorted)
    cross_a,cross_b = rank(a,b,b_sorted),rank(b,a,a_sorted)
    change = latest-former
    own = .5*(cross_a-former+latest-cross_b)
    split = 2*own-change
    values = torch.stack((latest,latest*abs(latest),change,change*abs(latest),change*abs(change),
                          split,split*abs(latest),split*abs(change)),dim=-1).transpose(1,2)
    return values.cpu().numpy()


def origins(grid):
    return np.concatenate([np.full(b-a,a,dtype=int) for a,b in rel.fields.parent.sessions(grid)])


def collect():
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in range(4,16):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        ts = np.concatenate([a+np.arange(32+d%4,b-a-60,4) for a,b in rel.fields.parent.sessions(grid)])
        # The rank features are expensive; training samples every 16 ticks, validation evaluates every row.
        select = np.arange(d%4,len(ts),4)
        with np.load(paths.OUT/f"day_{d:02d}.npz") as f:
            y,ok = f["target"][select],f["valid"][select]
        t = ts[select]
        start = origins(grid)
        chunks = []
        for ix in np.array_split(t,max(1,(len(t)+31)//32)):
            chunks.append(features(raw,ix,np.maximum(ix-20,start[ix]),device))
        x = np.concatenate(chunks)
        x -= (x*ok[:,:,None,None]).sum(1,keepdims=True)/ok.sum(1)[:,None,None,None].clip(1)
        x=x[ok].astype(np.float16)
        y=y[ok]
        # Store per-field covariance and cross-covariances for supervised projection.
        basis = np.ascontiguousarray(x.transpose(1,0,2),dtype=float)
        gram = basis.transpose(0,2,1)@basis/len(y)
        cross = basis.transpose(0,2,1)@y.astype(float)/len(y)
        np.savez(OUT/f"day_{d:02d}.npz",x=x,y=y,gram=gram,cross=cross)


def surface(g,r,cols):
    # Remove the local linear influence of rank and total rank change before pricing the source contrast.
    n = g.shape[1]
    ids = list(cols)
    proj = np.broadcast_to(np.eye(n)[:,ids],(100,n,len(ids))).copy()
    off = ids[0]
    if off:
        ctrl = g[:,:off,:off]
        reg = .08*np.eye(off)*(1e-4+np.diagonal(ctrl,axis1=1,axis2=2)[:,None,:])
        proj[:,:off] = -np.linalg.solve(ctrl+reg,g[:,:off,ids])
    gram = proj.transpose(0,2,1)@g@proj
    scale = np.diagonal(gram,axis1=1,axis2=2).clip(.001)
    w = np.linalg.solve(gram+.45*np.eye(len(ids))*scale[:,None,:],proj.transpose(0,2,1)@r)
    return proj@w


def train(end):
    stats = []
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            stats.append((f["gram"],f["cross"]))
    gs,cs = [np.stack(s) for s in zip(*stats)]
    def project(g,r):
        return np.stack([surface(g,r,group) for group in GROUPS],axis=2)
    prior = [project(gs[1-i::2].mean(0),cs[1-i::2].mean(0)) for i in (0,1)]
    # Combine correlated field forecasts using only their out-of-date-fit payoffs.
    grams=[]; cross=[]
    for i,d in enumerate(range(4,end)):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            x,y = f["x"].astype(np.float32),f["y"]
        u = np.einsum("nfc,fcah->ahnf",x,prior[i%2].astype(np.float32),optimize=True)
        g = u.transpose(0,1,3,2)@u/len(x)
        rhs = np.einsum("ahnf,nh->ahf",u,y,optimize=True)/len(y)
        grams.append(g); cross.append(rhs)
    mean = np.mean(cross,axis=0)
    uncertainty = np.var(cross,axis=0,ddof=1)/len(cross)
    g = np.mean(grams,axis=0).astype(float)
    diag = np.diagonal(g,axis1=-2,axis2=-1).clip(1e-7)
    weight = np.linalg.solve(g+1.5*np.eye(100)*diag[...,None,:],
                             (mean*mean**2/(mean**2+uncertainty+1e-12))[...,None])[...,0]
    w = project(gs.mean(0),cs.mean(0))*weight.transpose(2,0,1)[:,None,:,:]
    np.savez(OUT/f"fit_{end}.npz",w=w)
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        w = f["w"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = origins(grid)
    rows = []
    for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+47)//48)):
        x = features(raw,t,np.maximum(start[t],t-20),device)
        z = (x.reshape(len(t),50,-1).astype(float)@w.reshape(-1,9)).reshape(len(t),50,3,3)
        z -= z.mean(1,keepdims=True)
        rows.append(np.stack((z[...,0],z[...,0],z.sum(-1)),axis=-1))
    return np.concatenate(rows)


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
