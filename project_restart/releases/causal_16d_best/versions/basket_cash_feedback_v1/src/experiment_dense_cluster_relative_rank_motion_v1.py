"""Predict the middle block using relative field-rank changes within persistent co-moving stock groups."""
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import experiment_dense_middle_field_committee_v1 as middle

local,onset,rel,REFERENCE=middle.local,middle.onset,middle.rel,middle.REFERENCE
OUT=REFERENCE.parent/"dense_cluster_relative_rank_motion_v1"
ARMS=("within_group_turnover","other_group_members","group_identity_increment")


def correlation(d):
    raw,grid=[onset.settled.load(d,k) for k in ("x","time")]
    parts=[]
    for a,b in rel.fields.parent.sessions(grid):
        r=raw[a:b,:,0].astype(float)-.00136
        path=np.vstack((np.zeros(50),r.cumsum(0)))
        t=np.arange(20,b-a,10)
        v=path[t]-path[t-20]
        v-=v.mean(1,keepdims=True)
        parts.append(v)
    return np.corrcoef(np.concatenate(parts).T)


def clusters():
    corr=np.mean([correlation(d) for d in range(4)],axis=0)
    vals,vecs=np.linalg.eigh(corr)
    x=vecs[:,-7:]*np.sqrt(vals[-7:].clip(0))
    x/=np.linalg.norm(x,axis=1,keepdims=True).clip(.001)
    centers=[x[0]]
    for _ in range(4):
        dist=np.square(x[:,None]-centers).sum(-1).min(1)
        centers.append(x[dist.argmax()])
    centers=np.stack(centers)
    for _ in range(15):
        cost=np.square(x[:,None]-centers).sum(-1)
        rows,cols=linear_sum_assignment(cost.repeat(10,axis=1))
        labels=np.empty(50,int)
        labels[rows]=cols//10
        centers=np.stack([x[labels==k].mean(0) for k in range(5)])
    return np.stack((labels,labels[np.random.default_rng(2341).permutation(50)]))


def field_basis(raw,t,old,groups,device):
    a,b=[torch.as_tensor(np.asarray(raw[ix]),device=device).float().transpose(1,2).contiguous() for ix in (t,old)]
    def score(v):
        sort=v.sort(-1).values
        return (torch.searchsorted(sort,v)+torch.searchsorted(sort,v,right=True)-v.shape[-1])/(v.shape[-1]-1)
    full,prev=score(a),score(b)
    change=full-prev
    result=[full,change]
    peers=None
    for j,labels in enumerate(groups):
        relative=torch.zeros_like(change)
        support=torch.zeros_like(change)
        for k in range(5):
            idx=torch.as_tensor(np.flatnonzero(labels==k),device=device)
            relative[:,:,idx]=score(a.index_select(2,idx))-score(b.index_select(2,idx))-change.index_select(2,idx)
            member=change.index_select(2,idx)
            support[:,:,idx]=(member.sum(-1,keepdim=True)-member)/9
        result.extend((relative,relative*full.abs()))
        if j==0:
            peers=support
    result.extend((peers,peers*full.abs()))
    return torch.stack(result,dim=-1).transpose(1,2)


def collect():
    OUT.mkdir(exist_ok=True)
    groups=clusters()
    np.save(OUT/"groups.npy",groups)
    device="cuda" if torch.cuda.is_available() else "cpu"
    report=[]
    for d in range(4,16):
        raw,grid=[onset.settled.load(d,k) for k in ("x","time")]
        t=np.concatenate([a+np.arange(32+d%4,b-a-60,4) for a,b in rel.fields.parent.sessions(grid)])
        ix=np.arange(d%4,len(t),4)
        with np.load(local.paths.OUT/f"day_{d:02d}.npz") as f:
            ys=f["target"][ix,:,1]
            valid=f["valid"][ix]
        t=t[ix]
        start=local.origins(grid)
        gram=np.zeros((801,801))
        for sel in np.array_split(np.arange(len(t)),max(1,(len(t)+31)//32)):
            ti=t[sel]
            x=field_basis(raw,ti,np.maximum(ti-20,start[ti]),groups,device).reshape(len(ti),50,800).cpu().numpy()
            mask=valid[sel]
            x=x-(x*mask[...,None]).sum(1,keepdims=True)/mask.sum(1)[:,None,None].clip(1)
            v=np.column_stack((x[mask],ys[sel][mask]))
            gram+=v.T@v
        np.save(OUT/f"gram_{d:02d}.npy",gram/valid.sum())
        c=correlation(d)
        report.append([float(c[(g[:,None]==g)&~np.eye(50,dtype=bool)].mean()) for g in groups])
    (OUT/"group_audit.json").write_text(json.dumps(dict(fit_days=[1,2,3,4],groups=groups[0].tolist(),
                                                        subsequent_daily_true_and_shuffled=report),indent=2))


def train(end):
    rows=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    g=rows.mean(0)
    idx=np.arange(800).reshape(100,8)
    weights=[]
    for active,controls in (((2,3),(0,1)),((6,7),(0,1)),((2,3),(0,1,4,5))):
        target=idx[:,active].ravel()
        base=idx[:,controls].ravel()
        proj=np.eye(800)[:,target]
        ctrl=g[np.ix_(base,base)]
        proj[base]=-np.linalg.solve(ctrl+.35*np.diag(ctrl.diagonal().clip(.0005)),g[np.ix_(base,target)])
        gram=proj.T@g[:-1,:-1]@proj
        sd=np.sqrt(gram.diagonal()).clip(.004)
        ev,u=np.linalg.eigh(gram/np.outer(sd,sd))
        axes=proj@(u/(sd[:,None]*np.sqrt(ev.clip(0)+1.5)))
        grad=rows[:,:-1,-1]@axes
        mu=grad.mean(0)
        weights.append(axes@(mu*mu**2/(mu**2+grad.var(0,ddof=1)/len(rows)+1e-8)))
    np.savez(OUT/f"fit_{end}.npz",w=np.stack(weights,axis=-1),groups=np.load(OUT/"groups.npy"))
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        w,groups=f["w"],f["groups"]
    start=local.origins(grid)
    device="cuda" if torch.cuda.is_available() else "cpu"
    w=torch.as_tensor(w,device=device)
    z=np.zeros((*p.shape[:2],3,3))
    for t in np.array_split(np.arange(len(p)),max(1,(len(p)+63)//64)):
        current=field_basis(raw,t,np.maximum(t-20,start[t]),groups,device)
        v=(current.reshape(len(t),50,-1).double()@w).cpu().numpy()
        z[t,:,:,2]=v-v.mean(1,keepdims=True)
    return z


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
