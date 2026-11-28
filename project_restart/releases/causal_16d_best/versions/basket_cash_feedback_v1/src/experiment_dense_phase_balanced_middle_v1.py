"""Compare equal-size fixed-phase and balanced samples to all ticks for a common middle-block rank model."""
import json
import numpy as np
import torch
import experiment_dense_middle_field_committee_v1 as middle

local,onset,rel,REFERENCE=middle.local,middle.onset,middle.rel,middle.REFERENCE
OUT=REFERENCE.parent/"dense_phase_balanced_middle_v1"
ARMS=("one_phase_per_date","balanced_same_size","all_ticks")
WIDTH=400


def labels(raw,p,y,grid,d):
    with np.load(onset.settled.GEO) as geom:
        coef,k,missing=geom["coeff"],float(geom["k"]),geom["row"]
    info=[]
    g=np.zeros((3,3)); r=np.zeros(3)
    for a,b in rel.fields.parent.sessions(grid):
        t=np.arange(32,b-a-60)
        gaps=np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
        valid=gaps[t+61]==gaps[t]
        move=np.log1p(k*(y[a+t+20,:,1].astype(float)-coef[1,0])/coef[1,1])*coef[2,1]/k
        q=p[a+t].astype(float)
        q=(q-(q*valid[...,None]).sum(1,keepdims=True)/valid.sum(1)[:,None,None].clip(1))*valid[...,None]
        move=(move-(move*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1))*valid
        pos=(t-32)%4==d%4
        xx=q[pos][valid[pos]]
        g+=xx.T@xx
        r+=xx.T@move[pos][valid[pos]]
        info.append((a+t,q,move,valid,a))
    coef=np.linalg.solve(g+.015*np.diag(g.diagonal().clip(.001)),r)
    return [(t,move-q@coef,mask,a) for t,q,move,mask,a in info]


def marks(raw,t,start,device):
    cur=middle.ranks(raw,t,device)
    delta=cur-middle.ranks(raw,np.maximum(t-20,start),device)
    return torch.stack((cur,delta,delta*cur.abs(),delta*delta.abs()),dim=-1).reshape(len(t),50,WIDTH)


def samples(t,start,d):
    # Match exactly the historical chronological sample (including phase carried across the two sessions).
    index=np.arange(len(t))
    old=index[(t-start-32)%4==d%4][d%4::4]
    balanced=[]
    for a in np.unique(start):
        ix=index[start==a]
        for j,block in enumerate(np.array_split(ix,np.arange(16,len(ix),16))):
            balanced.append(block[(7*j+3*d)%len(block)])
    # Keep the trial matched in count even at a partial session endpoint.
    balanced=np.asarray(balanced,dtype=int)
    n=min(len(old),len(balanced))
    return old[:n],balanced[:n]


def collect():
    OUT.mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    report=[]
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d)
        info=labels(raw,p,y,grid,d)
        t=np.concatenate([a[0] for a in info])
        start=np.concatenate([np.full(len(a[0]),a[3]) for a in info])
        target=np.concatenate([a[1] for a in info])
        valid=np.concatenate([a[2] for a in info])
        old,balanced=samples(t,start,d)
        sparse=np.stack([np.isin(np.arange(len(t)),sel) for sel in (old,balanced)])
        gram=np.zeros((3,WIDTH+1,WIDTH+1)); counts=np.zeros(3)
        for idx in np.array_split(np.arange(len(t)),max(1,(len(t)+95)//96)):
            x=marks(raw,t[idx],start[idx],device)
            ok=torch.as_tensor(valid[idx],device=device)
            x=x-(x*ok[...,None]).sum(1,keepdim=True)/ok.sum(1)[:,None,None].clamp_min(1)
            z=torch.cat((x,torch.as_tensor(target[idx],device=device,dtype=x.dtype)[...,None]),dim=-1)
            for j in range(3):
                mask=ok if j==2 else ok&torch.as_tensor(sparse[j,idx],device=device)[:,None]
                u=z[mask]
                if len(u):
                    gram[j]+=(u.T@u).cpu().numpy()
                    counts[j]+=len(u)
        with np.load(local.OUT/f"day_{d:02d}.npz") as f:
            historical=f["y"][:,1]
        old_target=target[old][valid[old]]
        np.testing.assert_allclose(old_target,historical[:len(old_target)],atol=2e-6,rtol=1e-5)
        np.save(OUT/f"gram_{d:02d}.npy",gram/counts[:,None,None])
        report.append(dict(day=d+1,rows=counts.astype(int).tolist(),
                           phase_histograms=[np.bincount((t[s]-start[s])%16,minlength=16).tolist() for s in (old,balanced)]))
    (OUT/"phase_counts.json").write_text(json.dumps(report,indent=2))


def solve(rows):
    g=rows.mean(0)
    features=np.arange(WIDTH).reshape(100,4)
    base=features[:,0]; active=features[:,1:].ravel()
    proj=np.eye(WIDTH)[:,active]
    ctrl=g[np.ix_(base,base)]
    proj[base]=-np.linalg.solve(ctrl+.12*np.diag(ctrl.diagonal().clip(.001)),g[np.ix_(base,active)])
    cov=proj.T@g[:-1,:-1]@proj
    sd=np.sqrt(cov.diagonal()).clip(.005)
    ev,u=np.linalg.eigh(cov/np.outer(sd,sd))
    axes=proj@(u/(sd[:,None]*np.sqrt(ev.clip(0)+.8)))
    gradients=rows[:,:-1,-1]@axes
    mu=gradients.mean(0)
    return axes@(mu*mu**2/(mu**2+gradients.var(0,ddof=1)/len(rows)+1e-8))


def train(end):
    rows=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    w=np.stack([solve(rows[:,j]) for j in range(3)],axis=-1)
    np.save(OUT/f"fit_{end}.npy",w)
    return np.array([end])


def design(raw,grid,code,p,bank):
    device="cuda" if torch.cuda.is_available() else "cpu"
    w=torch.as_tensor(np.load(OUT/f"fit_{int(bank[0])}.npy"),device=device)
    start=local.origins(grid)
    z=np.zeros((*p.shape[:2],3,3))
    for t in np.array_split(np.arange(len(p)),max(1,(len(p)+95)//96)):
        v=(marks(raw,t,start[t],device).double()@w).cpu().numpy()
        z[t,:,:,2]=v-v.mean(1,keepdims=True)
    return z


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


def audit():
    result={}
    for end in (8,12):
        w=np.load(OUT/f"fit_{end}.npy")
        rows=np.stack([np.load(OUT/f"gram_{d:02d}.npy")[2] for d in range(end,end+4)])
        corr=[]
        for g in rows:
            a=w.T@g[:-1,-1]
            b=np.diag(w.T@g[:-1,:-1]@w)*g[-1,-1]
            corr.append(a/np.sqrt(b.clip(1e-15)))
        result[str(end+1)]=dict(arms=ARMS,middle_block_daily_ic=np.asarray(corr).tolist(),mean=np.mean(corr,axis=0).tolist())
    (OUT/"dense_middle_skill.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
    audit()
