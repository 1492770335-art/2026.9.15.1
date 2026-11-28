"""Estimate reproducible directed stock-to-stock rank effects on the 21--40 tick return residual."""
import json
import numpy as np
import torch
import experiment_dense_phase_balanced_middle_v1 as phase
import experiment_dense_cluster_relative_rank_motion_v1 as cluster

middle,local,onset,rel,REFERENCE=phase.middle,phase.local,phase.onset,phase.rel,phase.REFERENCE
OUT=REFERENCE.parent/"dense_directed_rank_transfer_v1"
ARMS=("within_price_clusters","replicated_all_edges","directed_asymmetry")


def collect():
    OUT.mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d)
        info=phase.labels(raw,p,y,grid,d)
        g=np.zeros((100,50,50)); r=g.copy(); fields=np.zeros((100,100))
        n=0
        for ticks,target,valid,a in info:
            # Sample every four ticks; cycle the offset every sixteen ticks in the session.
            take=np.flatnonzero(np.arange(len(ticks))%4==(np.arange(len(ticks))//16+d)%4)
            for ix in np.array_split(take,max(1,(len(take)+95)//96)):
                t=ticks[ix]
                x=middle.ranks(raw,t,device)-middle.ranks(raw,np.maximum(a,t-20),device)
                mask=torch.as_tensor(valid[ix],device=device)
                x=(x-(x*mask[...,None]).sum(1,keepdims=True)/mask.sum(1)[:,None,None].clamp_min(1))*mask[...,None]
                z=x.permute(2,0,1).contiguous()
                yy=torch.as_tensor(target[ix],device=device,dtype=z.dtype)
                r+=(z.transpose(1,2)@yy).cpu().numpy()
                g+=(z.transpose(1,2)@z).cpu().numpy()
                flat=x.reshape(-1,100)
                fields+=(flat.T@flat).cpu().numpy()/50
                n+=len(ix)
        np.savez(OUT/f"day_{d:02d}.npz",g=g/n,r=r/n,fields=fields/n)


def prepare(g,r,field):
    sd=np.sqrt(np.diagonal(g,axis1=-2,axis2=-1)).clip(.015)
    norm=g/(sd[:,:,None]*sd[:,None,:])
    raw=r/sd[:,:,None]
    own=np.diagonal(raw,axis1=-2,axis2=-1)
    # Conditional donor covariance removes the receiver's own simultaneous field movement.
    effect=raw-norm*own[:,None,:]
    names=np.arange(50)
    effect[:,names,names]=0
    unit=np.sqrt(field.diagonal()).clip(.015)
    ev,u=np.linalg.eigh(field/np.outer(unit,unit))
    white=(u/np.sqrt(ev.clip(0)+1.))@u.T
    return (white@effect.reshape(100,-1)),white,norm,sd


def transfer(A,B,full,mask,skew=False):
    views=[]
    for cf in (A,B,full):
        z=cf.reshape(100,50,50)
        if skew:
            z=.5*(z-z.transpose(0,2,1))
        views.append((z*mask[None]).reshape(100,-1))
    a,b,mean=views
    # Select only field combinations for which two independent date subsets concur on the stock links.
    cross=a@b.T
    ev,u=np.linalg.eigh(.5*(cross+cross.T))
    use=ev.argsort()[-8:]
    basis=u[:,use]
    pa,pb=basis.T@a,basis.T@b
    agreement=(pa*pb).sum(1)/np.sqrt((pa**2).sum(1)*(pb**2).sum(1)).clip(1e-10)
    strength=np.maximum(0,agreement)**2
    w=(basis*strength)@(basis.T@mean)
    return w,agreement


def train(end):
    rows=[]
    for d in range(4,end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("g","r","fields")))
    g,r,f=[np.stack(c) for c in zip(*rows)]
    a=prepare(g[::2].mean(0),r[::2].mean(0),f[::2].mean(0))[0]
    b=prepare(g[1::2].mean(0),r[1::2].mean(0),f[1::2].mean(0))[0]
    full,white,corr,sd=prepare(g.mean(0),r.mean(0),f.mean(0))
    labels=cluster.clusters()[0]
    allowed=~np.eye(50,dtype=bool)
    weights=[];checks=[]
    for mask,skew in (((labels[:,None]==labels)&allowed,False),(allowed,False),(allowed,True)):
        w,rep=transfer(a,b,full,mask,skew)
        w=(white.T@w).reshape(100,50,50)
        # Each edge operates on the donor innovation conditional on the recipient's rank motion.
        w[:,np.arange(50),np.arange(50)]-=(corr*w).sum(1)
        weights.append(w/sd[:,:,None]);checks.append(rep.tolist())
    np.savez(OUT/f"fit_{end}.npz",w=np.stack(weights,axis=-1))
    (OUT/f"agreement_{end}.json").write_text(json.dumps(checks))
    return np.array([end])


def read(raw,grid,p,bank,output):
    device="cuda" if torch.cuda.is_available() else "cpu"
    with np.load(output/f"fit_{int(bank[0])}.npz") as f:
        w=torch.as_tensor(f["w"].reshape(5000,150),device=device)
    z=np.zeros((*p.shape[:2],3,3))
    start=local.origins(grid)
    for t in np.array_split(np.arange(len(p)),max(1,(len(p)+95)//96)):
        change=middle.ranks(raw,t,device)-middle.ranks(raw,np.maximum(start[t],t-20),device)
        v=(change.transpose(1,2).reshape(len(t),5000).double()@w).reshape(len(t),50,3).cpu().numpy()
        z[t,:,:,2]=v-v.mean(1,keepdims=True)
    return z


def design(raw,grid,code,p,bank):
    return read(raw,grid,p,bank,OUT)


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
