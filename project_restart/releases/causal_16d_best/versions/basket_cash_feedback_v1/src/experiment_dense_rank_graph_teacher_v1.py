"""Identify rank-transfer directions from future short forecasts and price them on separate dates' realized payoffs."""
import numpy as np
import torch
import experiment_dense_directed_rank_transfer_v1 as graph

phase,middle,local,onset,rel,REFERENCE=graph.phase,graph.middle,graph.local,graph.onset,graph.rel,graph.REFERENCE
OUT=REFERENCE.parent/"dense_rank_graph_teacher_v1"
ARMS=("middle_teacher_geometry","short_revision_geometry","replicated_teacher_and_payoff")


def collect():
    OUT.mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d)
        info=phase.labels(raw,p,y,grid,d)
        blocks=[]
        for ticks,target,valid,a in info:
            take=np.flatnonzero(np.arange(len(ticks))%4==(np.arange(len(ticks))//16+d)%4)
            t=ticks[take]
            xx=p[t].astype(float)
            future=np.stack((p[t+20,:,1],sum(p[t+lag,:,0] for lag in (20,25,30,35))),axis=-1).astype(float)
            mask=valid[take]
            def center(x):
                return mask[...,None]*(x-(x*mask[...,None]).sum(1,keepdims=True)/mask.sum(1)[:,None,None].clip(1))
            blocks.append((t,a,center(xx),center(future),mask))
        gram=sum(x[mask].T@x[mask] for t,a,x,f,mask in blocks)
        rhs=sum(x[mask].T@f[mask] for t,a,x,f,mask in blocks)
        co=np.linalg.solve(gram+.015*np.diag(gram.diagonal().clip(.001)),rhs)
        moments=np.zeros((2,100,50,50)); n=0
        for t,a,x,f,mask in blocks:
            future=f-x@co
            for ix in np.array_split(np.arange(len(t)),max(1,(len(t)+95)//96)):
                ti=t[ix]
                z=middle.ranks(raw,ti,device)-middle.ranks(raw,np.maximum(ti-20,a),device)
                ok=torch.as_tensor(mask[ix],device=device)
                z=(z-(z*ok[...,None]).sum(1,keepdims=True)/ok.sum(1)[:,None,None].clamp_min(1))*ok[...,None]
                z=z.permute(2,1,0).contiguous()
                for j in range(2):
                    teacher=torch.as_tensor(future[ix,:,j],device=device,dtype=z.dtype)
                    moments[j]+=(z@teacher).cpu().numpy()
                n+=len(ix)
        np.save(OUT/f"teacher_{d:02d}.npy",moments/n)


def train(end):
    rows=[];teachers=[]
    for d in range(4,end):
        with np.load(graph.OUT/f"day_{d:02d}.npz") as f:
            rows.append(tuple(f[k] for k in ("g","r","fields")))
        teachers.append(np.load(OUT/f"teacher_{d:02d}.npy"))
    g,r,field=[np.stack(c) for c in zip(*rows)]
    teachers=np.stack(teachers)
    labels=graph.cluster.clusters()[0]
    # Use the price partition fixed on the first four dates. Both eigenvectors and graph coefficients
    # are fitted without any labels from the fold in which predictions are evaluated.
    mask=(labels[:,None]==labels)&~np.eye(50,dtype=bool)
    ids=np.arange(len(g))
    views=[]
    for k in (0,1):
        use=ids%2==k
        def transform(target):
            z=graph.prepare(g[use].mean(0),target[use].mean(0),field[use].mean(0))[0].reshape(100,50,50)
            return (z*mask).reshape(100,-1)
        views.append([transform(v) for v in (r,teachers[:,0],teachers[:,1])])
    full,white,corr,sd=graph.prepare(g.mean(0),r.mean(0),field.mean(0))
    weights=[]
    for j in range(3):
        pair=[]
        for k in (0,1):
            src=views[k][1 if j==0 else 2]
            if j==2:
                other=views[1-k][0]
                kernel=src@other.T
            else:
                kernel=src@src.T
            ev,u=np.linalg.eigh(.5*(kernel+kernel.T))
            use=ev.argsort()[-8:]
            coef=u[:,use]
            projected=coef@(coef.T@views[1-k][0])
            pair.append(projected)
        wa,wb=pair
        # Retain stronger weight when the two cross-date estimates agree on an edge's sign.
        w=.5*(wa+wb)
        support=np.divide(2*wa*wb,wa**2+wb**2+1e-10).clip(0,1)
        w=(white.T@(w*(.25+.75*support))).reshape(100,50,50)
        w[:,np.arange(50),np.arange(50)]-=(corr*w).sum(1)
        weights.append(w/sd[:,:,None])
    np.savez(OUT/f"fit_{end}.npz",w=np.stack(weights,axis=-1))
    return np.array([end])


def design(raw,grid,code,p,bank):
    return graph.read(raw,grid,p,bank,OUT)


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
