"""Price the 21--40 tick residual with rank changes replicated on separate dates and distinct fields."""
import json
import numpy as np
import torch
import experiment_dense_rank_motion_sources_v1 as local

onset,rel,REFERENCE=local.onset,local.rel,local.REFERENCE
OUT=REFERENCE.parent/"dense_middle_field_committee_v1"
ARMS=("middle_all_fields","replicated_diverse_fields","middle_beyond_near")


def combine(grams,cross,sparse):
    gram=grams.mean(0)
    unit=np.sqrt(gram.diagonal()).clip(1e-5)
    cov=gram/np.outer(unit,unit)
    signal=cross/unit
    mu=signal.mean(0)
    se=signal.std(0,ddof=1)/np.sqrt(len(signal))
    reliability=mu**2/(mu**2+se**2+1e-8)
    reward=mu*reliability
    if not sparse:
        return np.linalg.solve(cov+1.5*np.eye(len(mu)),reward)/unit
    keep=[]
    w=np.zeros(len(mu))
    # Add a field only when it contributes beyond fields already selected.
    for _ in range(12):
        gradient=signal-grams@(w/unit)/unit
        gain=gradient.mean(0)
        lower=abs(gain)-.55*gradient.std(0,ddof=1)/np.sqrt(len(signal))
        positive=((gradient*np.sign(gain))>0).mean(0)
        merit=lower*reliability
        merit[positive<.65]=-np.inf
        if keep:
            merit[np.max(abs(cov[keep]),axis=0)>.82]=-np.inf
            merit[keep]=-np.inf
        best=int(merit.argmax())
        if merit[best]<=0:
            break
        keep.append(best)
        w[keep]=np.linalg.solve(cov[np.ix_(keep,keep)]+.8*np.eye(len(keep)),reward[keep])
    return w/unit


def build_surface(gs,cs):
    w=local.surface(gs,cs,local.GROUPS[1])[:,:5,:2]
    near,mid=w[...,0],w[...,1]
    g=gs[:,:5,:5]
    def inner(a,b):
        return np.einsum("fc,fcd,fd->f",a,g,b)
    orth=mid-near*(inner(near,mid)/inner(near,near).clip(1e-6))[:,None]
    return np.stack((mid,orth),axis=-1)


def train(end):
    days=list(range(4,end))
    stats=[]
    for d in days:
        with np.load(local.OUT/f"day_{d:02d}.npz") as f:
            stats.append((f["gram"],f["cross"]))
    gs,cs=[np.stack(v) for v in zip(*stats)]
    grams=[]; crosses=[]
    for i,d in enumerate(days):
        elsewhere=np.arange(len(days))!=i
        w=build_surface(gs[elsewhere].mean(0),cs[elsewhere].mean(0))
        with np.load(local.OUT/f"day_{d:02d}.npz") as f:
            x=f["x"][...,:5].astype(np.float32)
            y=f["y"][:,1]
        z=np.einsum("nfc,fca->anf",x,w.astype(np.float32),optimize=True)
        grams.append(z.transpose(0,2,1)@z/len(y))
        crosses.append(z.transpose(0,2,1)@y/len(y))
    grams,crosses=np.stack(grams),np.stack(crosses)
    full=build_surface(gs.mean(0),cs.mean(0))
    columns=[]; selected=[]
    for j,sparse in ((0,False),(0,True),(1,True)):
        v=combine(grams[:,j].astype(float),crosses[:,j].astype(float),sparse)
        columns.append(full[...,j]*v[:,None])
        selected.append(np.flatnonzero(v).tolist())
    np.savez(OUT/f"fit_{end}.npz",w=np.stack(columns,axis=-1))
    (OUT/f"fields_{end}.json").write_text(json.dumps(dict(zip(ARMS,selected)),indent=2))
    return np.array([end])


def ranks(raw,t,device):
    a=torch.as_tensor(np.asarray(raw[t]),device=device,dtype=torch.float32).transpose(1,2).contiguous()
    ordered=a.sort(-1).values
    return ((torch.searchsorted(ordered,a)+torch.searchsorted(ordered,a,right=True)-50)/49).transpose(1,2)


def motion(cur,old):
    change=cur-old
    return torch.stack((cur,cur*cur.abs(),change,change*cur.abs(),change*change.abs()),dim=-1)


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        w=f["w"].reshape(-1,3)
    device="cuda" if torch.cuda.is_available() else "cpu"
    coef=torch.as_tensor(w,device=device)
    start=local.origins(grid)
    z=np.zeros((*p.shape[:2],3,3))
    for t in np.array_split(np.arange(len(p)),max(1,(len(p)+63)//64)):
        diff=motion(ranks(raw,t,device),ranks(raw,np.maximum(t-20,start[t]),device))
        value=(diff.reshape(len(t),50,-1).double()@coef).cpu().numpy()
        z[t,:,:,2]=value-value.mean(1,keepdims=True)
    return z


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
