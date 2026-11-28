"""Estimate predictable five-tick forecast errors at their issuance and transport them to the current book."""
import numpy as np
import experiment_dense_future_expert_value_v1 as future

ref,rel,panel=future.ref,future.rel,future.panel
OUT=ref.base.ROOT/"artifacts/dense_projected_future_surprises_v1"
ARMS=("later_book_calibration","distal_calibration","replicated_unexplained")
views=panel.views


def collect():
    OUT.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    for day in range(4,16):
        def load(k):return np.load(ref.base.CACHE/f"day_{day:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y").astype(float)
        p=future.forecast(day).astype(float)
        p-=p.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
        scale=((p*y).sum((0,1))/(p*p).sum((0,1))).clip(.2,4)
        # The proxy uses each earlier release's out-of-fold forecasts from this training day.
        unit=4*ref.base.UNITS[0]/scale[2]
        err=unit*(y[...,0]-scale[0]*p[...,0])
        x=panel.book(raw,grid,code)
        all_state=np.concatenate((p.astype(np.float32),x),axis=2)
        # Future-time controls span the predicted return curve and the complete contemporary core book.
        state=all_state[:,:,:99]
        n=all_state.shape[2];r=state.shape[2]
        cross=np.zeros((3,n,r));self_cov=np.zeros((3,r,r));rhs=np.zeros((3,r));count=0
        for a,b in rel.fields.parent.sessions(grid):
            t=np.arange(a,b-60,4)
            if not len(t):continue
            for band in range(3):
                offsets=np.arange(20*band,20*band+20,5)
                times=(t[:,None]+offsets).ravel()
                ids,multiplicity=np.unique(times,return_counts=True)
                for block in np.array_split(np.arange(len(ids)),6):
                    z=state[ids[block]].reshape(-1,r).astype(float)
                    weights=np.repeat(multiplicity[block],50)/4
                    self_cov[band]+=z.T@(z*weights[:,None])
                    rhs[band]+=z.T@(err[ids[block]].ravel()*weights)
                for k in np.array_split(t,5):
                    avg=sum(state[k+h].astype(float) for h in offsets)/4
                    cross[band]+=all_state[k].reshape(-1,n).astype(float).T@avg.reshape(-1,r)
            count+=len(t)*50
        np.savez(OUT/f"day_{day:02d}.npz",ab=cross/count,bb=self_cov/count,bs=rhs/count)
        print(f"Training day {day+1} complete",flush=True)


def train(days):
    records=[]
    for d in range(4,days):
        with np.load(future.OUT/f"day_{d:02d}.npz") as g,np.load(OUT/f"day_{d:02d}.npz") as f:
            records.append((g["gram"],g["cross"],f["ab"],f["bb"],f["bs"]))
    gs,xy,ab,bb,bs=map(np.stack,zip(*records))
    proj=np.linalg.solve(gs[:,:3,:3]+.01*np.eye(3)*gs[:,:3,:3].diagonal(axis1=1,axis2=2)[:,None],gs[:,:3,3:])
    ctrl=proj.transpose(0,2,1)
    gram=(gs[:,3:,3:]-ctrl@gs[:,:3,3:]-gs[:,3:,:3]@proj+ctrl@gs[:,:3,:3]@proj).mean(0)
    xy=xy[:,3:]-ctrl@xy[:,:3]
    ab=ab[:,:,3:]-ctrl[:,None]@ab[:,:,:3]
    repair=np.empty((len(records),3,gram.shape[0]))
    for fold in (0,1):
        history=slice(1-fold,None,2)
        test=slice(fold,None,2)
        cov=bb[history].mean(0)
        # The five-tick error has the same units across bands, but each later book has its own date weights.
        price=np.linalg.solve(cov+.45*np.eye(cov.shape[-1])*cov.diagonal(axis1=1,axis2=2)[:,None].clip(.0002),
                              bs[history].mean(0)[...,None])[...,0]
        repair[test]=np.einsum("dbij,bj->dbi",ab[test],price)
    sd=np.sqrt(gram.diagonal()).clip(.015)
    eig,axes=np.linalg.eigh(gram/np.outer(sd,sd))
    def modes(z):return (z/sd)@axes
    corrected=modes(xy[:,:,1]+repair.sum(1))
    distal=modes(xy[:,:,1]+repair[:,1:].sum(1))
    direct=modes(xy[:,:,2])
    difference=direct-corrected
    bias=difference.mean(0)
    se=difference.std(0,ddof=1)/np.sqrt(len(records))
    reliable=bias*np.abs(bias)/(np.abs(bias)+se+1e-9)
    signals=np.column_stack((corrected.mean(0),distal.mean(0),corrected.mean(0)+reliable))
    w=(axes@(signals/(eig[:,None].clip(0)+.55)))/sd[:,None]
    risk=np.einsum("ij,ik,kj->j",w,gs.mean(0)[3:,3:],w)
    return w/np.sqrt(risk.clip(1e-12))


if __name__=="__main__":
    collect()
    rel.main(OUT,ARMS,train,views)
