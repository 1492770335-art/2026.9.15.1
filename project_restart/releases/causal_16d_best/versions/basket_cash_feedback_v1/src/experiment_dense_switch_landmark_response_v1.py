"""Estimate residual returns after a reversal from the partial observed trajectory at the current spell age."""
import numpy as np

import experiment_dense_switch_impulse_curve_v1 as impulse

clock,onset,rel,REFERENCE = impulse.clock,impulse.onset,impulse.rel,impulse.REFERENCE
OUT = REFERENCE.parent / "dense_switch_landmark_response_v1"
ARMS = ("partial_price_response","shape_beyond_total_move","stock_specific_trajectory")
DIMS = (50,5,5,3,2)
N = int(np.prod(DIMS))
STOCK = np.arange(50)[None,:,None]


def sessions(raw,grid,code,p,rates):
    q = p-p.mean(1,keepdims=True,dtype=float)
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"][:,1],f["row"]
    for a,b,v,s,start,_ in onset.sessions(raw,grid,code):
        visible = ~np.all(raw[a:b]==missing,axis=-1)
        r = np.log1p(k*(raw[a:b,:,0].astype(float)-.00136))*coef[2]/k
        r -= (r*visible).sum(1,keepdims=True)/visible.sum(1,keepdims=True).clip(1)
        age,keys = clock.keys(v,s,start)
        chance = rates[keys,0]
        origin = np.maximum(0,start-1)
        prior = np.take_along_axis(chance,origin,axis=0)
        gaps = np.vstack((np.zeros((1,50),int),(~visible).cumsum(0)))
        use = (age>=1)&(age<=120)&(start>1)&(gaps[1:,:,None]==gaps[origin,STOCK])
        centered = r*visible
        yield a,b,v,s,start,age,prior,use,centered,centered.cumsum(0),q[a:b]


def features(t,v,s,start,age,prior,use,moves,cash,q):
    st,dt,side = start[t],age[t],s[t]
    mid = st+dt//2
    origin = cash[st,STOCK]
    halfway = cash[mid,STOCK]
    norm = np.sqrt(1+dt/12)
    full = side*(cash[t,:,None]-origin)/norm
    # Contrasting the later and earlier return controls for total signed displacement.
    curve = side*((cash[t,:,None]-halfway)-(halfway-origin))/norm
    opening = side*moves[st,STOCK]
    agree = side*s[t].mean(2,keepdims=True)>0
    key = np.ravel_multi_index(np.broadcast_arrays(STOCK,np.arange(5)[None,None,:],
              np.digitize(dt,[4,12,30,65]),np.digitize(prior[t],[.08,.30]),agree.astype(int)),DIMS)
    data = np.concatenate((side[...,None]*q[t,:,None,:],(side*v[t])[...,None],
                           np.stack((np.tanh(opening),np.tanh(full),np.tanh(curve)),axis=-1)),axis=-1)
    return key,side,use[t],data


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        q = p-p.mean(1,keepdims=True,dtype=float)
        target = y-y.mean(1,keepdims=True,dtype=float)
        beta = ((q*target).sum((0,1))/(q*q).sum((0,1))).clip(.2,4)
        err = target/beta-q
        moment = np.zeros((N,8,11))
        for a,b,*state in sessions(raw,grid,code,p,clock.rates(d)):
            for t in np.array_split(np.arange(25+d%5,b-a-60,5),5):
                ids,sgn,use,x = features(t,*state)
                xx = np.column_stack((np.ones(use.sum()),x[use]))
                yy = (sgn[...,None]*err[a+t,:,None,:])[use]
                z = np.column_stack((xx,yy))
                idx = ids[use]
                for j in range(8):
                    for l in range(j,11):
                        moment[:,j,l] += np.bincount(idx,xx[:,j]*z[:,l],minlength=N)
        for j in range(1,8):
            moment[:,j,:j] = moment[:,:j,j]
        np.save(OUT/f"moment_{d:02d}.npy",moment)


def train(end):
    cov,margins = [],[]
    for d in range(4,end):
        m = np.load(OUT/f"moment_{d:02d}.npy").reshape(*DIMS,8,11)
        cov.append(m[...,1:,1:] - m[...,1:,:1]*m[...,:1,1:] / m[...,:1,:1].clip(1))
        margins.append(m[...,0,:8])
    hist = np.stack(cov)
    sums = np.sum(margins,axis=0)
    pooled = sums.sum(0,keepdims=True)
    center = (sums[...,1:]+16*pooled[...,1:]/pooled[...,:1].clip(1))/(sums[...,:1]+16)
    def fit(axes, indices):
        g = hist.sum(axis=axes,keepdims=True)
        n = g[..., :7].mean(0)
        yi = g[...,7:]
        fs = list(range(4))+indices
        block = n[...,fs,:][...,fs]
        scale = np.diagonal(block,axis1=-2,axis2=-1)
        reg = block+np.eye(len(fs))*(4+.5*scale)[...,None,:]
        daily = np.linalg.solve(reg,yi[...,fs,:])
        mu = daily.mean(0)
        noise = daily.var(0,ddof=1)/len(daily)
        keep = mu*mu**2/(mu**2+noise+.00002)
        weight = np.zeros((*n.shape[:-2],7,3))
        weight[...,indices,:] = keep[...,4:,:]
        # Remove in-group exposure to the incumbent and current strength.
        ctrl = n[...,:4,:4]
        proj = np.linalg.solve(ctrl+np.eye(4)*(.05*np.diagonal(ctrl,axis1=-2,axis2=-1)+1)[...,None,:],
                               n[...,:4,:]@weight)
        weight[...,:4,:] -= proj
        return weight
    coarse = (1,4,5)
    general = fit(coarse,[4,5])
    shape = fit(coarse,[4,5,6])-general
    name = fit((3,4,5),[4,5,6])-fit((1,3,4,5),[4,5,6])
    weights = np.stack(np.broadcast_arrays(general,shape,name),axis=-2)
    weights = np.broadcast_to(weights,(*DIMS,7,3,3)).reshape(N,7,3,3)
    np.savez(OUT/f"fit_{end}.npz",rate=clock.rates(end),center=center.reshape(N,7),weights=weights)
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        rate,center,weights = f["rate"],f["center"],f["weights"]
    out = np.zeros((*p.shape[:2],3,3))
    for a,b,*state in sessions(raw,grid,code,p,rate):
        for t in np.array_split(np.arange(b-a),max(1,(b-a+63)//64)):
            cell,side,use,x = features(t,*state)
            dx = (x-center[cell])*(side*use)[...,None]
            out[a+t] = np.einsum("tsfi,tsfiah->tsah",dx,weights[cell])
    return out-out.mean(1,keepdims=True)


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[:,:,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
