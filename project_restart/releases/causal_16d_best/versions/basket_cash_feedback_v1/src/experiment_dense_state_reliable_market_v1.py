"""Learn market residuals from independent baskets with state-dependent loadings and robust target estimation."""
import json
import numpy as np

import experiment_dense_market_basket_error_teacher_v1 as basket

onset, rel, REFERENCE = basket.onset, basket.rel, basket.REFERENCE
OUT = REFERENCE.parent / "dense_state_reliable_market_v1"
ARMS = ("mean_teacher", "robust_teacher", "reliability_weighted_inputs")
MASKS = [np.arange(50) % 2 == 0] + [np.random.default_rng(234+i).permutation(50) < 25 for i in range(3)]
MASKS = MASKS + [~mask for mask in MASKS]


def samples(d):
    with np.load(basket.OUT / f"day_{d:02d}.npz") as f:
        x, y = f["x"].astype(np.float32), f["y"][..., 2].astype(float)
    grid = onset.settled.load(d, "time")
    t = np.concatenate([np.arange(a+30+d%4, b-60, 4) for a,b in rel.fields.parent.sessions(grid)])
    return x, y, grid[t]


def cells(x, time):
    activity = np.abs(x[..., 23:28]).mean(-1)
    return np.minimum(time.astype(int)//1170, 3)[:,None]*3 + np.digitize(activity,[.08,.22])


def risk(data):
    # The proxy for each name excludes both the name and its parity basket.
    moment = np.zeros((12,50,4))
    name = np.arange(50)[None, :]
    for x,y,grid in data:
        key = cells(x,grid)*50 + name
        left, right = y[:, ::2].mean(1), y[:, 1::2].mean(1)
        other = np.where(name%2==0,right[:,None],left[:,None])
        inside = np.where(name%2==0,left[:,None],right[:,None])
        own_excluded = (25*inside-y)/24
        obs = (np.ones_like(y), y*other, own_excluded*other, y*y)
        for j,z in enumerate(obs):
            moment[...,j] += np.bincount(key.ravel(),z.ravel(),minlength=600).reshape(12,50)
    total = moment.sum(0,keepdims=True)
    stats = (moment[...,1:]+180*total[...,1:]/total[...,:1].clip(1))/(moment[...,:1]+180)
    beta = (.65*stats[...,0]/stats[...,1].clip(.01)+.35).clip(.3,2.5)
    noise = (stats[...,2]-beta**2*stats[...,1]).clip(.025)
    return np.stack((beta,np.sqrt(noise)),axis=-1)


def teacher(y,pars,mask):
    beta,sd = pars[:,mask,0], pars[:,mask,1]
    z = y[:,mask]
    base = z.mean(1)
    m = base.copy()
    for _ in range(3):
        dist = np.abs(z-beta*m[:,None])/sd
        w = np.minimum(1,1.3/dist.clip(.001))/sd**2
        m = (w*beta*z).sum(1)/(w*beta**2).sum(1)
    return base, m


def basis(x, pars, mask):
    u = x[:,mask].astype(float)
    w = pars[:,mask,0]/pars[:,mask,1]**2
    w = (w/w.mean(1,keepdims=True)).clip(.25,3)
    w /= w.mean(1,keepdims=True)
    # Price disagreement, book strength and breadth within each basket.
    nonlinear = np.concatenate((np.tanh(2*u[:,:,:8]), u[:,:,:3]*np.tanh(2*np.abs(u[:,:,:3])),
                                u[:,:,3:8]*np.abs(u[:,:,23:28]).clip(0,1)),axis=-1)
    full = np.concatenate((u,nonlinear),axis=-1)
    simple = full.mean(1)
    weighted = (full*w[...,None]).mean(1)
    return np.stack((simple,weighted),axis=1)


def train(end,data=None,output=OUT):
    if data is None:
        data = [samples(d) for d in range(4,end)]
    # Out-of-day teacher fits prevent a date from selecting its own noise scale.
    tables = [risk(data[1-k::2]) for k in range(2)]
    rows = [[] for _ in ARMS]
    for i,(x,y,grid) in enumerate(data):
        pars = tables[i%2][cells(x,grid),np.arange(50)]
        daily = [0. for _ in ARMS]
        for mask in MASKS:
            inputs = basis(x,pars,mask)
            plain,robust = teacher(y,pars,~mask)
            for a,(f,target) in enumerate(((inputs[:,0],plain),(inputs[:,0],robust),(inputs[:,1],robust))):
                z = np.column_stack((f,target))
                z -= z.mean(0)
                daily[a] += z.T@z/(len(z)*len(MASKS))
        for a in range(len(ARMS)):
            rows[a].append(daily[a])
    weights = []
    for daily in rows:
        gram = np.mean(daily,axis=0)[:-1,:-1]
        unit = np.sqrt(gram.diagonal()).clip(.003)
        eig, axes = np.linalg.eigh(gram/np.outer(unit,unit))
        proj = axes/(unit[:,None]*np.sqrt(eig.clip(0)+1.))
        grad = np.stack([row[:-1,-1]@proj for row in daily])
        mu = grad.mean(0)
        weights.append(proj@(mu*mu**2/(mu**2+grad.var(0,ddof=1)/len(grad)+1e-8)))
    np.savez(output/f"fit_{end}.npz", weights=np.stack(weights),table=risk(data),rate=basket.front.direction.table(end))
    return np.array([end])


def design(raw,grid,code,p,bank,output=OUT):
    with np.load(output/f"fit_{int(bank[0])}.npz") as f:
        weights,table,rate = f["weights"],f["table"],f["rate"]
    x = basket.features(raw,grid,code,p,rate)
    parts = []
    for t in np.array_split(np.arange(len(x)), max(1,(len(x)+159)//160)):
        z = x[t]
        pars = table[cells(z,grid[t]),np.arange(50)]
        # Average across the same deterministic balanced partitions used during fit.
        f = np.mean([basis(z,pars,m) for m in MASKS],axis=0)
        parts.append(np.column_stack((f[:,0]@weights[0], f[:,0]@weights[1], f[:,1]@weights[2])))
    return np.concatenate(parts)


def audit():
    result = {}
    for end in (8,12):
        with np.load(OUT/f"fit_{end}.npz") as f:
            table = f["table"]
        rows = []
        for day in range(end,end+4):
            x,y,grid = samples(day)
            pars = table[cells(x,grid),np.arange(50)]
            sums = np.zeros((4,3))
            for m in MASKS:
                plain,robust = teacher(y,pars,m)
                target = y[:,~m].mean(1)
                phase = np.minimum(grid//1170,3)
                for k in range(4):
                    keep = phase==k
                    sums[k] += (np.sum((plain[keep]-target[keep])**2),np.sum((robust[keep]-target[keep])**2),keep.sum())
            rows.append(sums)
        total = np.sum(rows,axis=0)
        result[str(end+1)] = dict(by_phase_mse=(total[:,:2]/total[:,2:]).tolist(),
                                   aggregate=(total.sum(0)[:2]/total.sum(0)[2]).tolist())
    (OUT/"independent_basket_mse.json").write_text(json.dumps(result,indent=2))


def predict(raw,grid,p,params):
    out = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    out[...,2] += (z@params["w"])[:,None].astype(out.dtype)
    return out


if __name__ == "__main__":
    basket.rolling(OUT,ARMS,train,design)
    audit()
