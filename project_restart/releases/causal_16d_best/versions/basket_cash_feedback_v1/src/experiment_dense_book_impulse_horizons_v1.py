"""Estimate how book innovations affect future five-tick returns and integrate their unexpired effects into Y3."""
import json
import numpy as np
import torch
import experiment_dense_rank_graph_timing_v1 as timing

onset,rel,REFERENCE = timing.onset,timing.rel,timing.REFERENCE
OUT = REFERENCE.parent/"dense_book_impulse_horizons_v1"
ARMS = ("direct_twenty_tick_blocks","integrated_five_tick_response","delayed_response")
LAGS = (5,10,20,40)
DIM = 32
TAPS = 16


def tensor_day(raw,device):
    return timing.panel(raw,device)


def history(panel,t):
    return torch.stack([panel[(t-lag).clamp_min(0)] for lag in LAGS],dim=-1)


def prepare():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gram = np.zeros((100,5,5))
    records = []
    for d in range(4):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        for a,b in rel.fields.parent.sessions(grid):
            p = tensor_day(raw[a:b],device)
            for start in range(40+d,b-a,512):
                t = torch.arange(start,min(start+512,b-a),8,device=device)
                h = torch.cat((history(p,t),p[t,...,None]),dim=-1)
                z = h.permute(2,0,1,3).reshape(100,-1,5)
                gram += (z.transpose(1,2)@z).cpu().numpy()
                records.append(h.cpu().numpy().astype(np.float16))
    ar = np.linalg.solve(gram[:,:4,:4]+.015*np.eye(4)[None]*gram[:,4:,4:],gram[:,:4,4:])[...,0]
    cov = 0.
    for h in records:
        e = h[...,4]-np.einsum("tsfl,fl->tsf",h[...,:4].astype(float),ar)
        u = e.reshape(-1,100)
        cov += u.T@u
    cov /= sum(len(x)*50 for x in records)
    sd = np.sqrt(cov.diagonal().clip(.0001))
    val,vec = np.linalg.eigh(cov/np.outer(sd,sd))
    proj = vec[:,-DIM:]/(sd[:,None]*np.sqrt(val[-DIM:]+.3))
    np.savez(OUT/"dictionary.npz",ar=ar,proj=proj)


def innovations(raw,device):
    with np.load(OUT/"dictionary.npz") as f:
        ar,proj = [torch.as_tensor(f[k],device=device,dtype=torch.float32) for k in ("ar","proj")]
    panel = tensor_day(raw,device)
    result = []
    for start in range(0,len(raw),64):
        t = torch.arange(start,min(start+64,len(raw)),device=device)
        e = panel[t]-(history(panel,t)*ar).sum(-1)
        result.append((e@proj).cpu().numpy())
    return np.concatenate(result)


def basis(e,t,p):
    times = np.maximum(t[:,None]-5*np.arange(TAPS),0)
    past = e[times].transpose(0,2,1,3)
    available = (t[:,None]>=5*np.arange(TAPS))[:,None,:,None]
    return np.concatenate((p[t],(past*available).reshape(len(t),50,-1)),axis=-1)


def target(labels,t,coef,k):
    blocks = np.stack([labels[t+offset,:,1] for offset in (0,20,40)],axis=-1).astype(float)
    blocks = np.log1p(k*(blocks-coef[1,0])/coef[1,1])*coef[2,1]/k
    small = np.log1p(k*(labels[t,:,0].astype(float)-coef[0,0])/coef[0,1])*coef[2,1]/k
    return np.concatenate((small[...,None],blocks),axis=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    prepare()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(onset.settled.GEO) as f:
        k,coef,missing = float(f["k"]),f["coeff"],f["row"]
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,key) for key in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
             onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        total = 0.
        count = 0
        for a,b in rel.fields.parent.sessions(grid):
            e = innovations(raw[a:b],device)
            gaps = np.vstack((np.zeros(50,int),np.all(raw[a:b]==missing,axis=-1).cumsum(0)))
            ticks = np.arange(120,b-a-60)
            idx = np.arange(len(ticks))
            sampled = ticks[idx%8==(d+idx//24)%8]
            for start in range(0,len(sampled),64):
                t = sampled[start:start+64]
                ok = gaps[t+61]==gaps[t]
                x = basis(e,t,p[a:b])
                z = np.concatenate((x,target(y[a:b],t,coef,k)),axis=-1)
                z -= (z*ok[...,None]).sum(1,keepdims=True)/ok.sum(1)[:,None,None].clip(1)
                flat = torch.as_tensor(z[ok],dtype=torch.float32,device=device)
                total += (flat.T@flat).cpu().numpy().astype(float)
                count += len(flat)
        np.save(OUT/f"gram_{d:02d}.npy",total/count)


def fit(rows,with_kernel=False):
    g = rows.mean(0)
    n = len(g)-4
    select = np.eye(n)[:,3:]
    base = g[:3,:3]
    select[:3] = -np.linalg.solve(base+.01*np.diag(base.diagonal()),g[:3,3:n])
    gram = select.T@g[:n,:n]@select
    sd = np.sqrt(gram.diagonal().clip(.0001))
    ev,rot = np.linalg.eigh(gram/np.outer(sd,sd))
    axes = select@(rot/(sd[:,None]*np.sqrt(ev.clip(0)+1.)))
    grad = axes.T@rows[:,:n,n:]
    mean = grad.mean(0)
    raw = axes@(mean*mean**2/(mean**2+grad.var(0,ddof=1)/len(grad)+1e-8))
    # At future origin t+5j, tap k+j refers to an innovation already observed at t-5k.
    impulse = raw[3:,0].reshape(TAPS,DIM)
    integrated = np.zeros((n,3))
    for h in range(3):
        taps = integrated[3:,h].reshape(TAPS,DIM)
        for j in range(4*h,4*h+4):
            taps[:TAPS-j] += impulse[j:]
    # Add the same projection on current forecasts to all integrated block estimates.
    integrated[:3] = select[:3]@integrated[3:]
    if with_kernel:
        return integrated,impulse,select
    return raw[:,1:],integrated


def train(end):
    rows = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    direct,known = fit(rows)
    heads = np.column_stack((direct.sum(1),known.sum(1),known[:,1:].sum(1)))
    np.savez(OUT/f"fit_{end}.npz",w=heads,direct=direct,known=known)
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        heads = f["w"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = np.zeros((*p.shape[:2],3,3))
    for a,b in rel.fields.parent.sessions(grid):
        e = innovations(raw[a:b],device)
        for start in range(0,b-a,64):
            t = np.arange(start,min(start+64,b-a))
            z = basis(e,t,p[a:b]).astype(float)@heads
            result[a+t,:,:,2] = z-z.mean(1,keepdims=True)
    return result


def audit():
    report = {}
    for end in (8,12):
        g = np.mean([np.load(OUT/f"gram_{d:02d}.npy") for d in range(end,end+4)],axis=0)
        n = len(g)-4
        c = g[3:,3:]-g[3:,:3]@np.linalg.solve(g[:3,:3],g[:3,3:])
        scores = {}
        with np.load(OUT/f"fit_{end}.npz") as f:
            for key in ("direct","known"):
                weights = f[key][3:]
                cov = weights.T@c[:n-3,-3:]
                var = np.diag(weights.T@c[:n-3,:n-3]@weights)
                scores[key] = (cov.diagonal()/np.sqrt(var*c.diagonal()[-3:]).clip(1e-12)).tolist()
        report[end+1] = scores
    (OUT/"forward_blocks.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def predict(raw,grid,p,params):
    q = p.copy()
    z = design(raw,grid,params["code"],p,params["bank"])
    q[...,2] += (rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
    audit()
