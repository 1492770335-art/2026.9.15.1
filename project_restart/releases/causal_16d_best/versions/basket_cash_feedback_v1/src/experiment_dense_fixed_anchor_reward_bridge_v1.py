"""Measure whether an expanding incumbent overweights a fixed short-reward signal relative to fixed BookNet."""
from functools import lru_cache
import sys
import numpy as np
import torch

import y3_four_day_neural_references as networks
import experiment_dense_trace_reward_rollout_v1 as rollout

strip,base,rel,onset,REFERENCE=rollout.strip,rollout.base,rollout.rel,rollout.onset,rollout.REFERENCE
OUT=REFERENCE.parent/"dense_fixed_anchor_reward_bridge_v1"
ARMS=("anchor_relative_payoff","absolute_payoff","two_calibrations_agree")


@lru_cache(maxsize=1)
def frozen():
    path=networks.book_checkpoint(4)
    state=torch.load(path,weights_only=True,map_location="cpu")
    m=networks.phase.BookNet(state["current.0.weight"].shape[1]).double().eval().cuda()
    m.load_state_dict(state)
    return m


@torch.inference_mode()
def book(raw):
    t=np.arange(len(raw))
    x,vol=networks.phase.spline.features(raw,t)
    out=[]
    for i in range(0,len(t),64):
        ix=np.minimum(i+np.arange(64),len(t)-1)
        z=torch.from_numpy(x[ix]).cuda().double()
        scale=torch.from_numpy((.45+.55*vol[ix]).clip(.55,3)).cuda().double()
        gate=torch.as_tensor(np.exp(-ix[:,None]/400),device=z.device)
        out.append(frozen()(z,scale,gate)[0][:min(64,len(t)-i)].cpu().numpy())
    return strip.center(np.concatenate(out))


def law():
    mats=[]
    for d in range(4):
        with np.load(rollout.OUT/f"day_{d:02d}.npz") as f:
            mats.append(f["mom"])
    return rollout.laws(np.stack(mats))[:,0]


def panels(raw,grid,p,co):
    fixed=book(raw)
    for a,b,x in rollout.sessions(raw,grid):
        yield a,b,np.concatenate((fixed[a:b],x@co,strip.center(p[a:b].astype(float))),axis=-1)


def collect():
    OUT.mkdir(exist_ok=True)
    co=law()
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=base.read_p(d)
        total=np.zeros((10,10));count=0
        for a,b,z in panels(raw,grid,p,co):
            t=np.arange(d%4,len(z)-60,4)
            v=np.concatenate((z[t],strip.center(y[a+t,:,1:].astype(float))),axis=-1).reshape(-1,10)
            total+=v.T@v
            count+=len(v)
        np.save(OUT/f"gram_{d:02d}.npy",total/count)


def bridge(g):
    # Remove the three forecasts of the fixed BookNet from the two rollout values.
    v=np.eye(8)[:,3:5]
    v[:3]=-rollout.ridge(g[:3,:3],g[:3,3:5],.015)
    risks=(v.T@g[:8,:8]@v).diagonal().clip(1e-6)
    return v,risks


def train(end):
    rows=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    fixed=[];absolute=[];beta=[]
    for g in rows:
        v,risk=bridge(g)
        # Ratio of directional payoffs in stable forecast coordinates, independent of incumbent units.
        pay=(v.T@g[:8,8:]).diagonal()/risk
        start=g[1:3,8:].diagonal()/g.diagonal()[1:3].clip(1e-6)
        fixed.append(pay/start.clip(.05))
        absolute.append(pay)
        beta.append(g[6:8,8:].diagonal()/g.diagonal()[6:8].clip(1e-6))
    return np.r_[law().ravel(),rows.mean(0)[:8,:8].ravel(),np.mean(fixed,0),
                 np.mean(absolute,0),np.mean(beta,0)]


def signals(raw,grid,p,bank):
    co=bank[:-70].reshape(-1,2)
    history=bank[-70:-6].reshape(8,8)
    fixed,absolute,beta=bank[-6:-4],bank[-4:-2],bank[-2:]
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,z in panels(raw,grid,p,co):
        g=history.copy()
        for i in range(0,len(z),32):
            v,risk=bridge(g)
            current=(v.T@g[:,6:8]).diagonal()/risk
            anchor=g[1:3,6:8].diagonal()/g.diagonal()[1:3].clip(1e-6)
            relprice=fixed*anchor-current
            absprice=absolute/beta.clip(.05)-current
            same=np.where(relprice*absprice>0,np.sign(relprice)*np.minimum(abs(relprice),abs(absprice)),0.)
            # Preserve the local covariance with all three incumbent outputs.
            v[5:]-=rollout.ridge(g[5:,5:],g[5:]@v,.015)
            sigma=np.sqrt((v.T@g@v).diagonal().clip(1e-7))
            weights=np.stack((relprice,absprice,same))
            weights=np.clip(weights,-.025/sigma,.025/sigma)
            zz=z[i:i+32]
            out[a+i:a+i+len(zz)]=(zz@v)[:,:,None,:]*weights
            flat=zz.reshape(-1,8)
            decay=np.exp(-len(zz)/640.)
            g=g*decay+(1-decay)*(flat.T@flat/len(flat))
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=="__main__":
    torch.set_num_threads(2)
    collect()
    strip.evaluation.rolling(sys.modules[__name__])
