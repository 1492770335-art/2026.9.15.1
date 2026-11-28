"""Use predictable residual-rank extremity to calibrate each horizon's stock and market exposure."""
import json

import numpy as np
import torch

import experiment_dense_balanced_panel_ranks_v1 as rank
from experiment_dense_peer_activity_bridge_v1 import load
from experiment_dense_shared_horizon_calibration_v1 import score

rel,REFERENCE=rank.rel,rank.REFERENCE
OUT=REFERENCE.parent/"dense_rank_risk_calibration_v1"
ARMS=("usual_stock_risk","current_risk_change","common_forecast_exposure")
RADII=np.linspace(-.9,.9,rank.BINS)**2


def bank(end):
    if not (rank.OUT/f"net_{end}.pt").exists():
        rank.train(end)
    prior=[]
    for d in range(4,end):
        with np.load(rank.band.OUT/f"day_{d:02d}.npz") as f:
            y=torch.from_numpy(f["y"][...,3])
        prior.append((rank.rank_targets(y).mean(0).double().numpy()@RADII))
    return np.r_[end,np.mean(prior,0)]


@torch.inference_mode()
def design(raw,grid,code,p,params):
    artifact=torch.load(rank.OUT/f"net_{int(params[0])}.pt",map_location="cpu",weights_only=True)
    model=rank.PanelRank(len(artifact["unit"])).double().eval()
    model.load_state_dict(artifact["state"])
    inputs=rank.band.pairs.net.features(raw,grid,code,p)
    parts=[]
    for i in range(0,len(inputs),64):
        u=(torch.as_tensor(inputs[i:i+64]).double()/artifact["unit"]).clamp(-5,5)
        logits,_=model(u)
        parts.append(rank.assignment(logits).numpy()@RADII)
    current=np.concatenate(parts)
    prior=params[1:]-params[1:].mean()
    change=current-current.mean(1,keepdims=True)-prior
    m=p.mean(1,keepdims=True,dtype=float)
    q=p-m
    z=np.stack((q*prior[None,:,None],q*change[...,None],m*change[...,None]),axis=2)
    return z-z.mean(1,keepdims=True)


def main(output=OUT, arms=ARMS, fit_bank=bank, read=design, reference=REFERENCE):
    output.mkdir(exist_ok=True)
    code=rel.fields.parent.codec()
    common_y1=rel.rolling_short_changes()
    history=[[] for _ in range(3)]
    rows={(h,k):[] for h in range(3) for k in arms}
    for d in range(6,16):
        if d in (6,8,12):
            params=fit_bank(d)
        p=np.load(reference/f"rolling_{d:02d}.npy")
        if d in common_y1:
            p[:,:,0]+=common_y1[d][:,None]
        x,grid,y=load(d,"x"),load(d,"time"),load(d,"y")
        z=read(x,grid,code,p,params)
        if d==8:
            np.testing.assert_allclose(z[:512],read(x[:512],grid[:512],code,p[:512],params),atol=1e-7,rtol=0)
        for h in range(3):
            if d in (8,12):
                if h==0:
                    weights={}
                for j,k in enumerate(arms):
                    weights[h,k]=rel.relative.fit(history[h],j,h)
            if d>=8:
                xx=rel.relative.features(p,z[...,h])
                for k in arms:
                    rows[h,k].append(rel.ref.market.metric(p[:,:,h],y[:,:,h],xx@weights[h,k]))
            history[h].append(rel.relative.stats(p,y,z[...,h]))
    report={str(h):{k:score(rows[h,k]) for k in arms} for h in range(3)}
    selected={}
    w=np.zeros((3+len(arms),3))
    for h in range(3):
        tests=report[str(h)]
        passed=[k for k,v in tests.items() if min(v["folds"]+v["omit_one"])>0 and sum(a>0 for a in v["days"])>=5]
        if passed:
            k=max(passed,key=lambda key:min(tests[key]["folds"]))
            selected[str(h)]=k
            w[:,h]=rel.relative.fit(history[h],arms.index(k),h)
    (output/"rolling.json").write_text(json.dumps(dict(selected=selected,results=report),indent=2))
    print(selected,{h:{k:v["folds"] for k,v in a.items()} for h,a in report.items()},flush=True)
    if selected:
        np.savez(output/"weights.npz",bank=fit_bank(16),w=w,code=code)


def predict(raw,grid,p,params):
    z=design(raw,grid,params["code"],p,params["bank"])
    q=p.copy()
    for h in range(3):
        q[:,:,h]+=(rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
