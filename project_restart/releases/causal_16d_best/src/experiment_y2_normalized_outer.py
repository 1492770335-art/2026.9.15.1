"""Frozen causal-normalization candidates: train 0-15, evaluate 16-19.

The late dates have been used previously. This is a retrospective validation
experiment, explicitly requested after the earlier-fold pilot did not pass.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import reproduce_restart_k16 as repro

ROOT=Path(__file__).resolve().parent
REPO=ROOT/"2026.9.15.1"
OUT=ROOT/"artifacts/project_restart_generalization/y2_normalized_outer"
BASE=ROOT/"artifacts/project_restart_k16_reproduction"
CAP=ROOT/"artifacts/project_restart_generalization/capacity_ensemble_outer"
CURRENT=ROOT/"artifacts/project_restart_generalization/y1_to_y2_blend_outer"

def prepare():
    os.chdir(REPO)
    sys.path[:0]=[str(REPO/"project_restart/src/models"),str(REPO/"project_restart"),
                 str(REPO/"project/project")]
    import gru_strong as src
    import exp_tc34_s2 as market
    src.DATA=str(common.DATA);src.CFG["window"]=16
    torch.set_num_threads(4)
    return src,market

def train(src):
    X,y,tid=norm.load_days(list(range(16)),normalized=True)
    print(json.dumps({"stage":"loaded_training","rows":len(X)}),flush=True)
    path=OUT/"training.json"
    records=json.loads(path.read_text()) if path.exists() else []
    done={(r["hidden"],r["seed"]):r for r in records}
    Y=torch.from_numpy(y[:,2:3]).to("cuda:0")
    for hidden in (128,64):
        for seed in (42,43,44):
            if (hidden,seed) in done:
                r=done[hidden,seed]
                if common.sha256(r["checkpoint"])!=r["sha256"]:
                    raise RuntimeError("Checkpoint changed")
                continue
            torch.manual_seed(seed)
            model=src.GRU(hidden).to("cuda:0")
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            started=time.perf_counter();losses=[]
            for epoch in range(2):
                model.train();total=0.0
                order=np.random.default_rng(seed+epoch).permutation(len(X))
                batches=(len(X)+1023)//1024
                for batch,start in enumerate(range(0,len(X),1024),1):
                    I=order[start:start+1024]
                    inp=torch.from_numpy(src.make_batch(X,I,tid,16)).to("cuda:0").float()
                    loss=F.mse_loss(model(inp),Y[I])
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                    total+=float(loss.detach())*len(I)
                    if batch==1 or batch%1000==0 or batch==batches:
                        print(json.dumps({"stage":"training","hidden":hidden,"seed":seed,
                                          "epoch":epoch+1,"batch":batch,"batches":batches,
                                          "mean_loss":total/min(batch*1024,len(X))}),flush=True)
                losses.append(total/len(X))
            checkpoint=OUT/f"norm_h{hidden}_s{seed}_y2.pth"
            torch.save(model.state_dict(),checkpoint)
            records.append({"hidden":hidden,"seed":seed,"losses":losses,
                            "checkpoint":str(checkpoint),"sha256":common.sha256(checkpoint),
                            "seconds":time.perf_counter()-started})
            common.save_json(path,records)
            print(json.dumps({"stage":"model_done",**records[-1]}),flush=True)
            del model,opt
            gc.collect();torch.cuda.empty_cache()
    del X,y,tid,Y
    gc.collect();torch.cuda.empty_cache()

def transform(raw,mu,sd):
    # Match pilot cache quantization, and update state after the current output.
    grid=raw.reshape(-1,50,100).astype(np.float16)
    output=np.empty_like(grid)
    mean=np.broadcast_to(mu,(50,83)).copy()
    var=np.broadcast_to(sd*sd,(50,83)).copy()
    for tick,x in enumerate(grid):
        current=x[:,:83].astype(np.float32)
        output[tick,:,:83]=np.clip((current-mean)/np.sqrt(np.maximum(var,.25*sd*sd)),-6,6)
        output[tick,:,83:]=x[:,83:]
        diff=current-mean
        mean+=norm.ALPHA*diff
        var=(1-norm.ALPHA)*var+norm.ALPHA*diff*diff
    return output.reshape(-1,100)

def metrics(p,y):
    cs=[norm.corr(p[:,j],y[:,j]) for j in range(3)]
    return {"corr":cs,"score":float(np.dot(cs,[.2,.3,.5])),"rows":len(y),
            "rounded":{"corr0":round(cs[0],4),"corr1":round(cs[1],4),
                       "corr2":round(cs[2],4),"score":round(float(np.dot(cs,[.2,.3,.5])),4)}}

def evaluate(src,market):
    X,y,tid=src.load_block([16,17,18,19])
    mu,sd=norm.global_stats()
    day_rows=[len(np.load(norm.CACHE/f"day_{d:02d}_x.npy",mmap_mode="r"))*50
              for d in (16,17,18,19)]
    if sum(day_rows)!=len(X):raise RuntimeError("Validation rows mismatch")
    parts=[];start=0
    for day,n in zip((16,17,18,19),day_rows):
        cached=np.load(norm.CACHE/f"day_{day:02d}_x.npy",mmap_mode="r")
        if not np.array_equal(X[start:start+n].astype(np.float16).reshape(cached.shape),cached):
            raise RuntimeError("Raw/cache row alignment mismatch")
        parts.append(transform(X[start:start+n],mu,sd));start+=n
    Xn=np.concatenate(parts);del parts
    Xm=repro.LazyMarketArray(X)
    components={}
    specs=[("y0",128,BASE,"market_full_s{seed}_y0.pth",Xm,True),
           ("y1",128,BASE,"market_full_s{seed}_y1.pth",Xm,True),
           ("raw128",128,BASE,"k16_full_s{seed}_y2.pth",X,False),
           ("raw64",64,CAP,"k16_full_s{seed}_y2.pth",X,False),
           ("norm128",128,OUT,"norm_h128_s{seed}_y2.pth",Xn,False),
           ("norm64",64,OUT,"norm_h64_s{seed}_y2.pth",Xn,False)]
    for name,hidden,directory,pattern,features,is_market in specs:
        ps=[]
        for seed in (42,43,44):
            model=(market.GRU300(hidden) if is_market else src.GRU(hidden)).to("cuda:0")
            model.load_state_dict(torch.load(directory/pattern.format(seed=seed),
                                             map_location="cuda:0",weights_only=True))
            ps.append(norm.predict(src,model,features,tid))
            del model
        components[name]=np.stack(ps).mean(0)
        print(json.dumps({"stage":"predictions_ready","component":name}),flush=True)
    p0,p1=components["y0"],components["y1"]
    options={"baseline":.5*p1+.25*(components["raw128"]+components["raw64"]),
             "replace128":.5*p1+.25*(components["norm128"]+components["raw64"]),
             "replace_both":.5*p1+.25*(components["norm128"]+components["norm64"])}
    result={"note":"Retrospective late-date validation; candidates fixed before this evaluation",
            "models":{},"baseline_official":json.loads((CURRENT/"results.json").read_text())["pooled_official"]}
    for name,p2 in options.items():
        pred=np.stack([p0,p1,p2],-1)
        result["models"][name]={"pooled":metrics(pred,y),"daily":{}}
        start=0
        for day,n in zip((16,17,18,19),day_rows):
            result["models"][name]["daily"][str(day)]=metrics(pred[start:start+n],y[start:start+n])
            start+=n
        print(json.dumps({"stage":"evaluated","name":name,**result["models"][name]["pooled"]}),flush=True)
    if result["models"]["baseline"]["pooled"]["rounded"]!=result["baseline_official"]:
        raise RuntimeError("Vectorized baseline does not match official recorded scores")
    np.savez(OUT/"validation_components.npz",**components,target=y,day_rows=np.array(day_rows))
    common.save_json(OUT/"vectorized_results.json",result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["train","evaluate"],required=True)
    args=parser.parse_args()
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError("Data SHA mismatch")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True,exist_ok=True)
    plan={"train_days":list(range(16)),"validation_days":[16,17,18,19],
          "window":16,"hidden":[128,64],"seeds":[42,43,44],"epochs":2,
          "lr":1e-4,"weight_decay":1e-4,"batch":1024,"clip":1.0,
          "preprocessing":"Same EWMA64 causal normalization as earlier pilot; global stats days0-3; float16 cache semantics",
          "primary":"replace128","secondary":"replace_both",
          "blend":"0.5*y1_market +0.25*y2_128 +0.25*y2_64",
          "late_dates_status":"Repeatedly exposed validation, not blind; user requested this check despite earlier promotion gate failure",
          "data_sha256":common.EXPECTED_SHA}
    path=OUT/"plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError("Plan mismatch")
    common.save_json(path,plan)
    mu,sd=norm.global_stats();np.savez(OUT/"normalization_stats.npz",mu=mu,sd=sd)
    src,market=prepare()
    if args.stage=="train":train(src)
    else:evaluate(src,market)

if __name__=="__main__":main()
