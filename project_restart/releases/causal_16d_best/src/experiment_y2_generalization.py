"""Causal normalization and day-balanced y2 loss on two forward folds.

Both candidates replace only the seed-42 raw GRU128 y2 member. The y1 and
raw GRU64 members and the 0.5/0.25/0.25 blend are held fixed. No dates 16-19
are read for model selection.
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

import experiment_y2_gru_capacity as common

ROOT=Path(__file__).resolve().parent
REPO=ROOT / "2026.9.15.1"
CACHE=ROOT / "artifacts/joint_transformer/cache"
OOF=ROOT / "artifacts/project_restart_generalization/y2_oof_corrector"
OUT=ROOT / "artifacts/project_restart_generalization/y2_generalization_v1"
ALPHA=2.0/65.0

def corr(p,y):
    return float(np.corrcoef(np.asarray(p,dtype=np.float64).ravel(),
                             np.asarray(y,dtype=np.float64).ravel())[0,1])

def global_stats():
    n=0; total=np.zeros(83,dtype=np.float64); squares=total.copy()
    for day in range(4):
        x=np.load(CACHE / f"day_{day:02d}_x.npy",mmap_mode="r")
        block=np.asarray(x[::16,:,:83],dtype=np.float64).reshape(-1,83)
        n+=len(block);total+=block.sum(0);squares+=(block*block).sum(0)
    mu=total/n
    sd=np.sqrt(np.maximum(squares/n-mu*mu,1e-6))
    return mu.astype(np.float32),sd.astype(np.float32)

def preprocess():
    norm_dir=OUT / "normalized_days"
    norm_dir.mkdir(parents=True,exist_ok=True)
    mu,sd=global_stats()
    for day in range(16):
        output=norm_dir / f"day_{day:02d}_x.npy"
        if output.exists():
            continue
        x=np.load(CACHE / f"day_{day:02d}_x.npy",mmap_mode="r")
        result=np.lib.format.open_memmap(output,mode="w+",dtype=np.float16,shape=x.shape)
        mean=np.broadcast_to(mu,(50,83)).copy()
        var=np.broadcast_to(sd*sd,(50,83)).copy()
        floor=np.broadcast_to(0.25*sd*sd,(50,83))
        for tick in range(len(x)):
            current=np.asarray(x[tick,:,:83],dtype=np.float32)
            z=(current-mean)/np.sqrt(np.maximum(var,floor))
            result[tick,:,:83]=np.clip(z,-6,6).astype(np.float16)
            result[tick,:,83:]=np.asarray(x[tick,:,83:],dtype=np.float16)
            diff=current-mean
            mean+=ALPHA*diff
            var=(1-ALPHA)*var+ALPHA*diff*diff
        result.flush();del result
        print(json.dumps({"stage":"normalized_day","day":day,
                          "rows":len(x)*50}),flush=True)
    return norm_dir,mu,sd

def load_days(days,normalized=False):
    base=(OUT / "normalized_days") if normalized else CACHE
    xx=[];yy=[];tt=[]
    for day in days:
        xname=f"day_{day:02d}_x.npy"
        x=np.load(base/xname,mmap_mode="r")
        y=np.load(CACHE/f"day_{day:02d}_y.npy",mmap_mode="r")
        xx.append(np.asarray(x).reshape(-1,100))
        yy.append(np.asarray(y).reshape(-1,3))
        tt.append(np.repeat(np.arange(len(x),dtype=np.int64),50))
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(tt)

def day_weights(y,days):
    cursor=0; sigmas=[]; counts=[]
    for day in days:
        x=np.load(CACHE/f"day_{day:02d}_x.npy",mmap_mode="r")
        n=len(x)*50
        sigmas.append(float(y[cursor:cursor+n,2].std()))
        counts.append(n);cursor+=n
    if cursor!=len(y):
        raise RuntimeError("Day alignment mismatch")
    reference=float(np.median(sigmas))
    levels=np.clip((reference/np.array(sigmas))**2,0.5,2.0)
    levels/=np.average(levels,weights=counts)
    weights=np.concatenate([np.full(n,w,dtype=np.float32)
                            for n,w in zip(counts,levels)])
    return weights,{str(d):{"sigma":s,"weight":float(w)}
                    for d,s,w in zip(days,sigmas,levels)}

def predict(src,model,X,tid):
    model.eval();outs=[]
    with torch.no_grad():
        for start in range(0,len(X),4096):
            idx=np.arange(start,min(start+4096,len(X)))
            batch=torch.from_numpy(src.make_batch(X,idx,tid,16)).to("cuda:0").float()
            outs.append(model(batch).cpu().numpy())
    return np.concatenate(outs).reshape(-1)

def evaluate(variant,fold,days,pred,truth):
    base_rows=np.concatenate([np.load(OOF/f"day_{d:02d}_oof.npy") for d in days])
    if len(base_rows)!=len(pred):
        raise RuntimeError("Prediction row mismatch")
    baseline=.5*base_rows[:,0]+.25*base_rows[:,1]+.25*base_rows[:,2]
    candidate=.5*base_rows[:,0]+.25*pred+.25*base_rows[:,2]
    result={"base":corr(baseline,truth[:,2]),
            "candidate":corr(candidate,truth[:,2]),"daily":{}}
    result["delta"]=result["candidate"]-result["base"]
    cursor=0
    for day in days:
        count=len(np.load(CACHE/f"day_{day:02d}_x.npy",mmap_mode="r"))*50
        sl=slice(cursor,cursor+count)
        a=corr(baseline[sl],truth[sl,2]);b=corr(candidate[sl],truth[sl,2])
        result["daily"][str(day)]={"base":a,"candidate":b,"delta":b-a}
        np.save(OUT/f"{variant}_{fold}_day_{day:02d}_y2_128.npy",pred[sl].astype(np.float32))
        cursor+=count
    if cursor!=len(pred):raise RuntimeError("Validation day alignment mismatch")
    return result

def run_variant(src,variant):
    results={"variant":variant,"folds":{}}
    for fold,spec in common.FOLDS.items():
        started=time.perf_counter()
        normalized=variant=="normalized"
        X,y,tid=load_days(spec["train"],normalized)
        Xv,yv,tv=load_days(spec["validation"],normalized)
        weights,weight_info=day_weights(y,spec["train"])
        if variant!="weighted":weights[:]=1.0
        torch.manual_seed(42)
        model=src.GRU(128).to("cuda:0")
        Y=torch.from_numpy(y[:,2:3]).to("cuda:0")
        W=torch.from_numpy(weights).to("cuda:0")
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        losses=[]
        for epoch in range(2):
            model.train();order=np.random.default_rng(42+epoch).permutation(len(X))
            total=0.0;batches=(len(X)+1023)//1024
            for batch,start in enumerate(range(0,len(X),1024),1):
                I=order[start:start+1024]
                input_=torch.from_numpy(src.make_batch(X,I,tid,16)).to("cuda:0").float()
                err=(model(input_)-Y[I]).square().reshape(-1)
                loss=(err*W[I]).mean()
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                total+=float(loss.detach())*len(I)
                if batch==1 or batch%500==0 or batch==batches:
                    print(json.dumps({"stage":"training","variant":variant,"fold":fold,
                                      "epoch":epoch+1,"batch":batch,"batches":batches,
                                      "mean_loss":total/min(batch*1024,len(X))}),flush=True)
            losses.append(total/len(X))
        ckpt=OUT/f"{variant}_{fold}_s42_y2_128.pth"
        torch.save(model.state_dict(),ckpt)
        pred=predict(src,model,Xv,tv)
        result=evaluate(variant,fold,spec["validation"],pred,yv)
        result.update({"checkpoint":str(ckpt),"checkpoint_sha256":common.sha256(ckpt),
                       "weights":weight_info,"losses":losses,
                       "seconds":round(time.perf_counter()-started,1)})
        results["folds"][fold]=result
        common.save_json(OUT/f"{variant}_partial.json",results)
        print(json.dumps({"stage":"fold_complete","variant":variant,"fold":fold,
                          "base":result["base"],"candidate":result["candidate"],
                          "delta":result["delta"],"seconds":result["seconds"]}),flush=True)
        del X,y,tid,Xv,yv,tv,Y,W,weights,model,opt,pred
        gc.collect();torch.cuda.empty_cache()
    positives=sum(v["delta"]>=0 for fold in results["folds"].values()
                  for v in fold["daily"].values())
    results["daily_nonnegative"]=positives
    results["gate_passed"]=(all(results["folds"][f]["delta"]>=.008 for f in common.FOLDS)
                            and positives>=6)
    common.save_json(OUT/f"{variant}_results.json",results)
    print(json.dumps({"stage":"complete","variant":variant,
                      "gate_passed":results["gate_passed"],
                      "daily_nonnegative":positives}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant",choices=["weighted","normalized"],required=True)
    args=parser.parse_args()
    if common.sha256(common.DATA)!=common.EXPECTED_SHA:raise RuntimeError("Data SHA mismatch")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True,exist_ok=True)
    plan={"folds":common.FOLDS,"window":16,"seed":42,"epochs":2,
          "batch_size":1024,"lr":1e-4,"weight_decay":1e-4,
          "weighted":"train-day inverse y2 variance, capped 0.5..2, mean 1",
          "normalized":"causal per-stock EWMA(64) scale on f0..82, floor half global sigma, clip +/-6, f83..99 unchanged; global input stats train days 0..3",
          "base":"0.5*y1_market + 0.25*y2_raw128 + 0.25*y2_raw64",
          "candidate":"replace only raw128 y2 member",
          "gate":"pooled raw y2 gain >=.008 in both A/B and >=6/8 daily nonnegative",
          "outer_dates":[16,17,18,19],"outer_usage":"not for selection",
          "data_sha256":common.EXPECTED_SHA}
    path=OUT/"plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError("Plan mismatch")
    common.save_json(path,plan)
    if args.variant=="normalized":preprocess()
    os.chdir(REPO)
    sys.path[:0]=[str(REPO/"project_restart/src/models"),str(REPO/"project_restart"),
                 str(REPO/"project/project")]
    import gru_strong as src
    src.CFG["window"]=16
    torch.set_num_threads(4)
    run_variant(src,args.variant)

if __name__=="__main__":main()
