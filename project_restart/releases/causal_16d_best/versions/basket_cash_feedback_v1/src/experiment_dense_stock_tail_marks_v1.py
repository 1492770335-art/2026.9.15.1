"""Evaluate stock-level bulk, event polarity, and conditional size after removing the common shock."""
import json
import numpy as np
import torch
from torch.nn import functional as F

import experiment_dense_market_tail_marks_v1 as tail

reliable, basket, onset, rel, REFERENCE = tail.reliable, tail.basket, tail.onset, tail.rel, tail.REFERENCE
OUT = REFERENCE.parent / "dense_stock_tail_marks_v1"
ARMS = ("bulk_and_tail", "event_polarity", "conditional_severity")


def train(end):
    if not (tail.OUT/f"fit_{end}.pt").exists():
        tail.train(end)
    return np.array([end])


def read(end):
    pack = torch.load(tail.OUT/f"fit_{end}.pt",map_location="cpu",weights_only=True)
    net = tail.TailHead(len(pack["scale"])).double().eval()
    net.load_state_dict(pack["state"])
    return net, pack


@torch.inference_mode()
def values(x, grid, net, pack):
    pars = pack["risk"].numpy()[reliable.cells(x,grid),np.arange(50)]
    parts, odds = [], []
    for t in np.array_split(np.arange(len(x)),max(1,(len(x)+95)//96)):
        u = (torch.from_numpy(x[t]).double()/pack["scale"]).clamp(-6,6)
        logits = net(u)
        pr = logits[...,1:3].sigmoid()
        avg = pr*pack["prior"]
        excess = pr*F.softplus(logits[...,3:])
        freq = avg[...,1]-avg[...,0]
        skew = excess[...,1]-excess[...,0]
        z = torch.stack((logits[...,0]+skew,freq,skew-freq),axis=-1).numpy()
        parts.append(z*pars[t,:,1,None])
        odds.append(pr.numpy())
    z = np.concatenate(parts)
    return z-z.mean(1,keepdims=True), np.concatenate(odds)


def design(raw, grid, code, p, bank):
    net, pack = read(int(bank[0]))
    x = basket.features(raw,grid,code,p,pack["rate"].numpy())
    z,_ = values(x,grid,net,pack)
    return np.broadcast_to(z[...,None],(*z.shape,3))


def predict(raw, grid, p, params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


def audit():
    result = {}
    for end in (8,12):
        net,pack = read(end)
        metrics = []
        for day in range(end,end+4):
            x,y,grid = reliable.samples(day)
            truth = tail.labels(x,y,grid,pack["risk"].numpy())
            z,probs = values(x,grid,net,pack)
            event = abs(truth)>1
            rank = np.argsort(np.argsort(probs.sum(-1),axis=1),axis=1)
            # Compare sign discrimination within actual extremes to a non-informative constant score.
            odds = probs[...,1]/probs.sum(-1).clip(.001)
            positive = truth>1
            def pair_auc(v,labels):
                from scipy.stats import rankdata
                n = labels.sum()
                return (rankdata(v)[labels].sum()-n*(n+1)/2)/(n*(len(v)-n))
            metrics.append([event[rank<10].mean(),event[rank>=40].mean(),pair_auc(odds[event],positive[event]),
                            np.corrcoef((y-y.mean(1,keepdims=True)).ravel(),z[...,0].ravel())[0,1]])
        result[str(end+1)] = dict(low_high_event_rate_direction_auc_stock_ic=np.mean(metrics,axis=0).tolist(),daily=metrics)
    (OUT/"distribution_audit.json").write_text(json.dumps(result,indent=2))
    print({k:v["low_high_event_rate_direction_auc_stock_ic"] for k,v in result.items()},flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
    audit()
