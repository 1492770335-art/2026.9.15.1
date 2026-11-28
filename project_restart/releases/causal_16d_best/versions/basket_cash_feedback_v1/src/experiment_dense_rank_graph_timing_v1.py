"""Read the frozen peer graph at multiple causal lags and separate fresh, persistent, and reversed rank movements."""
import numpy as np
import torch
import experiment_dense_rank_graph_teacher_v1 as teacher
from prepare_dense_rank_graph_teacher_reference_v1 import OUT as REFERENCE

onset,rel=teacher.onset,teacher.rel
OUT=REFERENCE.parent/"dense_rank_graph_timing_v1"
ARMS=("fresh_above_uniform_motion","persistent_donor_motion","reversal_before_aggregation")


def train(end):
    return np.array([end])


def panel(raw,device):
    return torch.cat([teacher.middle.ranks(raw,t,device) for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+95)//96))])


def same(x,y):
    return torch.where(x*y>0,x.sign()*torch.minimum(x.abs(),y.abs()),0.)


def apply_graph(x,w):
    return x.transpose(1,2).reshape(len(x),5000).double()@w


def design(raw,grid,code,p,bank):
    device="cuda" if torch.cuda.is_available() else "cpu"
    with np.load(teacher.OUT/f"fit_{int(bank[0])}.npz") as f:
        w=torch.as_tensor(np.ascontiguousarray(f["w"][...,2].reshape(5000,50)),device=device)
    start=teacher.local.origins(grid)
    ranks=panel(raw,device)
    out=np.zeros((*p.shape[:2],3,3))
    for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+95)//96)):
        now,lag5,lag20,lag40=[ranks[np.maximum(start[t],t-j)] for j in (0,5,20,40)]
        late=now-lag5
        move=now-lag20
        previous=lag20-lag40
        fast=apply_graph(late-.25*move,w)
        persistent=apply_graph(same(move,previous),w)
        # Compare the price of reversals on individual edges with the reversal of the total received message.
        quiet=apply_graph(same(late,-(lag5-lag20)),w)
        agg=same(apply_graph(late,w),-apply_graph(lag5-lag20,w))
        z=torch.stack((fast,persistent,quiet-agg),dim=-1)
        z=z*(torch.as_tensor(t-start[t],device=device)[:,None,None]>=torch.tensor([20,40,20],device=device))
        out[t,:,:,2]=(z-z.mean(1,keepdim=True)).cpu().numpy()
    return out


def predict(raw,grid,p,params):
    q=p.copy()
    z=design(raw,grid,params["code"],p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z[...,2])@params["w"][:,2]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
