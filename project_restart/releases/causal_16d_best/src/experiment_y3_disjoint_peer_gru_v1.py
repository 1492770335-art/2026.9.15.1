"""Train two return heads directly with ordered historical blocks and learned cross-stock peer groups."""
import copy
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_paired_gross_flow_v1 as flow
import y3_disjoint_validation as trial

OUT=trial.clock.ART/'y3_disjoint_peer_gru_v1'
ARMS=['middle head','terminal head','combined','half combined','reversed history','permuted peers','own features only']
SECONDARY='last forty, middle twenty and final twenty stock-relative correlation'
statistics=trial.statistics


def state(raw):
    pressure,paired,move=flow.components(raw)
    cat=raw[:,:,83:].astype('float32')-1
    base=np.concatenate((pressure,paired,move,cat,np.abs(cat),np.tanh(raw[:,:,:1].astype('float32'))),2)
    n=len(raw)
    prefix=np.concatenate((np.zeros_like(base[:1]),base.cumsum(0,dtype=float)),axis=0)
    t=np.arange(n)+1
    out=np.empty((n,50,4,base.shape[-1]),dtype='float16')
    for i,(near,far) in enumerate([(64,192),(20,64),(4,20),(0,4)]):
        end=(t-near).clip(1);start=(t-far).clip(0)
        out[:,:,i]=((prefix[end]-prefix[start])/(end-start)[:,None,None]).clip(-6,6)
    return out


def observe(z,y,d):
    t=np.arange(d%9,len(y)-60,9)
    return z[t],trial.targets(y,t).clip(-3,3).astype('float16')


class PeerNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n,24),nn.LayerNorm(24),nn.SiLU())
        self.time=nn.GRU(24,24,batch_first=True)
        self.assignment=nn.Linear(24,6)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(72,32),nn.SiLU(),nn.Dropout(.18),nn.Linear(32,1)) for _ in range(2)])

    def forward(self,z,mode='normal'):
        x=self.encoder(z.float())
        if mode=='reverse':
            x=x.flip(2)
        h=self.time(x.flatten(0,1))[1][0].reshape(len(z),50,-1)
        assignment=self.assignment(h)
        peers=assignment.softmax(1).transpose(1,2) @ h
        v=assignment.softmax(2) @ peers
        if mode=='wrong':
            v=v.roll(17,1)
        elif mode=='own':
            v=torch.zeros_like(v)
        features=torch.cat([h,v,h-v],2)
        result=torch.cat([head(features) for head in self.heads],2)
        return result-result.mean(1,keepdim=True)


def fit(history):
    features,labels=zip(*history)
    x=torch.from_numpy(np.concatenate(features)).cuda()
    y=torch.from_numpy(np.concatenate(labels)).cuda()
    torch.manual_seed(22071)
    net=PeerNet(x.shape[-1]).cuda()
    avg=copy.deepcopy(net).eval()
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.06)
    for epoch in range(5):
        for ids in torch.randperm(len(x),device=x.device).split(48):
            truth=y[ids].float()
            pred=net(x[ids])
            # Downweight unusually volatile cross sections while retaining the scale of the returns.
            scale=truth.square().mean(1,keepdim=True).sqrt().clamp(.12,.8)
            loss=(F.smooth_l1_loss(pred,truth,beta=.7,reduction='none')/scale.sqrt()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),.7)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(avg.parameters()),list(net.parameters()),.035)
        if epoch in (2,3):
            opt.param_groups[0]['lr']*=.6
    return {k:v.cpu().numpy() for k,v in avg.state_dict().items()},dict(dates=len(history),frames=len(x))


@torch.inference_mode()
def changes(z,m):
    net=PeerNet(z.shape[-1]).cuda().eval()
    net.load_state_dict({k:torch.from_numpy(v) for k,v in m.items()})
    scores=[]
    for start in range(0,len(z),48):
        ids=np.arange(start,start+48).clip(max=len(z)-1)
        x=torch.from_numpy(z[ids]).cuda()
        n=min(48,len(z)-start)
        normal=net(x)
        rest=[net(x,k).sum(2) for k in ['reverse','wrong','own']]
        scores.append(torch.stack([normal[:,:,0],normal[:,:,1],normal.sum(2),.5*normal.sum(2),*rest],2)[:n].cpu().numpy())
    return list(np.moveaxis(.15*np.concatenate(scores),2,0))


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    eligible=[]
    for a,horizons in [(0,[0,1,2]),(1,[0,1,3]),(2,[0,1,2,3]),(3,[0,1,2,3])]:
        values=f[:,:,horizons,:][...,a:a+1]
        days=d[:,horizons,:][...,a:a+1]
        if trial.scoring.select(values,days) is not None:
            eligible.append(a)
    return int(max(eligible,key=lambda i:f[:,0,0,i].mean())) if eligible else None


def correction():
    return trial.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    trial.run(sys.modules[__name__])
