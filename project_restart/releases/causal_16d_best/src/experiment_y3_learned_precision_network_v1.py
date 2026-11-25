"""Train matched Y3 residual networks with and without risk estimates learned on disjoint dates."""
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import experiment_y3_ex_ante_error_routing_v1 as risk
import experiment_y3_rank_spacing_calibration_v1 as screen
import y3_pair_incumbent_validation as trial

OUT=trial.clock.ART/'y3_learned_precision_network_v1'
ARMS=['equal weight','learned precision','stock-shifted precision','precision training effect',
      'switch by risk','switch by risk in reverse','conditional specialization','half precision effect']
state=risk.state
merge_records=risk.merge_records


def inputs(z,p):
    riskx,q,conf=risk.design(z,p)
    scores=trial.clock.center(p)
    x=np.concatenate((scores, z, q[:,:,None]*z[:,:,16:19],
                      risk.filters.smooth(scores,16),risk.filters.smooth(scores,80),scores*np.tanh(2*scores)),2)
    return x,riskx


def record_with_state(z,p,y,d):
    t=np.arange(61+d%11,len(p)-60,11)
    x,rx=inputs(z,p)
    q=trial.clock.center(p[t,:,2])
    truth=trial.clock.center(y[t,:,2])
    beta=(q*truth).sum(dtype=float)/(q*q).sum(dtype=float).clip(1e-6)
    error=truth/np.clip(beta,.4,2.)-q
    return [x[t].astype('float16'),rx[t].astype('float16'),error.astype('float32')]


class Panel(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,24),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(48,24),nn.SiLU(),nn.Linear(24,1))

    def forward(self,x):
        h=self.local(x)
        y=self.head(torch.cat((h,h.mean(1,keepdim=True).expand_as(h)),2)).squeeze(2)
        return y-y.mean(1,keepdim=True)


def fit(hist):
    days=[[] for _ in hist]
    risks=[]
    for j in (0,1):
        rx=np.concatenate([rec[1] for rec in hist[1-j::2]]).astype('float32')
        y=np.concatenate([rec[2] for rec in hist[1-j::2]])
        w=risk.logistic(rx,(abs(y)>.8)[:,:,None])
        risks.append(w)
        for i in range(j,len(hist),2):
            days[i]=risk.risks(hist[i][1].astype('float32'),w)[:,:,0]
    device='cuda'
    pr=torch.as_tensor(np.concatenate(days),device=device)
    wt=1/(.08+pr)
    wt=(wt/wt.mean()).clamp(.25,2.5)
    xs=torch.as_tensor(np.concatenate([v[0] for v in hist]),device=device,dtype=torch.float32)
    ys=torch.as_tensor(np.concatenate([v[2] for v in hist]),device=device)
    params={'risk':np.mean(risks,0)}
    for name,weight in [('flat',torch.ones_like(wt)),('fit',wt),('wrong',wt.roll(17,1))]:
        torch.manual_seed(8341)
        net=Panel(xs.shape[2]).to(device)
        nn.init.zeros_(net.head[-1].bias)
        with torch.no_grad():net.head[-1].weight.mul_(.10)
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.05)
        for ep in range(5):
            for idx in torch.randperm(len(xs),device=device).split(40):
                pred=net(xs[idx])
                loss=(F.smooth_l1_loss(pred,ys[idx],beta=1.1,reduction='none')*weight[idx]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.6)
                opt.step()
            if ep==2:opt.param_groups[0]['lr']*=.45
        params.update({name+'_'+k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    return params


@torch.inference_mode()
def changes_with_reference(z,m,p):
    x,rx=inputs(z,p)
    heads=[]
    for name in ('flat','fit','wrong'):
        net=Panel(x.shape[2]).cuda().eval()
        net.load_state_dict({k[len(name)+1:]:torch.as_tensor(v) for k,v in m.items() if k.startswith(name+'_')})
        heads.append(np.concatenate([net(torch.as_tensor(chunk,device='cuda')).cpu().numpy()
                                    for chunk in np.array_split(x,(len(p)+63)//64)]))
    flat,fit,wrong=heads
    prob=risk.risks(rx,m['risk'])[:,:,0]
    gate=np.clip((prob-.10)/.35,0,1)
    switch=gate*flat+(1-gate)*fit
    reverse=gate*fit+(1-gate)*flat
    return [(.25*trial.clock.center(a)).astype('float32') for a in
            (flat,fit,wrong,fit-flat,switch,reverse,switch-.5*(flat+fit),.5*(fit-flat))]


def select(f,d):
    good=[]
    for i in (1,3,4,6,7):
        v=f[:,:,0,i]
        controls=(0,2) if i in (1,3,7) else (0,2,5)
        if ((v>0).all() and (d[:,0,i]>0).sum()>=4 and
            all((v-f[:,:,0,c]).mean(0).min()>0 for c in controls)):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    return trial.previous.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__])
