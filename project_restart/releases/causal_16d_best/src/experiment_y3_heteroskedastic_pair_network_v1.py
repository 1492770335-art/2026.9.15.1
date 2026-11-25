"""Estimate coherent stock payoffs from comparisons with pair-specific uncertainty predicted on other dates."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_learned_precision_network_v1 as direct
import experiment_y3_rank_spacing_calibration_v1 as screen
import experiment_y3_score_neighborhood_heads_v1 as local
import y3_pair_incumbent_validation as trial

OUT=trial.clock.ART/'y3_heteroskedastic_pair_network_v1'
ARMS=['constant pair scale','individual uncertainty','stock-shifted uncertainty','conditional scale effect',
      'local ranking adjustment','local scale effect','half conditional effect','conditional effect beyond reassignment']
state=direct.state
merge_records=direct.merge_records


def record_with_state(z,p,y,d):
    x,rx,e=direct.record_with_state(z,p,y,d)
    ticks=np.arange(61+d%11,len(p)-60,11)
    order=np.argsort(p[ticks,:,2],axis=1,kind='stable')
    return [x,rx,e,order.astype('int16')]


def fit(hist):
    risks=[None]*len(hist)
    for j in (0,1):
        examples=hist[1-j::2]
        rx=np.concatenate([v[1] for v in examples]).astype('float32')
        target=np.concatenate([v[2] for v in examples])
        w=direct.risk.logistic(rx,(abs(target)>.8)[:,:,None])
        for k in range(j,len(hist),2):
            prob=direct.risk.risks(hist[k][1].astype('float32'),w)[:,:,0].clip(.018,.78)
            # Map a two-sided Laplace exceedance probability to a scale, calibrated with past errors only.
            risks[k]=-.8/np.log(prob)
    def tensor(groups,dtype=torch.float32):
        return torch.as_tensor(np.concatenate(groups),device='cuda',dtype=dtype)
    s=tensor(risks)
    x=tensor([v[0] for v in hist])
    err=tensor([v[2] for v in hist])
    order=tensor([v[3] for v in hist],torch.int64)
    cuts=s.new_tensor([-1.2,-.45,0,.45,1.2])
    weights={}
    for name,scale in [('flat',torch.full_like(s,s.mean().item())),('fit',s),('wrong',s.roll(17,1))]:
        torch.manual_seed(4549)
        net=direct.Panel(x.shape[-1]).cuda()
        with torch.no_grad():
            net.head[-1].bias.zero_()
            net.head[-1].weight.mul_(.1)
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.05)
        for ep in range(6):
            for rows in torch.randperm(len(x),device='cuda').split(48):
                rank=order[rows].reshape(-1,5,10)
                # All names use the same connected edges. Each stock is compared to its score neighbors.
                i=rank.flatten(1)
                j=rank.roll(1+(ep%4),dims=2).flatten(1)
                def diff(values):
                    return values.gather(1,i)-values.gather(1,j)
                delta=diff(net(x[rows]))
                actual=diff(err[rows])
                noise=(scale[rows].gather(1,i)**2+scale[rows].gather(1,j)**2).sqrt().clamp(.19,2.2)
                logits=(delta[:,:,None]-cuts)/noise[:,:,None]
                loss=F.binary_cross_entropy_with_logits(logits,(actual[:,:,None]>cuts).float())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.6)
                opt.step()
            if ep==3:opt.param_groups[0]['lr']*=.40
        weights.update({name+'_'+k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    return weights


@torch.inference_mode()
def changes_with_reference(z,m,p):
    x,_=direct.inputs(z,p)
    heads=[]
    for name in ('flat','fit','wrong'):
        net=direct.Panel(x.shape[-1]).cuda().eval()
        net.load_state_dict({k[len(name)+1:]:torch.as_tensor(v) for k,v in m.items() if k.startswith(name+'_')})
        heads.append(np.concatenate([net(torch.as_tensor(chunk,device='cuda')).cpu().numpy() for chunk in
                                    np.array_split(x,(len(p)+63)//64)]))
    flat,fit,wrong=heads
    local_fit,local_flat=np.moveaxis(local.residual(np.stack((fit,flat),2),p[:,:,2],p[:,:,2],5),2,0)
    return [.28*v for v in (flat,fit,wrong,fit-flat,local_fit,local_fit-local_flat,.5*(fit-flat),fit-wrong)]


def select(f,d):
    good=[]
    for i in (1,3,4,5,6,7):
        v=f[:,:,0,i]
        if ((v>0).all() and (d[:,0,i]>0).sum()>=4 and
            all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,2))):
            good.append(i)
    return int(max(good,key=lambda i:f[:,0,0,i].mean())) if good else None


def correction():
    return trial.previous.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    screen.run(sys.modules[__name__])
