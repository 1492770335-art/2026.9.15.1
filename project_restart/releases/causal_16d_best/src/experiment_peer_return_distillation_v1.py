"""Volatility-scaled peer network with lower-variance training targets for future return bands."""
import json
import math
import numpy as np
import torch
from torch import nn

import experiment_multiscale_return_net_v1 as v1
from experiment_causal_return_basket_v1 import basket
from cache_opening_walkforward import load
prior=v1.prior
OUT=prior.ART/'peer_return_distillation_v1'
EPOCHS=[4,8,14,22]


class PeerNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.register_buffer('graph',torch.eye(50))
        self.register_buffer('scales',torch.tensor([1.,2.,math.sqrt(15),math.sqrt(40)]))
        self.register_buffer('target_scales',torch.from_numpy(v1.SCALE.copy()))
        self.local=nn.Sequential(nn.Linear(n,144),nn.LayerNorm(144),nn.SiLU(),nn.Dropout(.12),nn.Linear(144,96))
        self.peer=nn.Sequential(nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,32))
        self.query=nn.Linear(96,16)
        self.key=nn.Linear(96,16)
        self.value=nn.Linear(96,32)
        self.mix=nn.Sequential(nn.Linear(128,96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,96))
        self.head=nn.Sequential(nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.18),nn.Linear(96,4))

    def forward(self,x):
        vol=(.3+.7*(x[:,:,-6:-5].exp()-.05)).clamp(.5,3.5)
        h=self.local(x)
        msg=self.peer(h)
        att=(self.query(h)@self.key(h).transpose(-1,-2)/4.).softmax(-1)@self.value(h)
        avg=msg.mean(1,keepdim=True).expand_as(msg)
        context=torch.cat((self.graph@msg,att,avg,msg-avg),dim=2)
        v=self.head(h+self.mix(context))*vol
        return v,(v*self.scales).cumsum(-1)[:,:,1:]*self.target_scales


def load_teacher(end):
    path=v1.OUT/f'net_{end}_10.pth'
    net=v1.ReturnNet(713).cuda().eval()
    net.load_state_dict(torch.load(path,weights_only=True))
    return net


def prepared(end,teacher):
    selections=[np.arange(d%3,len(v1.grid(d))-1,3) for d in range(end)]
    total=sum(map(len,selections))
    data=torch.empty((total,50,713),device='cuda',dtype=torch.float16)
    targets=torch.empty((total,50,4),device='cuda')
    base_targets=torch.empty_like(targets[:,:,:3])
    pos=0
    for d,ticks in enumerate(selections):
        x=v1.grid(d); y=prior.labels(d).reshape(-1,50,3)
        feather=np.load(v1.OUT/f'features_{d}.npy',mmap_mode='r')
        with torch.inference_mode():
            next5=v1.predict(d,[teacher])[0,:,:,0]/v1.SCALE[0]
        # At training time, replace part of the unpredictable future innovations
        # in bins 6..20 and 21..60 with the teacher's expected five-step returns.
        sums=[]
        for start,stop in [(5,20),(20,60)]:
            idx=ticks[:,None]+np.arange(start,stop,5)
            valid=idx[:,-1]+5<len(x)
            pred=next5[np.minimum(idx,len(x)-1)].sum(1)/np.sqrt(stop-start)
            sums.append((valid,pred))
        seg=v1.targets(y,x)[ticks].copy()
        for j,(valid,pred) in enumerate(sums,2):
            seg[valid,:,j]=.50*seg[valid,:,j]+.50*pred[valid]
        sl=slice(pos,pos+len(ticks));pos+=len(ticks)
        data[sl].copy_(torch.from_numpy(feather[ticks]))
        targets[sl].copy_(torch.from_numpy(seg))
        base_targets[sl].copy_(torch.from_numpy(y[ticks]))
    return data,targets,base_targets


def train(end,epochs):
    torch.manual_seed(944)
    teacher=load_teacher(end)
    data,labels,y=prepared(end,teacher)
    del teacher
    model=PeerNet(713).cuda();model.graph.copy_(torch.from_numpy(basket(end)).float())
    ema=PeerNet(713).cuda();ema.load_state_dict(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.025)
    for ep in range(1,max(epochs)+1):
        for ids in torch.randperm(len(y),device='cuda').split(48):
            x=data[ids].float()
            vol=(.3+.7*(x[:,:,-6:-5].exp()-.05)).clamp(.55,3.5)
            seg,total=model(x)
            segment_loss=nn.functional.smooth_l1_loss(seg/vol,labels[ids]/vol,beta=1.)
            cum_loss=nn.functional.smooth_l1_loss(total/vol,y[ids]/vol,beta=1.)
            loss=.55*segment_loss+.45*cum_loss
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.012)
        if ep in (5,13):
            for g in opt.param_groups:g['lr']*=.55
        if ep in epochs:
            torch.save(ema.state_dict(),OUT/f'net_{end}_{ep}.pth')
        print('train',end,ep,round(float(loss),4),flush=True)


@torch.inference_mode()
def predict(day,model):
    feature=np.load(v1.OUT/f'features_{day}.npy',mmap_mode='r')
    items=[]
    for i in range(0,len(feature),96):
        x=torch.from_numpy(np.array(feature[i:i+96])).cuda().float()
        items.append(model(x)[1].cpu().numpy())
    return np.concatenate(items)


def coefficients(rows):
    # Compute empirical correlations of any mixture by its second moments.
    p,y,n=map(np.concatenate,zip(*rows))
    d=n-p;m=np.broadcast_to(d.mean(1,keepdims=True),d.shape)
    x=np.stack((p,m,d-m),axis=-1).reshape(-1,3,3).astype(float)
    x-=x.mean(axis=0)
    yy=y.reshape(-1,3).astype(float);yy-=yy.mean(axis=0)
    gram=np.einsum('nkj,nki->kji',x,x);rhs=np.einsum('nkj,nk->kj',x,yy)
    variance=(yy**2).sum(0)
    best=np.full(3,-np.inf); weights=np.zeros((2,3))
    for a in [0.,.1,.25,.4,.6,.8,1.]:
        for b in [0.,.1,.25,.4,.6,.8,1.]:
            w=np.array([1,a,b]);corr=(rhs@w)/np.sqrt(variance*np.einsum('j,kji,i->k',w,gram,w))
            improve=corr>best;best=np.maximum(best,corr)
            weights[:,improve]=[[a],[b]]
    return best,weights


def baseline(d,v1plan):
    end=12 if d<14 else (14 if d<16 else 16)
    p=np.asarray(load(d))
    ep=v1plan['epochs']; strength=np.array(v1plan['weights'])
    if d>=16:return np.load(v1.OUT/f'pred_{d}.npy')
    nets=[]
    for k in sorted(set(ep)):
        model=v1.ReturnNet(713).cuda().eval()
        model.load_state_dict(torch.load(v1.OUT/f'net_{end}_{k}.pth',weights_only=True))
        nets.append(model)
    pred=v1.predict(d,nets)
    ordered=sorted(set(ep));side=np.stack([pred[ordered.index(k),:,:,j] for j,k in enumerate(ep)],axis=-1)
    return v1.blend(p,side,strength).astype(np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    v1plan=json.loads((v1.OUT/'results.json').read_text())['16']
    report={}; early={ep:[] for ep in EPOCHS}
    for end,stop in [(12,14),(14,16),(16,20)]:
        epochs=EPOCHS if end<16 else sorted(set(chosen.tolist()))
        if not all((OUT/f'net_{end}_{ep}.pth').exists() for ep in epochs):
            train(end,epochs)
        rows={ep:[] for ep in epochs}
        base=[baseline(d,v1plan) for d in range(end,stop)]
        for ep in epochs:
            net=PeerNet(713).cuda().eval()
            net.load_state_dict(torch.load(OUT/f'net_{end}_{ep}.pth',weights_only=True))
            for d,p in zip(range(end,stop),base):
                pred=predict(d,net)
                rows[ep].append((p,prior.labels(d).reshape(p.shape),pred))
            if end<16:early[ep].extend(rows[ep])
        if end<16:
            best=np.full(3,-np.inf);weights=np.zeros((2,3));chosen=np.zeros(3,int)
            for ep in epochs:
                corr,w=coefficients(early[ep]);select=corr>best
                weights[:,select]=w[:,select];chosen[select]=ep;best=np.maximum(best,corr)
        entries=[]
        for i,d in enumerate(range(end,stop)):
            p,y,_=rows[epochs[0]][i]
            side=np.stack([rows[ep][i][2][:,:,j] for j,ep in enumerate(chosen)],axis=-1)
            q=v1.blend(p,side,weights)
            if end==16:np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        result=prior.measure(entries)
        report[str(stop)]={'epochs':chosen.tolist(),'weights':weights.tolist(),'result':result}
        print('score',stop,report[str(stop)],flush=True)
        (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
