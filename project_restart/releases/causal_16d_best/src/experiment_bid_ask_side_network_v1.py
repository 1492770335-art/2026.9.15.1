"""Share a quote-response network across the bid and ask sides and supervise their future movements separately."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_quote_lattice_replenishment_v1 as lattice
import experiment_phase_future_book_v1 as phase
from experiment_crossfit_predictable_tails_v1 import run_validation,criterion

OUT=phase.inn.prior.ART/'bid_ask_side_network_v1'
RATIO=float(phase.inn.SCALE[2]/phase.inn.SCALE[1])
H=(5,20,60)
GEOM_KEYS=('tick','spacing','center','scale')


def frames(raw,g):
    sp,q,b,a,valid=lattice.quotes(raw,g);n=len(raw);t=np.arange(n)
    x=np.arcsinh(raw[:,:,lattice.FIELDS].astype('float32'))
    bu,au=np.split((x-g['center'])/g['scale'],2,axis=2)
    bu=np.tanh(bu*.72);au=np.tanh(au*.72)
    def field(i):return np.arcsinh(raw[:,:,i].astype('float32'))
    signed=np.stack((field(0),field(11),field(26),field(34),field(36),field(38),field(64),field(74)),axis=-1)
    shared=np.stack((np.log1p(sp),np.log(q.clip(.01,None)),valid,field(7),field(16),field(40),field(77),field(78)),axis=2)
    r=raw[:,:,0].astype(float)-.00136
    v=(.45+.55*np.sqrt((8+lattice.flow.trailing(r*r,96))/(8+np.minimum(t+1,96))[:,None])).clip(.55,3).astype('float32')
    features=[]
    for sign,move,other,sz,op in ((1,b,-a,bu,au),(-1,-a,b,au,bu)):
        last=np.maximum.accumulate(np.where(move!=0,t[:,None],0),0)
        age=np.log1p(t[:,None]-last)/3
        old=np.concatenate((sz[:1],sz[:-1]));positive=1.1+sz
        flow=np.where(move[:,:,None]>0,positive,np.where(move[:,:,None]<0,-(1.1+old),sz-old))*valid[:,:,None]
        reached=np.stack((np.tanh(move/2),np.tanh(other/2),(move>0).astype(float),(move<0).astype(float),age),axis=2)
        signal=np.concatenate((flow,sz-op,signed*sign,reached[:,:,:4]),axis=2).astype('float32')
        cs=np.concatenate((np.zeros_like(signal[:1],dtype=float),signal.cumsum(0,dtype=float)))
        parts=[signed*sign,sz,op,shared,reached.astype('float32'),flow]
        for lo,hi in ((0,5),(5,22),(22,85),(85,280)):
            end=(t+1-lo).clip(0);left=(t+1-hi).clip(0);count=(end-left).clip(1)[:,None,None]
            avg=(cs[end]-cs[left])/count
            parts.append(avg.astype('float32'))
        f=np.concatenate(parts,axis=2).clip(-8,8).astype('float16')
        features.append(f)
    return features,v,sp,q


def labels(y,sp,q,v,t):
    move=np.stack([(sp[t+h]-sp[t])*q[t]*phase.inn.SCALE[k] for k,h in enumerate(H)],axis=-1)
    mid=y[t]
    bid=(mid-move)/v[t,:,None];ask=(-mid-move)/v[t,:,None]
    def full(a):return np.concatenate((a,a[:,:,2:3]-RATIO*a[:,:,1:2]),axis=2)
    return np.stack((full(bid),full(ask)),axis=2).astype(np.float32)


class SideNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,64))
        self.stock=nn.Parameter(torch.zeros(50,64))
        self.eq=nn.Linear(64,18);self.ek=nn.Linear(64,18)
        self.oq=nn.Linear(64,18,bias=False);self.ok=nn.Linear(64,18,bias=False)
        self.values=nn.Linear(64,32)
        self.head=nn.Sequential(nn.Linear(64*3+32*2,100),nn.LayerNorm(100),nn.SiLU(),nn.Dropout(.14),nn.Linear(100,4*6))
        self.register_buffer('tick',torch.zeros(50));self.register_buffer('spacing',torch.zeros(2))
        self.register_buffer('center',torch.zeros(14));self.register_buffer('scale',torch.ones(14))

    def forward(self,xx,v):
        h=self.encoder(xx)+self.stock[None,:,None,:]
        odd=h[:,:,0]-h[:,:,1];even=h.mean(2)
        score=self.eq(even)@self.ek(even).transpose(1,2)+self.oq(odd)@self.ok(odd).transpose(1,2)
        attn=(score/6).softmax(2)
        z=self.values(h);peer=torch.einsum('bij,bjsd->bisd',attn,z)
        market=z.mean(1,keepdim=True).expand_as(z)
        ctx=torch.cat((h,h.flip(2),h-h.flip(2),peer,market),axis=-1)
        law=self.head(ctx).float().reshape(*h.shape[:3],4,6)
        logits=law[...,:3];part=law[...,3:]
        mean=torch.stack((-.20-F.softplus(part[...,0]),.20*part[...,1].tanh(),.20+F.softplus(part[...,2])),dim=-1)
        expected=(mean*logits.softmax(-1)).sum(-1)
        mid=(expected[:,:,0]-expected[:,:,1])*.5*v[:,:,None]
        p=torch.stack((mid[:,:,0],mid[:,:,1],.57*mid[:,:,2]+.43*(RATIO*mid[:,:,1]+mid[:,:,3])),axis=-1)
        return p,expected,logits,mean


def train(end):
    torch.manual_seed(3364);geo=lattice.geometry()
    ts=[np.r_[np.arange(d%2,600,2),np.arange(600+d%6,len(phase.inn.grid(d))-61,6)] for d in range(end)]
    n=sum(map(len,ts));start=0
    y=torch.empty(n,50,3,device='cuda');v=torch.empty(n,50,device='cuda')
    side=torch.empty(n,50,2,4,device='cuda',dtype=torch.float16)
    for d,t in enumerate(ts):
        raw=phase.inn.grid(d);xx,vol,sp,quantum=frames(raw,geo)
        if d==0:x=torch.empty(n,50,2,xx[0].shape[-1],device='cuda',dtype=torch.float16)
        sl=slice(start,start+len(t));start+=len(t);yy=phase.inn.prior.labels(d).reshape(-1,50,3)
        for j in (0,1):x[sl,:,j].copy_(torch.from_numpy(xx[j][t]))
        side[sl].copy_(torch.from_numpy(labels(yy,sp,quantum,vol,t)))
        y[sl].copy_(torch.from_numpy(yy[t]));v[sl].copy_(torch.from_numpy(vol[t]));del xx
    net=SideNet(x.shape[-1]).cuda()
    for k in GEOM_KEYS:getattr(net,k).copy_(torch.from_numpy(geo[k]))
    ema=SideNet(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.028)
    for ep in range(16):
        for ids in torch.randperm(n,device='cuda').split(40):
            yy=y[ids];target=side[ids].float();vol=v[ids]
            classes=(target>.20).long()-(target<-.20).long()+1
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):p,means,logits,cond=net(x[ids].float(),vol)
            chosen=cond.gather(-1,classes.unsqueeze(-1)).squeeze(-1)
            loss=.78*criterion(p.float(),yy,vol)+.45*F.smooth_l1_loss(means,target,beta=1.35)
            loss+=.25*F.cross_entropy(logits.reshape(-1,3),classes.reshape(-1))+.30*F.smooth_l1_loss(chosen,target,beta=1.6)
            loss-=.065*F.cosine_similarity(p-p.mean(1,keepdim=True),yy-yy.mean(1,keepdim=True),dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.017)
        if ep in (5,10):
            for group in opt.param_groups:group['lr']*=.54
        if ep in (0,7,15):print('training',end,ep+1,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()


@torch.inference_mode()
def infer(net,raw):
    g={k:getattr(net,k).cpu().numpy() for k in GEOM_KEYS};xx,v,_,_=frames(raw,g)
    out=np.empty((len(raw),50,3),np.float32);device=next(net.parameters()).device
    for start in range(0,len(raw),40):
        sl=slice(start,start+40);z=np.stack((xx[0][sl],xx[1][sl]),axis=2)
        out[sl]=net(torch.from_numpy(z).to(device).float(),torch.from_numpy(v[sl]).to(device))[0].cpu().numpy()
    return out


if __name__=='__main__':run_validation(OUT,train,lambda st:SideNet(st['encoder.0.weight'].shape[1]),infer)
