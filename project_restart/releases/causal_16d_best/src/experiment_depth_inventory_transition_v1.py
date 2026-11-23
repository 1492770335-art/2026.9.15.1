"""Learn three forward intervals through auxiliary targets for depth changes, executions and repricing."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_crossfit_predictable_tails_v1 as val
import experiment_execution_volume_clock_v1 as clock
import experiment_reconstructed_depth_flow_v1 as book
from experiment_causal_cross_horizon_ridge_v1 import ART,labels

OUT=ART/'depth_inventory_transition_v1'
H=(0,5,20,60)
S=(.559,.272,.161)


def geom():
    with np.load(book.OUT/'model.npz') as st:return {k:st[k].copy() for k in st.files if k not in ('market','relative')}


def features(raw,g):
    B,A,imb,D=book.depths(raw,g);buy,sell,qty,b,a,sp,q,_=clock.decode(raw,g)
    old=lambda z:np.vstack((z[:1],z[:-1]))
    r=raw[:,:,0].astype(float)-.00136;t=np.arange(len(raw))
    risk=(.45+.55*np.sqrt((8+clock.flow.trailing(r*r,96))/(8+np.minimum(t+1,96))[:,None])).clip(.55,3).astype('float32')
    factors=np.stack([np.arcsinh(raw[:,:,j].astype('float32')) for j in (10,11,14,20,21,22,23,24,25,26,27,28,29,30,31,34,36,38,58,60,64,70,74)],axis=2)
    streams=[];targets=[]
    for orient,size,other,hit,op,move,mate in ((1,B,A,sell,buy,b,-a),(-1,A,B,buy,sell,-a,b)):
        h=hit*g['amount_scale'];queued=size-old(size);frac=queued/(.01+size+old(size))
        denom=.015+(D+old(D))*.5
        marks=np.stack((h/denom,op*g['amount_scale']/denom,queued/denom,frac,
                       (queued+h)*(move==0)/denom,frac*(move!=0),np.tanh(move/2),np.tanh(mate/2),
                       h/(.01+old(size)),orient*imb,orient*r/risk),axis=2)
        common=np.stack((np.log(size+.01),np.log(other+.01),np.log(D+.01),np.log(q),
                         np.log1p(sp.clip(0)),np.log(risk),np.broadcast_to(np.exp(-t[:,None]/360),D.shape)),axis=2)
        m=np.arcsinh(marks).clip(-6,6);cs=np.concatenate((np.zeros_like(m[:1]),m.cumsum(0,dtype=float)));end=t+1
        pieces=[orient*factors,common,m]
        for lo,hi in ((0,6),(6,24),(24,80),(80,240)):
            last=(end-lo).clip(0);start=(end-hi).clip(0)
            pieces.append(np.arcsinh((cs[last]-cs[start])/np.sqrt((last-start).clip(1))[:,None,None]))
        streams.append(np.concatenate(pieces,axis=2).astype('float16'))
        # Positive and negative increments are split, so net change is preserved at each price spell.
        targets.append(np.stack((h,queued.clip(0),(-queued).clip(0),move.clip(0),(-move).clip(0)),axis=2))
    return streams,risk,D,targets


def observations(raw,g,t):
    x,risk,D,targets=features(raw,g);aux=[];mid=[]
    for s in targets:
        cs=np.concatenate((np.zeros_like(s[:1]),s.cumsum(0,dtype=float)))
        parts=[]
        for j,(a,b) in enumerate(zip(H,H[1:])):
            sums=cs[t+b+1]-cs[t+a+1]
            sums[:,:,:3]/=(D[t,:,None]+.025)*np.sqrt(b-a)
            sums[:,:,3:]/=np.sqrt(b-a)
            parts.append(np.log1p(sums.clip(0,1000)))
        aux.append(np.stack(parts,axis=2))
    return [z[t] for z in x],risk[t],np.stack(aux,axis=2).astype('float16')


class Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.init=nn.Sequential(nn.Linear(n,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.08),nn.Linear(80,48))
        self.stock=nn.Parameter(torch.zeros(50,48))
        self.port=nn.Parameter(torch.randn(5,50)*.12)
        self.message=nn.Linear(48,16);self.time=nn.Embedding(3,8)
        self.dynamics=nn.GRUCell(48+16+16+8+5,48)
        self.flux=nn.Sequential(nn.LayerNorm(48),nn.Linear(48,5))
        self.response=nn.Sequential(nn.Linear(48*2+16+5+8,64),nn.SiLU(),nn.Dropout(.11),nn.Linear(64,1))

    def forward(self,x,risk):
        n,s=x.shape[:2];h=self.init(x)+self.stock[None,:,None,:]
        amount=h.new_zeros(n,s,2,5);states=[];rewards=[]
        for k in range(3):
            msg=self.message(h);average=msg.mean(1,keepdim=True);m=average.expand_as(msg)
            w=(3*self.port).softmax(1);v=(3*self.port.T).softmax(1)
            portfolio=torch.einsum('ks,bspd->bkpd',w,msg-average)
            local=torch.einsum('sk,bkpd->bspd',v,portfolio)
            ctx=self.time.weight[k].expand(n,s,2,-1)
            z=torch.cat((h.flip(2),m,local,ctx,amount),-1).reshape(n*s*2,-1)
            h=self.dynamics(z,h.reshape(n*s*2,48)).reshape(n,s,2,48)
            amount=F.softplus(self.flux(h)).float();states.append(amount)
            price=self.response(torch.cat((h,h-h.flip(2),m,amount,ctx),-1)).squeeze(-1)
            rewards.append((price[:,:,0]-price[:,:,1])*.5*risk)
        a,b,c=rewards
        p=torch.stack((a,.272/.559*a+b,.161/.559*a+.161/.272*b+c),2)
        return p,torch.stack(states,3)


def fit(end):
    rng=np.random.default_rng(3851);torch.manual_seed(3851);g=geom()
    ts=[np.r_[np.arange(d%3,600,3),np.arange(600+d%5,len(clock.pair.rawday(d))-61,5)] for d in range(end)]
    size=sum(map(len,ts));idx=0
    v=torch.empty(size,50,device='cuda');aux=torch.empty(size,50,2,3,5,device='cuda',dtype=torch.float16)
    y=torch.empty(size,50,3,device='cuda')
    for d,t in enumerate(ts):
        z,r,targets=observations(clock.pair.rawday(d),g,t)
        if d==0:x=torch.empty(size,50,2,z[0].shape[-1],device='cuda',dtype=torch.float16)
        sl=slice(idx,idx+len(t));idx+=len(t)
        for k in range(2):x[sl,:,k].copy_(torch.from_numpy(z[k]))
        v[sl].copy_(torch.from_numpy(r));aux[sl].copy_(torch.from_numpy(targets));y[sl].copy_(torch.from_numpy(labels(d).reshape(-1,50,3)[t]))
    net=Net(x.shape[-1]).cuda();ema=Net(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.00078,weight_decay=.029)
    for ep in range(15):
        for ids in torch.randperm(size,device='cuda').split(42):
            real=y[ids];risk=v[ids]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):p,state=net(x[ids].float(),risk)
            score=val.criterion(p.float(),real,risk)
            late=p[:,:,2]-.161/.272*p[:,:,1];actual=real[:,:,2]-.161/.272*real[:,:,1]
            score+=.30*F.smooth_l1_loss((late-late.mean(1,keepdim=True))/risk,(actual-actual.mean(1,keepdim=True))/risk,beta=1.)
            score+=.5*F.smooth_l1_loss(state,aux[ids].float(),beta=.32)
            score-=.05*F.cosine_similarity(p-p.mean(1,keepdim=True),real-real.mean(1,keepdim=True),dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);score.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.016)
        if ep in (5,10):
            for group in opt.param_groups:group['lr']*=.52
        if ep in (0,7,14):print('inventory transition',end,ep+1,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()


@torch.inference_mode()
def infer(net,raw):
    z,v,*_=features(raw,geom());out=np.empty((len(raw),50,3),np.float32);dev=next(net.parameters()).device
    for i in range(0,len(raw),48):
        s=slice(i,i+48)
        out[s]=net(torch.from_numpy(np.stack((z[0][s],z[1][s]),2)).to(dev).float(),torch.from_numpy(v[s]).to(dev))[0].cpu().numpy()
    return out


def restore(st):return Net(st['init.0.weight'].shape[1])


if __name__=='__main__':val.run_validation(OUT,fit,restore,infer)
