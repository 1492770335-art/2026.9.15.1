"""Forecast migration between observable book groups to allocate each group's forward return across stocks."""
import copy
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_conditional_book_sorts_v1 as study

parent,base,rank,history = study.parent,study.base,study.rank,study.history
OUT = parent.OUT.parent/'y3_book_group_migration_v1'
ARMS = ('unchanged','group-average distant prediction','reassigned future membership','migration allocation',
        'allocation beyond group average','full-horizon allocation above group average','migration above reassigned',
        'probability of staying with initial group','probability of leaving initial group','concordant full and far allocation')


def groups(x):
    v = x.float().reshape(len(x),50,5,16)
    values = torch.stack((v[:,:,0,4],v[:,:,1,0],v[:,:,1,11]-v[:,:,1,12],v[:,:,0,3]),1)
    _,a,b = study.order_bounds(values)
    edges = torch.arange(5,device=x.device)*10
    overlap = (torch.minimum(b[:,:,:,None],edges+10)-torch.maximum(a[:,:,:,None],edges)).clamp_min(0)
    return overlap/(b-a)[:,:,:,None]


def samples(d):
    raw = history.trial.clock.book.rawday(d)
    t = np.concatenate([np.arange(lo+(d*5)%10,hi-60,10) for lo,hi in base.segments(len(raw))])
    x,ph = rank.inputs.features(raw,t)
    future = rank.inputs.features(raw,t+20)[0]
    end = groups(torch.from_numpy(future).cuda()).cpu().numpy().astype('float16')
    y = history.trial.clock.labels(d).reshape(-1,50,3)[t]
    return x,ph,end,np.stack((y[:,:,2],y[:,:,2]-base.RATIO*y[:,:,1]),2).astype('float32')


class Migration(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(width,56),nn.LayerNorm(56),nn.SiLU(),nn.Linear(56,32),nn.SiLU())
        self.trans = nn.Sequential(nn.Linear(32+64+3,56),nn.LayerNorm(56),nn.SiLU(),nn.Dropout(.1),nn.Linear(56,20))
        self.pay = nn.Sequential(nn.Linear(32+64+3+4+5,56),nn.LayerNorm(56),nn.SiLU(),nn.Dropout(.1),nn.Linear(56,10))
        self.register_buffer('cells',torch.cat((torch.eye(4)[:,None].expand(-1,5,-1),torch.eye(5)[None].expand(4,-1,-1)),2))

    def forward(self,x,ph,current):
        h = self.enc(x.float())
        shared = torch.cat((h.mean(1),h.std(1,unbiased=False),ph),1)
        enc = torch.einsum('bksg,bsh->bkgh',current,h)/10
        hand = self.cells[None].expand(len(h),-1,-1,-1)
        pay = self.pay(torch.cat((enc,shared[:,None,None].expand(-1,4,5,-1),hand),3)).reshape(len(h),4,5,5,2)
        prob = self.trans(torch.cat((h,shared[:,None].expand(-1,50,-1)),2)).reshape(len(h),50,4,5).transpose(1,2).softmax(3)
        return prob,pay


def pieces(start,prob,pay):
    marginal = torch.einsum('bksg,bksj->bkgj',start,prob)/10
    own = torch.einsum('bksg,bksj,bkgjh->bsh',start,prob,pay)/4
    average = torch.einsum('bksg,bkgj,bkgjh->bsh',start,marginal,pay)/4
    return own,average


def fit(rows):
    x,ph,end,y = [torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*rows)]
    nets = []
    for matched in (False,True):
        torch.manual_seed(54731)
        net = Migration(x.shape[-1]).cuda()
        ema = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.055)
        for epoch in range(7):
            if epoch in (4,6):
                opt.param_groups[0]['lr'] *= .55
            for ids in torch.randperm(len(x),device='cuda').split(48):
                start = groups(x[ids])
                target = y[ids]-y[ids].mean(1,keepdim=True)
                future = end[ids].float()
                if not matched:
                    future = future.roll(13,2)
                prob,pay = net(x[ids],ph[ids],start)
                mass = torch.einsum('bksg,bksj->bkgj',start,future)
                truth = torch.einsum('bksg,bksj,bsh->bkgjh',start,future,target)/mass[:,:,:,:,None].clamp_min(.001)
                # A sparsely occupied transition contributes in proportion to its observed mass.
                price_error = F.smooth_l1_loss(pay,truth,beta=.4,reduction='none')*mass[:,:,:,:,None]
                own,_ = pieces(start,prob,pay)
                loss = price_error.mean()/2+.5*F.smooth_l1_loss(own,target,beta=.5)
                loss -= .35*(future*prob.clamp_min(1e-8).log()).sum(3).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.055)
        nets.append(ema)
    return nets


@torch.inference_mode()
def signals(raw,nets):
    x,ph = rank.inputs.features(raw,np.arange(len(raw)))
    out = []
    for i in range(0,len(x),64):
        v,s = [torch.from_numpy(w[i:i+64]).cuda() for w in (x,ph)]
        start = groups(v)
        a,b = nets[0](v,s,start)
        c,d = nets[1](v,s,start)
        wrong,wrong_avg = pieces(start,a,b)
        own,avg = pieces(start,c,d)
        diff = own-avg
        peer = wrong-wrong_avg
        diagonal = torch.eye(5,device=v.device)[None,None,:,:,None]
        keep,keep_avg = pieces(start,c,d*diagonal)
        stick = (keep-keep_avg)[:,:,1]
        full,far = diff[:,:,0],diff[:,:,1]
        agree = torch.where(full*far>0,full.sign()*torch.minimum(abs(full),abs(far)),0.)
        bank = (avg[:,:,1],peer[:,:,1],own[:,:,1],far,full,far-peer[:,:,1],stick,far-stick,agree)
        z = torch.stack(bank,2)
        out.append((z-z.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def correction():
    with np.load(OUT/'model.npz') as f:
        nets = [Migration(int(f['width'])).cuda().eval() for _ in range(2)]
        for i,n in enumerate(nets):
            n.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in n.state_dict()})
        w,pick = float(f['w']),int(f['pick'])
    return lambda raw:(signals(raw,nets)[:,:,pick]*w).astype('float32')


def layer():
    before,update = parent.layer(),correction()
    def apply(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw)
        return p
    return apply


def replay_with():
    update = correction()
    def read(day,raw):
        p,h,n = history.read(day,raw)
        p[:,:,2] += update(raw)
        h[:,:,2] += update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or study.train(sys.modules[__name__])):
    base.replay.replay(sys.modules[__name__])
