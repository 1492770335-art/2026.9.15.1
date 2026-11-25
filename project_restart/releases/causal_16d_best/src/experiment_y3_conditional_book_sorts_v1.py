"""Predict high-minus-low portfolio payoffs conditional on a second contemporaneous book characteristic."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_joint_rank_payout_v1 as rank
import y3_angular_forecast_references as history

parent,base = history.incumbent,history.base
OUT = parent.OUT.parent/'y3_conditional_book_sorts_v1'
ARMS = ('unchanged','shared spread across five groups','reassigned stock interaction','conditional spread',
        'within-group interaction','conditional minus shared model','session-wide far and early interaction',
        'agreement of full and far interactions','tail groups only','interior groups only')
GROUPS = 5


def order_bounds(v):
    sorted_values,order = v.sort(dim=2)
    # Ties in either characteristic share portfolio weights exactly.
    lower = torch.searchsorted(sorted_values,v.contiguous(),right=False)
    upper = torch.searchsorted(sorted_values,v.contiguous(),right=True)
    return order,lower,upper


def weights(x):
    v = x.float().reshape(len(x),50,5,16)
    first = torch.stack((v[:,:,2,0],v[:,:,3,3],v[:,:,2,7],v[:,:,3,11]),1)
    second = torch.stack((v[:,:,0,4],v[:,:,1,0],v[:,:,1,11]-v[:,:,1,12],v[:,:,0,3]),1)
    _,a,b = order_bounds(first)
    edges = torch.arange(GROUPS,device=x.device)*10
    overlap = (torch.minimum(b[:,:,:,None],edges+10)-torch.maximum(a[:,:,:,None],edges)).clamp_min(0)
    share = overlap/(b-a)[:,:,:,None]
    idx,left,right = order_bounds(second)
    by_value = share.gather(2,idx[:,:,:,None].expand_as(share))
    cumulative = F.pad(by_value.cumsum(2),(0,0,1,0))
    low = cumulative.gather(2,left[:,:,:,None].expand_as(share))
    high = cumulative.gather(2,right[:,:,:,None].expand_as(share))
    below = ((torch.minimum(high,torch.full_like(high,5))-low).clamp_min(0)/(high-low).clamp_min(1e-8)).clamp(0,1)
    signed = share*(1-2*below)
    return share,signed


class SpreadNet(nn.Module):
    def __init__(self,width,mode):
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.read = nn.Sequential(nn.Linear(24*4+4+5+3,56),nn.LayerNorm(56),nn.SiLU(),nn.Dropout(.12),nn.Linear(56,3))
        self.register_buffer('sort_id',torch.eye(4)[None,:,None,:].expand(1,4,5,4))
        self.register_buffer('group_id',torch.eye(5)[None,None,:,:].expand(1,4,5,5))

    def forward(self,x,ph,a,z):
        h = self.enc(x.float())
        avg = torch.einsum('bksg,bsh->bkgh',a,h)/10
        contrast = torch.einsum('bksg,bsh->bkgh',z,h)/10
        if self.mode==0:
            avg = avg.mean(2,keepdim=True).expand_as(avg)
            contrast = contrast.mean(2,keepdim=True).expand_as(contrast)
        common = torch.cat((h.mean(1),h.std(1,unbiased=False),ph),1)[:,None,None].expand(-1,4,5,-1)
        group = self.group_id.expand(len(h),-1,-1,-1)
        if self.mode==0:
            group = torch.zeros_like(group)
        return self.read(torch.cat((avg,contrast,common,self.sort_id.expand(len(h),-1,-1,-1),group),3))


def fit(rows):
    x,ph,y = [torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*rows)]
    # Keep masks on the GPU for only the minibatch being optimized.
    nets = []
    for mode in (0,1):
        torch.manual_seed(53829)
        net = SpreadNet(x.shape[-1],mode).cuda()
        ema = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.06)
        for epoch in range(7):
            if epoch in (4,6):
                opt.param_groups[0]['lr'] *= .55
            for idx in torch.randperm(len(y),device='cuda').split(48):
                v = x[idx]
                share,signed = weights(v)
                target = torch.einsum('bksg,bsh->bkgh',signed,y[idx])/10
                pred = net(v,ph[idx],share,signed)
                loss = F.smooth_l1_loss(pred,target,beta=.2)+.5*F.smooth_l1_loss(pred.sum(3),target.sum(3),beta=.38)
                if mode:
                    loss += .4*F.smooth_l1_loss(pred-pred.mean(2,keepdim=True),target-target.mean(2,keepdim=True),beta=.2)
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
    pieces = []
    for i in range(0,len(x),64):
        v,s = [torch.from_numpy(t[i:i+64]).cuda() for t in (x,ph)]
        a,z = weights(v)
        torch.testing.assert_close(a.sum(2),torch.full_like(a.sum(2),10),atol=3e-5,rtol=0)
        assert z.sum(2).abs().max()<3e-5
        shared,full = [net(v,s,a,z) for net in nets]
        local = full-full.mean(2,keepdim=True)
        def payout(m,mask=z):
            return torch.einsum('bksg,bkgh->bsh',mask,m)/4
        pooled,conditional,inter = [payout(table) for table in (shared,full,local)]
        wrong = payout(local,z.roll(13,2))
        early = inter[:,:,:2].sum(2)
        far = inter[:,:,2]
        agreed = torch.where(early*far>0,far.sign()*torch.minimum(early.abs(),far.abs()),0)
        tails = payout(local[:,:,[0,4]],z[:,:,:,[0,4]])[:,:,2]
        cols = (pooled[:,:,2],wrong[:,:,2],conditional[:,:,2],far,(conditional-pooled)[:,:,2],
                inter.sum(2),agreed,tails,far-tails)
        pieces.append(torch.stack(cols,2).cpu().numpy())
    return np.concatenate(pieces)


samples = rank.samples


def train(module=None):
    module = module or sys.modules[__name__]
    out,arms = module.OUT,module.ARMS
    out.mkdir(exist_ok=True)
    rows,cal,folds,daily = [],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            nets = module.fit(rows)
            print('fitted through day',d,flush=True)
            if d>=8:
                w = base.calibrate(cal)
                group = []
        if d>=4:
            raw = history.trial.clock.book.rawday(d)
            y = history.trial.clock.labels(d).reshape(-1,50,3)
            ps = history.references(d,raw) if d>=8 else parent.training_references(d,raw)
            z = module.signals(raw,nets)
            if d==4:
                for n in (125,2445):
                    np.testing.assert_allclose(z[:n],module.signals(raw[:n],nets),atol=2e-6,rtol=0)
            pair = [base.records(p,y,z) for p in ps]
            cal.append(pair)
            if d>=8:
                group.append(pair)
                daily.append([base.score(m,w)[0] for m in pair])
                if d in (11,15):
                    folds.append([base.score(m,w) for m in np.sum(group,0)])
        rows.append(module.samples(d))
    f,ds = np.asarray(folds),np.asarray(daily)
    ok = [i for i in range(3,len(arms)) if f[:,:,:,i].min()>0 and (ds[:,:,i]>0).sum(0).min()>=5 and
          (f[:,:,0,i].mean(0)>f[:,:,0,1:3].mean(0).max(1)).all()]
    pick = max(ok,key=lambda i:f[:,:,0,i].min()) if ok else 0
    (out/'results.json').write_text(json.dumps(dict(arms=arms,selected=pick,folds=f.tolist(),daily=ds.tolist(),windows=[[9,12],[13,16]]),indent=2))
    print('selected',arms[pick],flush=True)
    if pick:
        nets = module.fit(rows)
        state = {str(i)+'_'+k:v.cpu().numpy() for i,net in enumerate(nets) for k,v in net.state_dict().items()}
        np.savez(out/'model.npz',w=base.calibrate(cal)[pick-1],pick=pick-1,width=rows[0][0].shape[2],**state)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        nets = [SpreadNet(int(f['width']),i).cuda().eval() for i in (0,1)]
        for i,net in enumerate(nets):
            net.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in net.state_dict()})
        w,pick = float(f['w']),int(f['pick'])
    return lambda raw:(signals(raw,nets)[:,:,pick]*w).astype('float32')


def layer():
    before,update = parent.layer(),correction()
    def predict(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw)
        return p
    return predict


def replay_with():
    update = correction()
    def read(day,raw):
        p,h,n = history.read(day,raw)
        p[:,:,2] += update(raw)
        h[:,:,2] += update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
