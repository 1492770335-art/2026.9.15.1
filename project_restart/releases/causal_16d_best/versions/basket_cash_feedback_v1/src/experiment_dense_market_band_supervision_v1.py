"""Train a market head on local clock-band contrasts of predictions and labels inside the same session."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_dense_market_clock_views_v1 as clock
import experiment_dense_joint_strip_likelihood_v1 as dist

strip,rel,onset,REFERENCE,read_p=dist.strip,dist.rel,dist.onset,dist.REFERENCE,dist.read_p
OUT=REFERENCE.parent/'dense_market_band_supervision_v1'
ARMS=('band_trained_above_pointwise','twenty_to_sixty','sixty_to_two_forty')
W=clock.DIM
DEVICE=dist.DEVICE


def design(current,hist):
    v=hist.astype(np.float32).mean(1)
    return np.stack((current,v[:,-2:].mean(1),v[:,-8:-2].mean(1),v[:,:-8].mean(1)),axis=1)


class Market(nn.Module):
    def __init__(self):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(W,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(96,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,3))

    def forward(self,x):
        v=self.state(x)
        return self.head(v.flatten(-2))


def contrast(x,small,big):
    # Each element is a child-window mean minus its own parent-window mean.
    child=x.reshape(len(x),-1,small,3).mean(2)
    parent=child.reshape(len(x),-1,big//small,3).mean(2)
    return child-parent.repeat_interleave(big//small,dim=1)


def dataset(end):
    xs=[];ys=[];starts=[];offset=0
    for day in range(end):
        grid=onset.settled.load(day,'time')
        with np.load(clock.OUT/f'day_{day:02d}.npz') as f:
            x=design(f['c'],f['h'])
            y=f['y']
        xs.append(x);ys.append(y)
        total=0
        for a,b in rel.fields.parent.sessions(grid):
            ticks=np.arange(a+day%5,b,5)
            # Every label of a 240-tick training panel ends before this session closes.
            valid=np.where(ticks+60<b)[0]
            if len(valid)>=48:
                starts.extend(offset+total+valid[:len(valid)-47])
            total+=len(ticks)
        assert total==len(x)
        offset+=total
    x,y=[torch.from_numpy(np.concatenate(z)).to(DEVICE).float() for z in (xs,ys)]
    sx=x[::3].square().mean((0,1)).sqrt().clamp_min(.06)
    sy=y.std(0).clamp_min(.08)
    return (x/sx).clamp(-5,5),y/sy,torch.as_tensor(starts,device=DEVICE),sx,sy


def train(end):
    x,y,starts,sx,sy=dataset(end)
    packed=[sx.cpu().numpy(),sy.cpu().numpy()]
    rows=torch.arange(48,device=DEVICE)
    for band in (False,True):
        torch.manual_seed(62871)
        net=Market().to(DEVICE)
        opt=torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.07)
        for step in range(370):
            t=starts[torch.randint(len(starts),(28,),device=DEVICE)][:,None]+rows
            pred=net(x[t]); target=y[t]
            point=F.smooth_l1_loss(pred,target,beta=.85)
            if band:
                changes=sum(F.smooth_l1_loss(contrast(pred,s,b),contrast(target,s,b),beta=.55)
                            for s,b in ((4,12),(12,48)))
                loss=.4*point+.6*changes
            else:
                loss=point
            if step==255:
                opt.param_groups[0]['lr']*=.4
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        packed.extend(w.detach().cpu().numpy().ravel() for w in net.parameters())
    return np.concatenate(packed)


def avg(z,width):
    cs=np.vstack((np.zeros(z.shape[1]),z.cumsum(0)))
    end=np.arange(1,len(z)+1);start=(end-width).clip(0)
    return (cs[end]-cs[start])/(end-start)[:,None]


@torch.inference_mode()
def signals(raw,grid,p,bank):
    unit=torch.from_numpy(bank[:W]).double()
    sy=bank[W:W+3]
    pos=W+3
    nets=[]
    for _ in range(2):
        m=Market().double().eval()
        for w in m.parameters():
            w.copy_(torch.from_numpy(bank[pos:pos+w.numel()].reshape(w.shape)))
            pos+=w.numel()
        nets.append(m)
    assert pos==len(bank)
    out=np.empty((len(p),3,2))
    for a,b in rel.fields.parent.sessions(grid):
        s=clock.summary(raw[a:b])
        values=[]
        for start in range(0,len(s),64):
            t=np.arange(start,min(start+64,len(s)))
            z=design(s[t].astype(np.float16),clock.histories(s,t))
            x=(torch.from_numpy(z).double()/unit).clamp(-5,5)
            v=(nets[1](x)-nets[0](x)).numpy()*sy
            values.append(v[:,1:])
        delta=np.concatenate(values)
        out[a:b]=np.stack((delta,avg(delta,20)-avg(delta,60),avg(delta,60)-avg(delta,240)),axis=1)
    return np.broadcast_to(out[:,None],(*p.shape[:2],3,2))


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
