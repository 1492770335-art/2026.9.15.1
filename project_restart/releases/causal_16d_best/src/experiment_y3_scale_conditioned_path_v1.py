"""Fine-tune the return curve with a forecast of future movement magnitude at each horizon."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import y3_expanding_path_references as history

base=history.incumbent
path,trial=base.path,base.trial
OUT=trial.clock.ART/'y3_scale_conditioned_path_v1'
ARMS=['frozen orientation','normalized direction','direction times own magnitude','magnitude adjustment',
      'direction and magnitude after frozen path','wrong stock magnitude','finetuning increment','half magnitude',
      'direction times common magnitude']


class MagnitudePath(path.Path):
    def __init__(self,n):
        super().__init__(n)
        self.risk=nn.Linear(32,3)
        nn.init.zeros_(self.risk.weight)
        nn.init.zeros_(self.risk.bias)

    def forward(self,x,phi):
        h=self.path(self.patch(x.float()).flatten(2))
        market=h.mean(1,keepdim=True).expand_as(h)
        hidden=self.read[:2](torch.cat((h,market),2))
        direction=self.read[-1](hidden)@phi
        log_scale=(self.risk(hidden)@phi).clamp(-1,1.5)
        return direction-direction.mean(1,keepdim=True),log_scale


def scale_targets(raw,t,offs):
    squared=(raw[:,:,0].astype(float)-.00136)**2
    cs=np.vstack((np.zeros(50),squared.cumsum(0)))
    end=offs+t[:,None]+21
    start=end-20
    rms=np.sqrt(np.maximum(cs[end]-cs[start],0)/20)
    return (.4+.6*rms).clip(.4,4).astype('float32').transpose(0,2,1)


@lru_cache(maxsize=4)
def train(end):
    frozen=base.weights(end)
    data=[]
    offsets=base.dense.OFFSETS
    for d in range(end):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        t=np.arange(132+d%11,len(raw)-60,11)
        f=trial.clock.center(y[t[:,None]+offsets,:,1].transpose(0,2,1))
        b=trial.clock.center(-y[t[:,None]-84-offsets,:,1].transpose(0,2,1))
        data.append((path.patches(path.stream(raw),t),f,b,scale_targets(raw,t,offsets),
                     scale_targets(raw,t,-84-offsets)))
    x,fy,by,fs,bs=[torch.from_numpy(np.concatenate(v)).cuda() for v in zip(*data)]
    del data
    torch.manual_seed(8343)
    net=MagnitudePath(44).cuda()
    net.load_state_dict({k[6:]:torch.from_numpy(v) for k,v in frozen.items()},strict=False)
    optimizer=torch.optim.AdamW([{'params':net.patch.parameters(),'lr':.00010},
                                 {'params':net.path.parameters(),'lr':.00018},
                                 {'params':net.read.parameters(),'lr':.00024},
                                 {'params':net.risk.parameters(),'lr':.0011}],weight_decay=.05)
    phi=torch.from_numpy(base.dense.PHI).cuda()
    def loss(inputs,target,observed):
        direction,log_scale=net(inputs,phi)
        adjusted=direction*log_scale.exp()
        adjusted=adjusted-adjusted.mean(1,keepdim=True)
        return (F.smooth_l1_loss(direction,target/observed,beta=.85)+
                .55*F.smooth_l1_loss(adjusted,target,beta=.85)+
                .65*F.smooth_l1_loss(log_scale,observed.log(),beta=.30))
    for ep in range(4):
        for ids in torch.randperm(len(x),device='cuda').split(48):
            objective=loss(x[ids],fy[ids],fs[ids])+.7*loss(path.reverse(x[ids],True),by[ids],bs[ids])
            optimizer.zero_grad()
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(),.8)
            optimizer.step()
        if ep==2:
            for param in optimizer.param_groups:
                param['lr']*=.5
    return {**frozen, **{'fine/'+k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}}


@torch.inference_mode()
def encode(raw,m):
    s=path.stream(raw)
    old=path.Path(44).cuda().eval()
    old.load_state_dict({k[6:]:torch.from_numpy(v) for k,v in m.items() if k.startswith('dense/')})
    net=MagnitudePath(44).cuda().eval()
    net.load_state_dict({k[5:]:torch.from_numpy(v) for k,v in m.items() if k.startswith('fine/')})
    phi=torch.from_numpy(base.dense.PHI[:,[4,8]].copy()).cuda()
    rows=[]
    for t in np.array_split(np.arange(len(raw)),(len(raw)+95)//96):
        x=torch.from_numpy(path.patches(s,t)).cuda()
        rev=path.reverse(x,True)
        prev=.5*((old(x)-old(rev))@phi)
        a,ra=net(x,phi)
        b,rb=net(rev,phi)
        va,vb=ra.exp(),rb.exp()
        direction=.5*(a-b)
        own=.5*(a*va-b*vb)
        wrong=.5*(a*va.roll(17,1)-b*vb.roll(17,1))
        market=.5*(a*va.mean(1,keepdim=True)-b*vb.mean(1,keepdim=True))
        features=torch.cat((prev,direction,own,own-direction,wrong,direction-prev,market),2)
        rows.append((features-features.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(rows)


def historical_state(raw,d):
    return encode(raw,train(4*(d//4)))


record_with_state=base.record_with_state
changes_with_reference=base.changes_with_reference


def warm_history():
    rows=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        rows.append(record_with_state(historical_state(raw,d),path.early.load(d,raw),y,d))
    return rows


def fit(rows):
    gs,bs=[np.stack(v).astype(float) for v in zip(*rows)]
    gram=gs.mean(0)
    def solve(cols,n):
        basis=np.eye(len(gram))[:,cols]
        basis[:n]=-trial.curve.solve(gram[:n,:n],gram[:n,cols],.045)
        day=bs@basis
        mu=day.mean(0)
        mu*=abs(mu)/(abs(mu)+day.std(0)/np.sqrt(len(rows))+1e-7)
        return basis@trial.curve.solve(basis.T@gram@basis,mu,.9)
    effect=solve([9,10],7)
    bank=[solve([3,4],3),solve([5,6],3),solve([7,8],3),effect,solve([5,6,9,10],5),
          solve([11,12],3),solve([13,14],5),.5*effect,solve([15,16],3)]
    result=dict(w=(.40*np.column_stack(bank)).astype('float32'))
    if len(rows)==12:
        result.update(train(16))
    return result


def select(f,d):
    good=[]
    for j in (1,2,3,4,6,7,8):
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and
                (d[:,0,j]>0).sum()>=5 and f[:,:,1,j].mean(0).min()>0 and
                (v-f[:,:,0,5]).mean(0).min()>0):
            good.append(j)
    return max(good,key=lambda j:f[:,0,0,j].mean()) if good else None


def correction():
    with np.load(OUT/'model.npz') as f:
        model=dict(f)
    return lambda raw,p:changes_with_reference(encode(raw,model),model,p)[int(model['pick'])]


def layer():
    predict,update=base.layer(),correction()
    def apply(raw,p):
        q=predict(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    history.run(sys.modules[__name__])
