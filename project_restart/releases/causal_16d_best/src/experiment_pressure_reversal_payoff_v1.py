"""Predict disjoint band returns before and after the first reversal of observed book pressure."""
import copy
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_pressure_spell_survival_v1 as spell
import experiment_repricing_payoff_heads_v1 as heads

OUT=spell.peer.ART/'pressure_reversal_payoff_v1'


def pressure(raw):
    parts=[np.tanh(np.arcsinh(raw[:,:,i].astype('float32'))-np.arcsinh(raw[:,:,j].astype('float32')))
           for i,j in ((41,42),(54,55),(77,78))]
    f=spell.trailing(np.mean(parts,axis=0),8)
    t=np.arange(len(f))[:,None]
    strong=np.maximum.accumulate(np.where(abs(f)>.10,t,-1),axis=0)
    sign=np.where(strong>=0,np.sign(np.take_along_axis(f,strong.clip(0),axis=0)),0)
    change=np.vstack((np.ones_like(sign[:1],bool),sign[1:]!=sign[:-1]))
    start=np.maximum.accumulate(np.where(change,t,0),axis=0)
    z=np.stack((f,sign,np.log1p(t-start).clip(0,6)/4),axis=2).astype('float32')
    return start,sign,z


def inputs(raw,t,enc,day=None,*,pressure_fn=pressure):
    x,v=heads.block.paths.frozen.encoding(raw,t,enc,day,fixed_batch=True)
    start,sign,z=pressure_fn(raw)
    x=torch.cat((x,torch.from_numpy(z[t]).to(x)),dim=2)
    return x,v,start,sign


def dataset(end,enc,*,pressure_fn=pressure):
    buffers=[[],[],[],[]]
    for d in range(end):
        raw=spell.peer.PARENT.before.book.rawday(d)
        t=np.arange(d%7,len(raw)-61,7)
        x,v,start,sign=inputs(raw,t,enc,d,pressure_fn=pressure_fn)
        r=raw[:,:,0].astype('float32')-.00136
        y=spell.bands(spell.peer.labels(d).reshape(-1,50,3)[t])
        pay=np.zeros((*y.shape,2),'float32')
        occupancy=np.zeros_like(y)
        price=r.cumsum(0,dtype=float)
        for k,(lo,hi) in enumerate(((0,5),(5,20),(20,60))):
            for step in range(lo+1,hi+1):
                on=(sign[t]!=0)&(start[t+step]==start[t])
                pay[:,:,k,0]+=on*r[t+step]*heads.UNITS[k]
                occupancy[:,:,k]+=on
            occupancy[:,:,k]/=hi-lo
            # Preserve the official band even if rounded price increments differ.
            reconstructed=(price[t+hi]-price[t+lo])*heads.UNITS[k]
            pay[:,:,k,0]+=occupancy[:,:,k]*(y[:,:,k]-reconstructed)
            pay[:,:,k,1]=y[:,:,k]-pay[:,:,k,0]
        np.testing.assert_allclose(pay.sum(3),y,atol=2e-5,rtol=1e-5)
        values=(x,v,torch.from_numpy(pay).cuda()/v[:,:,None,None],torch.from_numpy(occupancy).cuda())
        for buf,value in zip(buffers,values):buf.append(value)
    return [torch.cat(buf) for buf in buffers]


class ReversalHead(nn.Module):
    def __init__(self,width,*,shared_pressure=True):
        super().__init__()
        self.shared_pressure=shared_pressure
        self.enc=nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,40),nn.SiLU())
        self.decode=nn.Sequential(nn.Linear(120,56),nn.LayerNorm(56),nn.SiLU(),nn.Linear(56,9))

    def forward(self,x,gated):
        h=self.enc(x.float())
        common=h.mean(1,keepdim=True).expand_as(h)
        sd=h.std(1,unbiased=False,keepdim=True).expand_as(h)
        z=self.decode(torch.cat((h,common,sd),2)).reshape(len(x),50,3,3)
        frac=z[:,:,:,2].sigmoid()
        if self.shared_pressure:
            frac=frac.cumprod(2)
        mixture=torch.stack((frac,1-frac),3)
        rates=z[:,:,:,:2]
        pay=rates*mixture if gated else rates
        return pay.sum(3),rates,frac


def train(end,enc,seed=17013,*,data_fn=dataset,arms=(False,True),shared_pressure=True):
    x,v,pay,occupancy=data_fn(end,enc)
    truth=heads.levels(pay.sum(3)*v[:,:,None])
    nets=[]
    for gated in arms:
        torch.manual_seed(seed)
        net=ReversalHead(x.shape[-1],shared_pressure=shared_pressure).cuda()
        ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.035)
        for epoch in range(9):
            for i in torch.randperm(len(x),device='cuda').split(64):
                pred,rates,frac=net(x[i],gated)
                err=heads.levels(pred*v[i,:,None])-truth[i]
                market=err.mean(1,keepdim=True)
                local=(err-market)/v[i,:,None]
                market=market/v[i].mean(1)[:,None,None]
                loss=F.smooth_l1_loss(local,torch.zeros_like(local),beta=.9)
                loss+=.6*F.smooth_l1_loss(market,torch.zeros_like(market),beta=.55)
                loss+=.28*F.binary_cross_entropy(frac.clamp(1e-6,1-1e-6),occupancy[i])
                weight=torch.stack((occupancy[i],1-occupancy[i]),3)
                comp=rates*weight if gated else rates
                loss+=.45*F.smooth_l1_loss(comp,pay[i],beta=.75)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
                with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.024)
            if epoch in (4,7):
                for group in opt.param_groups:group['lr']*=.55
        nets.append(ema)
    return nets


@torch.inference_mode()
def corrections(raw,t,enc,nets):
    x,v,_,_=inputs(raw,t,enc)
    rows=[]
    for i in range(0,len(t),64):
        s=slice(i,i+64)
        d=nets[1](x[s],True)[0]-nets[0](x[s],False)[0]
        rows.append((.15*heads.levels(d*v[s,:,None])).cpu().numpy())
    q=np.concatenate(rows);m=q.mean(1,keepdims=True)
    return q-m,m


def screen():
    folds,daily=[],[]
    for end,stop in ((8,12),(12,14),(14,16)):
        enc=heads.block.paths.decoder.net(end)
        nets=train(end,enc)
        for begin in range(end,stop,2):
            rows=[[],[]]
            for d in (begin,begin+1):
                raw=spell.peer.PARENT.before.book.rawday(d)
                t=np.arange(d%4,len(raw),4)
                p=spell.peer.reference(d)[t];y=spell.peer.labels(d).reshape(-1,50,3)[t]
                z=corrections(raw,t,enc,nets)
                if d==begin:
                    n=181
                    for a,b in zip(corrections(raw[:n],t[t<n],enc,nets),z):
                        np.testing.assert_allclose(a,b[:len(a)],rtol=0,atol=3e-5)
                for g,v in zip(rows,z):g.append(tuple(w.reshape(-1,3) for w in (p,p+v,y)))
                daily.append([heads.linear.gains(g[-1:]) for g in rows])
            folds.append([heads.linear.gains(g) for g in rows])
            print(begin+1,folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        np.save(OUT/'keep.npy',keep)
        for k,m in enumerate(train(16,heads.block.paths.decoder.net(16))):
            torch.save(m.cpu().state_dict(),OUT/f'head_{k}.pth')
    return keep.any()


def load(parent=None):
    if parent is None:
        parent=spell.peer.PARENT.load()
    enc=heads.block.paths.decoder.net(16)
    keep=np.load(OUT/'keep.npy')
    nets=[]
    for k in range(2):
        st=torch.load(OUT/f'head_{k}.pth',map_location='cpu',weights_only=True)
        net=ReversalHead(st['enc.0.weight'].shape[1]).cuda().eval()
        net.load_state_dict(st)
        nets.append(net)
    def predict(raw,p):
        q=parent(raw,p).copy()
        for z,k in zip(corrections(raw,np.arange(len(raw)),enc,nets),keep):q+=z*k
        return q
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    if not screen():return
    base=json.loads((spell.peer.PARENT.OUT/'reload_verification.json').read_text())['pooled']['score']
    first=None
    for name in ('raw_replay','reload_verification'):
        res=spell.peer.walk.replay(load())
        if first is not None:
            np.testing.assert_allclose(res['pooled']['candidate'],first['pooled']['candidate'],atol=1e-10,rtol=0)
            for d in res['daily']:
                assert abs(res['daily'][d]['score']-first['daily'][d]['score'])<1e-10
        (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
        print(name,res['pooled']['score'],flush=True)
        if res['pooled']['score']<=base:break
        first=res


if __name__=='__main__':main()
