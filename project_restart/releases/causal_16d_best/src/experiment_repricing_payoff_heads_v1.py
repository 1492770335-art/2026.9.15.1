"""Predict the probability and signed contribution of successive price changes in each horizon band."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_cross_date_response_shapes_v1 as linear
import experiment_block_error_whitening_v1 as block
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'repricing_payoff_heads_v1'
UNITS=linear.SCALE.astype('float32')
BANDS=((0,5),(5,20),(20,60))


def training_targets(raw,t,y,v,cuts=None):
    if cuts is None:
        cuts=np.broadcast_to(np.arange(1,4),(50,3,3))
    delta=raw[:,:,0].astype('float32')-.00136
    bands=y.copy()
    bands[:,:,1:]-=y[:,:,:2]*(UNITS[1:]/UNITS[:-1])
    values=np.zeros((*bands.shape,4),np.float32)
    present=np.zeros((*bands.shape,3),bool)
    for j,(start,stop) in enumerate(BANDS):
        count=np.zeros(bands.shape[:2],np.int32)
        for k in range(start+1,stop+1):
            change=delta[t+k]
            hit=np.abs(change)>.022
            count+=hit
            for m in range(3):
                slot=hit&(count>=cuts[:,j,m])
                if m<2:slot &= count<cuts[:,j,m+1]
                values[:,:,j,m]+=change*slot*UNITS[j]
        present[:,:,j]=count[:,:,None]>=cuts[:,j,:]
        # Tiny changes and official target scaling are accounted for in the final component.
        values[:,:,j,3]=bands[:,:,j]-values[:,:,j,:3].sum(2)
    np.testing.assert_allclose(values.sum(3),bands,atol=1e-5,rtol=2e-5)
    return torch.from_numpy(values).cuda()/v[:,:,None,None],torch.from_numpy(present).cuda()


def data(end,enc,cuts=None):
    rows=[[],[],[],[]]
    for d in range(end):
        raw=block.paths.book.rawday(d)
        t=np.arange(d%7,len(raw)-61,7)
        x,v=block.paths.frozen.encoding(raw,t,enc,d,fixed_batch=True)
        y=labels(d).reshape(-1,50,3)[t].astype('float32')
        pay,occur=training_targets(raw,t,y,v,cuts)
        for buf,value in zip(rows,(x,v,pay,occur)):buf.append(value)
    return tuple(torch.cat(v) for v in rows)


class RepricingHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,40),nn.SiLU())
        self.decode=nn.Sequential(nn.Linear(120,56),nn.LayerNorm(56),nn.SiLU(),nn.Linear(56,21))

    def forward(self,x,conditional):
        h=self.enc(x.float())
        m=h.mean(1,keepdim=True).expand_as(h)
        sd=h.std(1,keepdim=True,unbiased=False).expand_as(h)
        z=self.decode(torch.cat((h,m,sd),dim=2)).reshape(len(x),50,3,7)
        # P(at least k events) = product of continuation probabilities.
        probability=z[:,:,:,4:].sigmoid().cumprod(3).clamp(1e-5,1-1e-5)
        amount=z[:,:,:,:4]
        term=amount[:,:,:,:3]*probability if conditional else amount[:,:,:,:3]
        return term.sum(3)+amount[:,:,:,3],probability,amount


def levels(bands):
    units=bands.new_tensor(UNITS)
    return (bands/units).cumsum(2)*units


def train(end,enc,cuts=None,arms=(False,True)):
    x,v,pay,occur=data(end,enc,cuts)
    true=levels(pay.sum(3)*v[:,:,None])
    nets=[]
    for conditional in arms:
        torch.manual_seed(13829)
        net=RepricingHead(x.shape[-1]).cuda()
        ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.035)
        for ep in range(10):
            for i in torch.randperm(len(x),device='cuda').split(64):
                forecast,prob,amount=net(x[i],conditional)
                risk=v[i,:,None]
                residual=levels(forecast*risk)-true[i]
                m=residual.mean(1,keepdim=True)
                local=(residual-m)/risk
                market=m/risk.mean(1,keepdim=True)
                loss=F.smooth_l1_loss(local,torch.zeros_like(local),beta=.9)
                loss+=.60*F.smooth_l1_loss(market,torch.zeros_like(market),beta=.55)
                loss+=.25*F.binary_cross_entropy(prob,occur[i].float())
                each=F.smooth_l1_loss(amount,pay[i],reduction='none',beta=.8)
                pieces=each[:,:,:,:3]
                if conditional:pieces=pieces*occur[i]
                loss+=.38*pieces.mean()+.10*each[:,:,:,3].mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
                with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.024)
            if ep in (5,8):
                for g in opt.param_groups:g['lr']*=.5
        nets.append(ema)
    return nets


@torch.inference_mode()
def changes(raw,t,enc,nets,arms=(False,True)):
    x,v=block.paths.frozen.encoding(raw,t,enc,fixed_batch=True)
    out=[]
    for start in range(0,len(t),64):
        sl=slice(start,start+64)
        q=nets[1](x[sl],arms[1])[0]-nets[0](x[sl],arms[0])[0]
        out.append(levels(q*v[sl,:,None]).cpu().numpy())
    value=np.concatenate(out)
    common=value.mean(1,keepdims=True)
    return value-common,common


def load(out=OUT,arms=(False,True),parent=None):
    if parent is None:parent=linear.best.load()
    enc=block.paths.decoder.net(16)
    mask=np.load(out/'mask.npy')
    nets=[]
    for k in (0,1):
        s=torch.load(out/f'net_{k}.pth',map_location='cpu',weights_only=True)
        net=RepricingHead(s['enc.0.weight'].shape[1]).cuda().eval()
        net.load_state_dict(s)
        nets.append(net)
    def predict(raw,p):
        q=parent(raw,p).copy()
        for z,k in zip(changes(raw,np.arange(len(raw)),enc,nets,arms),mask):q+=.15*z*k
        return q
    return predict


def main(out=OUT,trainer=train,arms=(False,True)):
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    folds,daily=[],[]
    for end,stop in ((8,12),(12,14),(14,16)):
        enc=block.paths.decoder.net(end)
        nets=trainer(end,enc)
        for begin in range(end,stop,2):
            rows=[[],[]]
            for d in (begin,begin+1):
                raw=block.paths.book.rawday(d)
                t=np.arange(d%4,len(raw),4)
                y=labels(d).reshape(-1,50,3)[t]
                p=np.load(block.segment.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r')[t].copy()
                p[:,:,1]=np.load(linear.earlier.OUT/f'y2_day_{d}.npy',mmap_mode='r')[t]
                sides=changes(raw,t,enc,nets,arms)
                if d==begin:
                    early=t[t<177]
                    for a,b in zip(changes(raw[:177],early,enc,nets,arms),sides):
                        np.testing.assert_allclose(a,b[:len(early)],atol=4e-5,rtol=0)
                for rs,z in zip(rows,sides):
                    rs.append(tuple(a.reshape(-1,3) for a in (p,p+.15*z,y)))
                daily.append([linear.gains(rs[-1:]) for rs in rows])
            folds.append([linear.gains(rs) for rs in rows])
            print(begin+1,'local and market IC gains',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    (out/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        np.save(out/'mask.npy',keep)
        enc=block.paths.decoder.net(16)
        for k,net in enumerate(trainer(16,enc)):
            torch.save(net.cpu().state_dict(),out/f'net_{k}.pth')
        base=json.loads((linear.best.OUT/'reload_verification.json').read_text())['pooled']['score']
        for name in ('raw_replay','reload_verification'):
            res=linear.best.online.best.roll.gls.best.walk.replay(load(out,arms))
            if name=='reload_verification':
                old=json.loads((out/'raw_replay.json').read_text())
                np.testing.assert_allclose(res['pooled']['candidate'],old['pooled']['candidate'],rtol=0,atol=1e-10)
            (out/f'{name}.json').write_text(json.dumps(res,indent=2))
            print(name,res['pooled']['score'],flush=True)
            if res['pooled']['score']<=base:break


if __name__=='__main__':main()
