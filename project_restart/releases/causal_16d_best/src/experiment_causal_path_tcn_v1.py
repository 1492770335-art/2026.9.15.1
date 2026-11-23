"""Train a causal temporal convolution on complete stock frames and overlapping target horizons."""
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_multiscale_return_net_v1 as v1
import experiment_peer_return_distillation_v1 as peer
from experiment_causal_return_basket_v1 import basket
prior=v1.prior
OUT=prior.ART/'causal_path_tcn_v1'
DILATIONS=(1,2,4,8,16,32)
HISTORY=2*sum(DILATIONS)
SUPERVISED=96
EPOCHS=(2,5,9,14)
FEATURES=list(range(100))+list(range(300,400))+list(range(500,600))+list(range(700,713))


class TemporalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.dilation=dilation
        self.norm=nn.LayerNorm(width)
        self.linear=nn.Linear(width,2*width)
        self.convolution=nn.Conv1d(width,width,3,dilation=dilation,groups=width)
        self.out=nn.Linear(width,width)
        self.drop=nn.Dropout(.1)

    def forward(self,h):
        b,t,s,d=h.shape
        v,gate=self.linear(self.norm(h)).chunk(2,-1)
        v=v.permute(0,2,3,1).reshape(b*s,d,t)
        v=self.convolution(F.pad(v,(2*self.dilation,0)))
        v=v.reshape(b,s,d,t).permute(0,3,1,2)
        return h+self.drop(self.out(F.silu(v)*torch.sigmoid(gate)))


class PathNet(nn.Module):
    def __init__(self):
        super().__init__()
        w=48
        self.encoder=nn.Sequential(nn.Linear(len(FEATURES),w),nn.LayerNorm(w),nn.SiLU())
        self.blocks=nn.ModuleList([TemporalBlock(w,d) for d in DILATIONS])
        self.register_buffer('graph',torch.eye(50))
        self.message=nn.Linear(w,16)
        self.mix=nn.Linear(32,w)
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,4))
        self.register_buffer('sigma',torch.from_numpy(v1.SCALE.copy()))
        self.register_buffer('lengths',torch.tensor([1.,2.,math.sqrt(15.),math.sqrt(40.)]))

    def forward(self,x):
        h=self.encoder(x)
        for i,block in enumerate(self.blocks):
            h=block(h)
            if i==2:
                v=self.message(h)
                m=v.mean(2,keepdim=True)
                h=h+self.mix(torch.cat((m.expand_as(v),self.graph@v-m),dim=-1))
        scale=(.55+.45*(x[:,:,:,-6:-5].exp()-.05)).clamp(.6,3.2)
        bands=self.head(h)*scale
        return (bands*self.lengths).cumsum(-1)[:,:,:,1:]*self.sigma,bands


def extract(d):
    x=np.load(v1.OUT/f'features_{d}.npy',mmap_mode='r')
    # Take features in consecutive runs to keep the time-major layout contiguous.
    return np.concatenate((x[:,:,:100],x[:,:,300:400],x[:,:,500:600],x[:,:,700:713]),axis=-1)


def forward_blocks(model,x):
    model.eval()
    preds=[]; device=next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0,len(x),SUPERVISED):
            t=np.maximum(0,np.arange(start-HISTORY,min(len(x),start+SUPERVISED)))
            v=torch.from_numpy(x[t]).to(device).float().unsqueeze(0)
            p=model(v)[0][0,HISTORY:]
            preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def train(end,epochs):
    torch.manual_seed(329); rng=np.random.default_rng(329)
    data=[torch.from_numpy(extract(d)) for d in range(end)]
    ys=[torch.from_numpy(prior.labels(d).reshape(-1,50,3)) for d in range(end)]
    segs=[torch.from_numpy(v1.targets(y.numpy(),v1.grid(d))) for d,y in enumerate(ys)]
    net=PathNet().cuda(); net.graph.copy_(torch.from_numpy(basket(end)).float())
    ema=PathNet().cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.015)
    ticks=np.arange(-HISTORY,SUPERVISED)
    for ep in range(1,max(epochs)+1):
        seqs=[(d,start) for d in range(end) for start in range(rng.integers(0,SUPERVISED),len(data[d])-SUPERVISED-1,SUPERVISED)]
        rng.shuffle(seqs)
        for offset in range(0,len(seqs),4):
            batch=seqs[offset:offset+4]
            x=torch.stack([data[d][np.maximum(0,k+ticks)] for d,k in batch]).cuda().float()
            y=torch.stack([ys[d][k:k+SUPERVISED] for d,k in batch]).cuda()
            target=torch.stack([segs[d][k:k+SUPERVISED] for d,k in batch]).cuda()
            vol=(.5+.5*(x[:,HISTORY:,:,-6:-5].exp()-.05)).clamp(.6,2.8)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                total,bands=net(x)
            pred=total[:,HISTORY:].float()
            loss=F.smooth_l1_loss(pred/vol,y/vol,beta=1.2)+.25*F.smooth_l1_loss(bands[:,HISTORY:].float()/vol,target/vol,beta=1.2)
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.025)
        if ep in (4,9):
            for g in opt.param_groups:g['lr']*=.55
        if ep in epochs:torch.save(ema.state_dict(),OUT/f'net_{end}_{ep}.pth')
        print('train',end,ep,float(loss),flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    plan=json.loads((v1.OUT/'results.json').read_text())['16']
    early={ep:[] for ep in EPOCHS};report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        epochs=EPOCHS if end<16 else sorted(set(chosen.tolist()))
        if not all((OUT/f'net_{end}_{ep}.pth').exists() for ep in epochs):
            train(end,epochs)
        base=[peer.baseline(d,plan) for d in range(end,stop)]
        rows={ep:[] for ep in epochs}
        for d,p in zip(range(end,stop),base):
            x=extract(d)
            y=prior.labels(d).reshape(p.shape)
            for ep in epochs:
                net=PathNet().cuda().eval();net.load_state_dict(torch.load(OUT/f'net_{end}_{ep}.pth',weights_only=True))
                n=forward_blocks(net,x)
                if d==end and ep==epochs[0]:
                    np.testing.assert_allclose(n[:135],forward_blocks(net,x[:135]),atol=4e-5,rtol=4e-5)
                rows[ep].append((p,y,n))
        if end<16:
            best=np.full(3,-np.inf);chosen=np.zeros(3,int);weights=np.zeros((2,3))
            for ep in epochs:
                early[ep].extend(rows[ep]);corr,w=peer.coefficients(early[ep])
                selected=corr>best;best=np.maximum(best,corr);chosen[selected]=ep;weights[:,selected]=w[:,selected]
        entries=[]
        for i,d in enumerate(range(end,stop)):
            p,y,_=rows[epochs[0]][i]
            side=np.stack([rows[ep][i][2][:,:,j] for j,ep in enumerate(chosen)],axis=-1)
            q=v1.blend(p,side,weights)
            np.save(OUT/f'prediction_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'epochs':chosen.tolist(),'weights':weights.tolist(),'result':prior.measure(entries)}
        print('score',stop,report[stop],flush=True)
        (OUT/'results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
