"""Learn a smooth future-return curve per stock from multiple terminal prices and factor portfolios."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_pooled_price_impact_v1 as impact
import experiment_endpoint_innovation_v1 as inn
import experiment_relative_residual_tree_v1 as tree
import experiment_peer_return_distillation_v1 as fits
import experiment_multiscale_return_net_v1 as v1
prior=inn.prior
OUT=prior.ART/'term_structure_net_v1'
H=np.array([1,2,3,5,8,12,16,20,30,45,60,90,120])
J=[3,7,10]
RATE=np.array([1.,3.,8.,20.,55.,160.])
BASIS=((1-np.exp(-H[:,None]/RATE))*np.sqrt(RATE)/np.sqrt(H[:,None])).astype(np.float32)
BOOK=[0,5,7,10,11,13,14,16,18,20,22,23,24,25,26,29,31,34,36,38,39,40,41,42,46,50,51,54,55,56,60,64,67,72,75,77,79,82,84,87,90,92,94,97,99]

def features(d,t):
    grid=inn.grid(d);feat=np.load(v1.OUT/f'features_{d}.npy',mmap_mode='r')
    return feature_rows(grid,feat,t)

def feature_rows(grid,feat,t):
    cur=feat[t,:,:100].astype(np.float32)
    parts=[cur]
    for off in (100,300,400,500,600):
        a=feat[t,:,off:off+100][:,:,BOOK].astype(np.float32)
        parts.append(a-cur[:,:,BOOK])
    parts.append(feat[t,:,700:].astype(np.float32))
    price=grid[:,:,0].astype(float)-.00136
    v=np.sqrt((6+inn.moving_sum(price**2,120))/(np.minimum(np.arange(len(grid))+1,120)+6)[:,None]).clip(.3,4)
    return np.concatenate(parts,axis=-1).astype(np.float16),v[t].astype(np.float32)

def targets(d,t):
    raw=inn.grid(d);r=raw[:,:,0].astype(float)-.00136;price=r.cumsum(0)
    out=np.stack([(price[t+h]-price[t])/np.sqrt(h) for h in H],axis=-1)
    return out.astype(np.float32)

class CurveNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.local=nn.Sequential(nn.Linear(n,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.12),nn.Linear(128,80),nn.SiLU())
        self.compress=nn.Linear(80,24)
        self.idio=nn.Sequential(nn.Linear(80+48,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.14),nn.Linear(80,6))
        self.market=nn.Sequential(nn.Linear(24*7,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.25),nn.Linear(80,6))
        self.register_buffer('basis',torch.from_numpy(BASIS.T.copy()))
        self.register_buffer('port',torch.zeros(5,50))
        self.register_buffer('beta',torch.ones(50))
    def forward(self,x,vol):
        h=self.local(x);z=self.compress(h)
        factors=self.port@z; avg=z.mean(1,keepdim=True)
        sec=self.port.T@factors
        loc=self.idio(torch.cat((h,avg.expand_as(z),sec),axis=-1))@self.basis
        loc=loc*vol[:,:,None];loc=loc-loc.mean(1,keepdim=True)
        shared=self.market(torch.cat((avg[:,0],z.std(1,unbiased=False),factors.flatten(1)),dim=1))@self.basis
        shared=shared*vol.mean(1,keepdim=True)
        return loc,shared

def train(end):
    torch.manual_seed(3841)
    xs=[];ys=[];vs=[]
    for d in range(end):
        t=np.arange((d*3+1)%5,len(inn.grid(d))-H[-1],5)
        x,vol=features(d,t);xs.append(torch.from_numpy(x));vs.append(torch.from_numpy(vol));ys.append(torch.from_numpy(targets(d,t)))
    x=torch.cat(xs).cuda();del xs
    y=torch.cat(ys).cuda();del ys
    vol=(.4+.6*torch.cat(vs).cuda()).clamp(.5,3);del vs
    net=CurveNet(x.shape[-1]).cuda()
    port,rv=impact.exposure(end);net.port.copy_(torch.from_numpy(port[-5:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(inn.base.ex.geometry(end)[0]))
    ema=CurveNet(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.025)
    for ep in range(1,15):
        for ids in torch.randperm(len(y),device='cuda').split(60):
            r,m=net(x[ids].float(),vol[ids]);truth=y[ids]
            ym=truth.mean(1);yr=truth-net.beta[None,:,None]*ym[:,None]
            error=(r-yr)/vol[ids,:,None]
            lm=F.smooth_l1_loss(m/vol[ids].mean(1,keepdim=True),ym/vol[ids].mean(1,keepdim=True),beta=.4)
            loss=F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.1)+lm*.55
            loss-=.075*F.cosine_similarity(r[:,:,J],yr[:,:,J],dim=1,eps=1e-6).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.02)
        if ep in (5,10):
            for group in opt.param_groups:group['lr']*=.6
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()

def infer(model,d):
    n=len(inn.grid(d));pred=np.empty((n,50,3),np.float32)
    with torch.inference_mode():
        for offset in range(4):
            t=np.arange(offset,n,4);a,v=features(d,t);seq=[]
            for k in range(0,len(t),80):
                z=torch.from_numpy(a[k:k+80]).cuda().float();vv=torch.from_numpy(v[k:k+80]).cuda()
                r,m=model(z,(.4+.6*vv).clamp(.5,3))
                seq.append((r+model.beta[None,:,None]*m[:,None,:])[:,:,J].cpu().numpy())
            pred[t]=np.concatenate(seq)*(np.sqrt(H[J])*v1.SCALE)
    return pred

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        model=train(end);rows=[]
        for d in range(end,stop):
            p=np.load(tree.OUT/f'pred_{d}.npy')
            n=infer(model,d);rows.append((p,prior.labels(d).reshape(p.shape),n))
        if end<16:early.extend(rows);_,w=fits.coefficients(early)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=v1.blend(p,n,w);np.save(OUT/f'side_{d}.npy',n);np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append(tuple(x.reshape(-1,3) for x in (p,q,y)))
        report[stop]={'w':w.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)
if __name__=='__main__':main()
