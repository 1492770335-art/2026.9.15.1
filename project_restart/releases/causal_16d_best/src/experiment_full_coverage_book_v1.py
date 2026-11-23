"""Cycle training endpoints through every admissible tick while retaining a single GPU-sized shard per epoch."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_variance_ratio_calibration_v1 as variance

OUT=phase.inn.prior.ART/'full_coverage_book_v1'
CACHE=OUT/'training_features'

def prepare(d):
    if d>=16:raise ValueError('Training feature cache only accepts dates 0 through 15')
    root=CACHE/f'day_{d:02d}'
    if (root/'complete').is_file():return
    root.mkdir(parents=True,exist_ok=True)
    raw=phase.inn.grid(d);n=len(raw)-61
    files={}
    for k in range(3):
        t=np.arange(k,n,3);x,v=phase.spline.features(raw,t);a=phase.auxiliary(raw,t)
        if k==0:
            for key,sample in [('x',x),('v',v),('a',a)]:
                files[key]=np.lib.format.open_memmap(root/f'{key}.npy',mode='w+',dtype=sample.dtype,
                                                      shape=(n,*sample.shape[1:]))
        for key,sample in [('x',x),('v',v),('a',a)]:files[key][t]=sample
    for obj in files.values():obj.flush()
    (root/'complete').write_text(str(n))
    print('prepared training date',d+1,flush=True)


def train(end):
    torch.manual_seed(281)
    caches=[]
    for d in range(end):
        prepare(d);root=CACHE/f'day_{d:02d}'
        caches.append(tuple(np.load(root/f'{key}.npy',mmap_mode='r') for key in ('x','v','a')))
    max_n=sum(300+(len(row[0])-600+5)//6 for row in caches)
    width=caches[0][0].shape[-1]
    x=torch.empty(max_n,50,width,device='cuda',dtype=torch.float16)
    v=torch.empty(max_n,50,device='cuda');y=torch.empty(max_n,50,3,device='cuda')
    aux=torch.empty(max_n,50,3*len(phase.OBS),device='cuda',dtype=torch.float16);g=torch.empty(max_n,1,device='cuda')
    net=phase.BookNet(width).cuda();net.port.copy_(torch.from_numpy(phase.impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(phase.inn.base.ex.geometry(end)[0]))
    ema=phase.BookNet(width).cuda();ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.025)
    coverage=[np.zeros(len(c[0]),dtype=bool) for c in caches]
    for ep in range(16):
        at=0
        for d,(fx,fv,fa) in enumerate(caches):
            ticks=np.r_[np.arange((d+ep)%2,600,2),np.arange(600+(d+ep)%6,len(fx),6)]
            coverage[d][ticks]=True
            sl=slice(at,at+len(ticks));at+=len(ticks)
            x[sl].copy_(torch.from_numpy(fx[ticks]));aux[sl].copy_(torch.from_numpy(fa[ticks]))
            y[sl].copy_(torch.from_numpy(phase.inn.prior.labels(d).reshape(-1,50,3)[ticks]))
            v[sl].copy_(torch.from_numpy((.45+.55*fv[ticks]).clip(.55,3.)))
            g[sl].copy_(torch.from_numpy(np.exp(-ticks[:,None]/400).astype('float32')))
        for ids in torch.randperm(at,device='cuda').split(48):
            yy=y[ids];vol=v[ids];ym=yy.mean(1);yr=yy-net.beta[None,:,None]*ym[:,None]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                p,delta,r,m=net(x[ids].float(),vol,g[ids])
            loss=F.smooth_l1_loss(r/vol[:,:,None],yr/vol[:,:,None],beta=1.25)
            loss+=.6*F.smooth_l1_loss(m/vol.mean(1,keepdim=True),ym/vol.mean(1,keepdim=True),beta=.45)
            loss+=.35*F.smooth_l1_loss(delta.float(),aux[ids].float(),beta=.6)
            loss-=.06*F.cosine_similarity(r,yr,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.016)
        if ep+1 in (6,11):
            for group in opt.param_groups:group['lr']*=.55
        if ep+1 in (1,6,16):print('training',end,ep+1,'coverage',round(sum(c.sum() for c in coverage)/sum(map(len,coverage)),4),flush=True)
    assert all(c.all() for c in coverage)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()

def main():
    OUT.mkdir(exist_ok=True);report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=True);net=phase.BookNet(state['current.0.weight'].shape[1]).cuda().eval()
            net.load_state_dict(state)
        else:net=train(end)
        rows=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);n=phase.infer(net,raw)
            if d==end:np.testing.assert_allclose(n[:157],phase.infer(net,raw[:157]),atol=1e-4)
            p=variance.baseline(d);y=phase.inn.prior.labels(d).reshape(p.shape)
            rows.append((p,y,n));np.save(OUT/f'side_{d}.npy',n)
        if end==12:_,w=phase.fits.coefficients(rows);w=w.clip(0,.2)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=phase.v1.blend(p,n,w).astype('float32');np.save(OUT/f'pred_{d}.npy',q)
            entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        val=phase.inn.prior.measure(entries);report[str(stop)]={'result':val,'w':w.tolist()}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[str(stop)],flush=True)
        if end==14:
            w*=np.array(val['candidate'])>val['base']
            if not np.any(w):break
if __name__=='__main__':main()
