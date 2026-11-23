"""Condition endpoint expectations on the first barrier reached along future relative and common price paths."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_causal_price_level_centering_v1 as readout
import experiment_delayed_tail_band_skill_v1 as early
import experiment_full_coverage_book_v1 as cache
import experiment_pairwise_book_tree_v1 as book
import experiment_phase_future_book_v1 as phase
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, corr, measure

OUT=ART/'first_passage_return_v1'
PARENT=ART/'session_restart_v1'
HORIZONS=(5,20,60)


def events(raw,t,v):
    r=raw[:,:,0].astype(float)-.00136
    m=r.mean(1,keepdims=True)
    path=np.concatenate((r-m,m),axis=1).cumsum(0)
    # The common path is less volatile; both scales are known at the forecast origin.
    vol=np.column_stack((v,.42*v.mean(1)))
    event=np.ones((len(t),51,3),np.uint8)
    for j,h in enumerate(HORIZONS):
        barrier=.68*np.sqrt(h)*vol
        open_mask=np.ones_like(vol,bool)
        for k in range(1,h+1):
            value=path[t+k]-path[t]
            passed=open_mask&(np.abs(value)>=barrier)
            event[:,:,j][passed]=np.where(value[passed]>0,2,0)
            open_mask &= ~passed
    return event


class Model(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(width,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.10),nn.Linear(128,80))
        self.stock=nn.Parameter(torch.zeros(50,80))
        self.key=nn.Linear(80,6)
        self.value=nn.Linear(80,20)
        self.local=nn.Sequential(nn.Linear(120,92),nn.LayerNorm(92),nn.SiLU(),nn.Dropout(.14),nn.Linear(92,21))
        self.market=nn.Sequential(nn.Linear(161,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.18),nn.Linear(96,21))
        self.register_buffer('order',torch.tensor([-1.,0.,1.]))

    def forward(self,x,v,age):
        z=self.encoder(x)+self.stock
        msg=self.value(z)
        key=self.key(z)
        pool=key.transpose(1,2).softmax(2)@msg
        recv=key.softmax(2)@pool
        avg=msg.mean(1,keepdim=True)
        local=self.local(torch.cat((z,recv,avg.expand_as(msg)),dim=2))
        market=self.market(torch.cat((pool.flatten(1),avg[:,0],msg.std(1,unbiased=False),age),dim=1))
        heads=torch.cat((local,market[:,None]),dim=1).float().reshape(-1,51,3,7)
        logits=heads[:,:,:,:3]
        centers=heads[:,:,:,3:6]+.45*self.order
        expected=(logits.softmax(-1)*centers).sum(-1)
        unit=torch.cat((v,.45*v.mean(1,keepdim=True)),dim=1)
        outcome=(.8*expected+.2*heads[:,:,:,6])*unit[:,:,None]
        relative=outcome[:,:50]
        relative=relative-relative.mean(1,keepdim=True)
        return relative+outcome[:,50:],logits,centers,unit


def fit(end):
    torch.manual_seed(74631)
    ticks=[np.r_[np.arange(d%3,600,3),np.arange(600+d%7,len(book.rawday(d))-61,7)] for d in range(end)]
    n=sum(map(len,ticks))
    v=torch.empty(n,50,device='cuda')
    y=torch.empty(n,50,3,device='cuda')
    e=torch.empty(n,51,3,dtype=torch.uint8,device='cuda')
    age=torch.empty(n,1,device='cuda')
    pos=0
    for d,t in enumerate(ticks):
        cache.prepare(d)
        root=cache.CACHE/f'day_{d:02d}'
        fx=np.load(root/'x.npy',mmap_mode='r')
        if d==0:
            x=torch.empty((n,50,fx.shape[-1]),device='cuda',dtype=torch.float16)
        vol=(.45+.55*np.load(root/'v.npy',mmap_mode='r')[t]).clip(.55,3)
        sl=slice(pos,pos+len(t));pos+=len(t)
        for dst,a in ((x,fx[t]),(v,vol),(y,labels(d).reshape(-1,50,3)[t]),(e,events(book.rawday(d),t,vol)),
                      (age,np.exp(-t[:,None]/400).astype('float32'))):
            dst[sl].copy_(torch.from_numpy(a))
    model=Model(x.shape[-1]).cuda()
    avg=Model(x.shape[-1]).cuda()
    avg.load_state_dict(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.035)
    for ep in range(14):
        for ix in torch.randperm(n,device='cuda').split(56):
            yy=y[ix];market=yy.mean(1,keepdim=True);local=yy-market
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                q,logits,centers,unit=model(x[ix].float(),v[ix],age[ix])
            m=q.mean(1,keepdim=True);cls=e[ix].long()
            targets=torch.cat((local,market),dim=1)/unit[:,:,None]
            chosen=centers.gather(-1,cls.unsqueeze(-1)).squeeze(-1)
            def balanced(a,b):
                return F.smooth_l1_loss(a[:,:50],b[:,:50],beta=1.2)+.5*F.smooth_l1_loss(a[:,50],b[:,50],beta=1.)
            forecast=torch.cat((q-m,m),dim=1)/unit[:,:,None]
            loss=balanced(forecast,targets)+.34*balanced(chosen,targets)
            loss+=.18*F.cross_entropy(logits[:,:50].reshape(-1,3),cls[:,:50].reshape(-1))
            loss+=.09*F.cross_entropy(logits[:,50].reshape(-1,3),cls[:,50].reshape(-1))
            loss-=.055*F.cosine_similarity(q-m,local,dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(avg.parameters()),list(model.parameters()),.018)
        if ep in (5,9):
            for group in opt.param_groups:
                group['lr']*=.55
        if ep in (0,6,13):
            print('fit days',end,'epoch',ep+1,flush=True)
    torch.save(avg.state_dict(),OUT/f'net_{end}.pth')
    return avg.eval()


@torch.inference_mode()
def infer(raw,model):
    device=next(model.parameters()).device
    pred=np.empty((*raw.shape[:2],3),np.float32)
    for offset in range(4):
        ids=np.arange(offset,len(raw),4)
        x,v=phase.spline.features(raw,ids)
        for i in range(0,len(ids),48):
            sl=slice(i,i+48);t=ids[sl]
            vol=torch.from_numpy((.45+.55*v[sl]).clip(.55,3)).to(device)
            age=torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')).to(device)
            pred[t]=model(torch.from_numpy(x[sl]).to(device).float(),vol,age)[0].cpu().numpy()
    return pred


def main():
    OUT.mkdir(exist_ok=True)
    hist=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        if end>=14:
            w=readout.fit(hist)
        if end==16:
            mask=np.min(daily,axis=0)>0
            report['eligible']=mask.tolist()
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            if not mask.any():
                return
            w*=mask[:,None,:]
            np.save(OUT/'coef.npy',w)
        path=OUT/f'net_{end}.pth'
        if path.exists():
            st=torch.load(path,weights_only=True,map_location='cpu')
            net=Model(st['encoder.0.weight'].shape[1]).cuda().eval()
            net.load_state_dict(st)
        else:
            net=fit(end)
        rows=[];daily=[]
        def advance(raw,p):
            m,r=readout.basis(p,infer(raw,net))
            return (p+(m@w[0])[:,None]+r@w[1]).astype('float32')
        for d in range(end,stop):
            p=np.load(PARENT/f'pred_{d}.npy') if end==16 else early.base(d)
            raw=book.rawday(d)
            n=infer(raw,net)
            if d==end:
                np.testing.assert_allclose(n[:149],infer(raw[:149],net),atol=1e-5)
            m,r=readout.basis(p,n)
            if end>=14:
                terms=((m@w[0])[:,None],r@w[1])
                q=(p+sum(terms)).astype('float32')
                if end==16:
                    np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:
                hist.append(readout.moments(p,y,(m,r)))
            if end>=14:
                row=tuple(v.reshape(-1,3) for v in (p,q,y))
                rows.append(row)
                if end==14:
                    base=corr(row[0],row[2])
                    daily.append([corr((p+c).reshape(-1,3),row[2])-base for c in terms])
        if end>=14:
            report[str(stop)]=measure(rows)
            print(stop,report[str(stop)],flush=True)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if report['20']['gain']>0:
        replay.main(PARENT,OUT,advance)


if __name__=='__main__':
    main()
