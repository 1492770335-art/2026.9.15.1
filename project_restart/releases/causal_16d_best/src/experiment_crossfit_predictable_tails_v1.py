"""Cross-fit short-horizon return teachers and supervise predictable and unexplained return bands separately."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_phase_future_book_v1 as phase
import experiment_full_coverage_book_v1 as cache
import experiment_pairwise_book_tree_v1 as pair
import experiment_causal_price_level_centering_v1 as readout

OUT=phase.inn.prior.ART/'crossfit_predictable_tails_v1'
LAGS=(1,5,10)
BANDS=((0,5),(5,20),(20,60))
SCALE=phase.inn.SCALE
WIDTH=np.sqrt(np.array([5,15,40],dtype='float32'))


def design(state,v,t):
    z=state[t];m=z.mean(1,keepdims=True)
    x=z-m
    signed=x[:,:,:31];level=np.log(v[t])[:,:,None]
    x=np.concatenate([x,np.tanh(1.6*signed),signed*level],axis=2).astype('float32')
    m0=m[:,0]
    market=np.column_stack([m0,np.tanh(2*m0[:,:31]),m0[:,:31]*np.exp(-t[:,None]/350),
                            z[:,:,:31].std(1)*m0[:,:31]]).astype('float32')
    return market,x


def teacher_stats(d):
    raw=pair.rawday(d);s,v=pair.state(raw)
    t=np.arange(d%4,len(raw)-10,4)
    m,x=design(s,v,t)
    p=(raw[:,:,0].astype(float)-.00136).cumsum(0)
    y=np.stack([(p[t+h]-p[t])/np.sqrt(h) for h in LAGS],axis=2).clip(-7,7).astype('float32')
    ym=y.mean(1)
    xr=x.reshape(-1,x.shape[-1]);yr=(y-ym[:,None]).reshape(-1,3)
    return ((m.T@m/len(m),m.T@ym/len(m)),(xr.T@xr/len(xr),xr.T@yr/len(xr)))


def teacher_weights(stats):
    out=[]
    for strength,items in zip((.68,.18),zip(*stats)):
        gram,rhs=(np.mean(a,axis=0,dtype=float) for a in zip(*items))
        out.append(np.linalg.solve(gram+strength*np.diag(gram.diagonal().clip(.0001,None)),rhs).astype('float32'))
    return out


def forecast_teacher(raw,weights):
    s,v=pair.state(raw);out=np.empty((len(raw),50,3),np.float32)
    for lo in range(0,len(raw),160):
        t=np.arange(lo,min(lo+160,len(raw)))
        m,x=design(s,v,t);r=x@weights[1]
        out[t]=r-r.mean(1,keepdims=True)+(m@weights[0])[:,None]
    return out


def predictable(y,t):
    result=[]
    for lo,hi in BANDS:
        forecasts=[]
        for j,h in enumerate(LAGS):
            if (hi-lo)%h:continue
            seq=t[:,None]+np.arange(lo,hi,h)
            forecasts.append(y[seq,: ,j].sum(1)*np.sqrt(h/(hi-lo)))
        result.append(np.mean(forecasts,axis=0))
    return np.stack(result,axis=2).astype('float32')


class Student(phase.BookNet):
    def __init__(self,width):
        super().__init__(width)
        self.aux=nn.Identity()
        self.future=nn.Linear(96,66)
        self.error_local=nn.Sequential(nn.Linear(96,56),nn.SiLU(),nn.Dropout(.16),nn.Linear(56,3))
        self.error_market=nn.Sequential(nn.Linear(192,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.18),nn.Linear(48,3))
        self.register_buffer('width',torch.from_numpy(WIDTH.copy()))
        self.register_buffer('unit',torch.from_numpy(SCALE.copy()))

    def forward(self,x,v,g):
        estimate,h,_,_=super().forward(x,v,g)
        error=self.error_local(h)*v[:,:,None]
        m=self.error_market(torch.cat((h.mean(1),h.std(1,unbiased=False)),1))*v.mean(1,keepdim=True)
        error=error-error.mean(1,keepdim=True)+self.beta[None,:,None]*m[:,None]
        return ((self.width*(estimate+error)).cumsum(-1)*self.unit,
                self.future(h),estimate,error)


def criterion(p,y,v):
    pm=p.mean(1);ym=y.mean(1)
    return F.smooth_l1_loss((p-pm[:,None])/v[:,:,None],(y-ym[:,None])/v[:,:,None],beta=.9)+.62*F.smooth_l1_loss(pm/v.mean(1,keepdim=True),ym/v.mean(1,keepdim=True),beta=.44)


def train(end,teacher):
    torch.manual_seed(31579)
    ticks=[np.r_[np.arange(d%2,600,2),np.arange(600+d%6,len(pair.rawday(d))-61,6)] for d in range(end)]
    size=sum(map(len,ticks));start=0
    y=torch.empty(size,50,3,device='cuda');pred=torch.empty_like(y)
    v=torch.empty(size,50,device='cuda');g=torch.empty(size,1,device='cuda')
    book=torch.empty(size,50,66,device='cuda',dtype=torch.float16)
    for d,t in enumerate(ticks):
        # Every pseudo-label for this training date comes from other training dates.
        weights=teacher_weights(teacher[:d]+teacher[d+1:end])
        known=forecast_teacher(pair.rawday(d),weights)
        target=predictable(known,t)
        sl=slice(start,start+len(t));start+=len(t)
        cache.prepare(d);root=cache.CACHE/f'day_{d:02d}'
        fx=np.load(root/'x.npy',mmap_mode='r')
        if d==0:x=torch.empty(size,50,fx.shape[-1],device='cuda',dtype=torch.float16)
        x[sl].copy_(torch.from_numpy(fx[t]))
        y[sl].copy_(torch.from_numpy(phase.inn.prior.labels(d).reshape(-1,50,3)[t]))
        pred[sl].copy_(torch.from_numpy(target))
        book[sl].copy_(torch.from_numpy(np.load(root/'a.npy',mmap_mode='r')[t]))
        v[sl].copy_(torch.from_numpy((.45+.55*np.load(root/'v.npy',mmap_mode='r')[t]).clip(.55,3)))
        g[sl].copy_(torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')))
    net=Student(x.shape[-1]).cuda()
    net.port.copy_(torch.from_numpy(phase.impact.exposure(end)[0][-4:]*np.sqrt(50)))
    net.beta.copy_(torch.from_numpy(phase.inn.base.ex.geometry(end)[0]))
    ema=Student(x.shape[-1]).cuda();ema.load_state_dict(net.state_dict())
    actual=torch.diff(y/net.unit,dim=-1,prepend=torch.zeros_like(y[:,:,:1]))/net.width
    opt=torch.optim.AdamW(net.parameters(),lr=.00068,weight_decay=.027)
    for ep in range(16):
        for ids in torch.randperm(size,device='cuda').split(48):
            vol=v[ids]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                q,aux,predictable_band,unexplained=net(x[ids].float(),vol,g[ids])
            loss=.74*criterion(predictable_band,pred[ids],vol)
            loss+=.28*criterion(unexplained,actual[ids]-pred[ids],vol)
            loss+=.58*criterion(q,y[ids],vol)+.28*F.smooth_l1_loss(aux.float(),book[ids].float(),beta=.6)
            loss-=.045*F.cosine_similarity(q-q.mean(1,keepdim=True),y[ids]-y[ids].mean(1,keepdim=True),dim=1,eps=1e-5).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.017)
        if ep in (5,10):
            for group in opt.param_groups:group['lr']*=.54
        if ep in (0,7,15):print('training',end,ep+1,flush=True)
    torch.save(ema.state_dict(),OUT/f'net_{end}.pth');return ema.eval()


def run_validation(out,fit_model,restore,infer_model=phase.infer):
    out.mkdir(exist_ok=True);hist=[];report={};good=np.ones((2,3),bool)
    for end,stop in ((12,14),(14,16),(16,20)):
        if end>=14:
            w=readout.fit(hist)*good[:,None,:]
            if end==16:np.save(out/'coef.npy',w)
        ck=out/f'net_{end}.pth'
        if ck.exists():
            st=torch.load(ck,map_location='cpu',weights_only=True)
            net=restore(st).cuda().eval();net.load_state_dict(st)
        else:
            net=fit_model(end)
        rows=[];daily=[]
        for d in range(end,stop):
            p=np.load((pair.OUT if d>=14 else phase.ref.OUT)/f'pred_{d}.npy')
            raw=pair.rawday(d);n=infer_model(net,raw)
            if d==end:np.testing.assert_allclose(n[:147],infer_model(net,raw[:147]),atol=.0001)
            np.save(out/f'side_{d}.npy',n);z=readout.basis(p,n)
            if end>=14:
                a=(z[0]@w[0])[:,None];b=z[1]@w[1];q=(p+a+b).astype('float32')
                np.save(out/f'pred_{d}.npy',q)
            y=phase.inn.prior.labels(d).reshape(p.shape)
            if end<16:hist.append(readout.moments(p,y,z))
            if end>=14:
                rows.append(tuple(c.reshape(-1,3) for c in (p,q,y)))
                if end==14:
                    old=phase.inn.prior.corr(rows[-1][0],rows[-1][2])
                    daily.append([phase.inn.prior.corr((p+v).reshape(-1,3),rows[-1][2])-old for v in (a,b)])
        if end>=14:
            report[str(stop)]=phase.inn.prior.measure(rows);print(stop,report[str(stop)],flush=True)
            if end==14:
                good=np.min(daily,axis=0)>0;report['eligible']=good.tolist()
            (out/'results.json').write_text(json.dumps(report,indent=2))
            if not good.any():break
def main():
    teacher=[]
    def fit_model(end):
        teacher.extend(teacher_stats(d) for d in range(len(teacher),end))
        return train(end,teacher)
    run_validation(OUT,fit_model,lambda st:Student(st['current.0.weight'].shape[1]))

if __name__=='__main__':main()

