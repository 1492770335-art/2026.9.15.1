"""Adjust sector routing from trailing peer co-movement, using disjoint return blocks and no labels."""
import json
import numpy as np
import torch
from torch.nn import functional as F
import experiment_y3_sector_anchored_factors_v1 as sector

base=sector.base
clock=sector.clock
incumbent=sector.incumbent
OUT=clock.ART/'y3_adaptive_sector_routing_v1'
SPAN=12
DEPTH=32


def correlations(raw,ids):
    r=raw[:,:,0].astype(float)-.00136
    count=len(r)//SPAN
    if not count:return np.zeros((1,50,4))
    value=r[:count*SPAN].reshape(count,SPAN,50).sum(1)
    value=clock.center(value)
    membership=np.eye(4)[ids]
    sums=value@membership
    peer=(sums[:,None,:]-value[:,:,None]*membership)/(membership.sum(0)-membership)
    def window(x):
        cs=np.concatenate((np.zeros_like(x[:1]),x.cumsum(0)))
        t=np.arange(1,count+1);s=(t-DEPTH).clip(0)
        return (cs[t]-cs[s])/(t-s).reshape((-1,)+(1,)*(x.ndim-1))
    av=window(value);pv=window(peer)
    cv=window(peer*value[:,:,None])-av[:,:,None]*pv
    va=window(value**2)-av**2;vb=window(peer**2)-pv**2
    rho=cv/np.sqrt((va.clip(0)[:,:,None]+.18)*(vb.clip(0)+.10))
    return np.concatenate((np.zeros_like(rho[:1]),rho))


def prior(end,ids):
    return np.mean([correlations(clock.book.rawday(d),ids)[DEPTH:].mean(0) for d in range(end)],0)


@torch.inference_mode()
def changes(raw,t,m,ids,typical,levels_only=False):
    trace=correlations(raw,ids)
    when=(t+1)//SPAN
    confidence=when[:,None,None].clip(0,DEPTH)/(when[:,None,None].clip(0,DEPTH)+10.)
    shift=3.0*confidence*(trace[when]-typical)
    # Shuffle among recipients in the same static group. All current sector average changes
    # and each recipient's original sector membership are present in the control.
    perm=np.arange(50)
    for k in range(4):
        cells=np.flatnonzero(ids==k);perm[cells]=np.roll(cells,5)
    out=[]
    for positions in np.array_split(np.arange(len(t)),max(1,(len(t)+500)//501)):
        x,v=base.cache.phase.spline.features(raw,t[positions])
        x=torch.tensor(x,device='cuda');v=torch.tensor((.45+.55*v).clip(.55,3),device='cuda')
        shifted=torch.tensor(shift[positions].astype('float32'),device='cuda')
        for xx,vv,extra in zip(x.split(64),v.split(64),shifted.split(64)):
            h=F.silu(m.enc(xx.float())+m.stock);bias=m.assign(h)+m.bias
            value=m.value(h)
            mu=(value.sum(1,keepdim=True)-value)/49
            sd=((value.square().sum(1,keepdim=True)-value.square())/49-mu.square()).clamp_min(1e-7).sqrt()
            def forward(adjust):
                logits=bias+adjust
                weight=logits.softmax(1)
                modes=weight.transpose(1,2)@value
                peer=(modes[:,None]-weight[:,:,:,None]*value[:,:,None])/(1-weight[:,:,:,None]).clamp_min(.0001)
                routed=(logits.softmax(2)[:,:,:,None]*peer).sum(2)
                p=m.read(torch.cat((h,routed,mu,sd),2))*vv[:,:,None]
                return p-p.mean(1,keepdim=True)
            a,c=forward(0.),forward(extra)
            far=lambda q:q[:,:,2]-clock.RATIO*q[:,:,1]
            if levels_only:
                value=torch.stack((far(a),far(c)),2)
            else:
                b=forward(extra[:,perm])
                value=.20*torch.stack((far(c)-far(a),far(c)-far(b),c[:,:,2]-b[:,:,2]),2)
            out.append(value.cpu().numpy())
    return np.concatenate(out)


def run():
    folds=[];days=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        ids=sector.grouping(end);typical=prior(end,ids);m=sector.train(end,ids)
        for begin in range(end,stop,2):
            groups=[[[],[],[]],[[],[],[]]]
            for d in (begin,begin+1):
                raw=clock.book.rawday(d);t=np.arange(d%4,len(raw),4)
                z=changes(raw,t,m,ids,typical)
                if d==begin:
                    for n in (137,613):
                        st=t[t<n]
                        np.testing.assert_allclose(changes(raw[:n],st,m,ids,typical),z[:len(st)],rtol=0,atol=3e-6)
                y=clock.labels(d).reshape(-1,50,3)[t]
                for j,ref in enumerate(incumbent.REFS):
                    p=ref(d).reshape(-1,50,3)[t]
                    for k in range(3):groups[j][k].append((p,y,z[:,:,k]))
                days.append([base.chain.metric.gain(g[-1:]) for g in groups[0]])
            folds.append([[base.chain.metric.gain(g) for g in side] for side in groups]);print(begin+1,folds[-1][0],flush=True)
    good=[k for k in range(3) if all(sum(r[j][k]['y3']>0 for r in folds)>=3 and
        np.mean([r[j][k]['y3'] for r in folds])>0 and np.mean([r[j][k]['relative_far'] for r in folds])>0
        for j in (0,1)) and min(r[0][k]['y3'] for r in folds[-2:])>0 and sum(d[k]['y3']>0 for d in days)>=5]
    pick=max(good,key=lambda k:np.mean([r[0][k]['y3'] for r in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,chosen=pick,
        strategies=['dynamic minus fixed far','matched minus shuffled far','matched minus shuffled whole']),indent=2))
    if pick is not None:
        ids=sector.grouping(16)
        torch.save(dict(state=sector.train(16,ids).cpu().state_dict(),typical=torch.tensor(prior(16,ids)),chosen=pick),OUT/'model.pth')
    return pick is not None


def layer():
    parent=incumbent.layer();pkg=torch.load(OUT/'model.pth',map_location='cpu',weights_only=True)
    state=pkg['state'];ids=state['bias'].argmax(1).numpy();typ=pkg['typical'].numpy()
    m=sector.Anchored(state['enc.0.weight'].shape[1],ids).cuda().eval();m.load_state_dict(state)
    def apply(raw,p):
        q=parent(raw,p);q[:,:,2]+=changes(raw,np.arange(len(q)),m,ids,typ)[:,:,pkg['chosen']]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as old
    parent=old.load();update=layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():base.chain.replay.verify(OUT,layer,incumbent.OUT)
