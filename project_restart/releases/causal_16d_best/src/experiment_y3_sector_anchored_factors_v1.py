"""Anchor sparse donor and recipient factors in training-only co-movement groups, with a matched random control."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import experiment_y3_martingale_factor_net_v1 as base

clock=base.clock
incumbent=base.incumbent
OUT=clock.ART/'y3_sector_anchored_factors_v1'


def day_correlation(d):
    raw=clock.book.rawday(d);r=raw[:,:,0].astype(float)-.00136
    block=r[:len(r)//20*20].reshape(-1,20,50).sum(1)
    block-=block.mean(1,keepdims=True)
    return np.corrcoef(block.T)


def grouping(end):
    return balanced_groups(np.mean([day_correlation(d) for d in range(end)],0))


def balanced_groups(cov):
    val,vec=np.linalg.eigh(cov)
    factors=vec[:,-5:]*np.sqrt(val[-5:]);factors/=np.linalg.norm(factors,axis=1,keepdims=True).clip(.025)
    centers=KMeans(n_clusters=4,n_init=15,random_state=701).fit(factors).cluster_centers_
    slots=np.repeat(np.arange(4),(13,13,12,12))
    last=None
    for _ in range(20):
        cost=((factors[:,None]-centers[slots])**2).sum(2)
        rows,idx=linear_sum_assignment(cost)
        ids=np.empty(50,int);ids[rows]=slots[idx]
        if np.array_equal(last,ids):break
        centers=np.stack([factors[ids==i].mean(0) for i in range(4)])
        last=ids.copy()
    return ids


class Anchored(base.FactorNet):
    def __init__(self,width,ids):
        super().__init__(width)
        self.register_buffer('bias',2.5*F.one_hot(torch.tensor(ids,dtype=torch.long),4).float())

    def encode_and_predict(self,x,v):
        h=F.silu(self.enc(x.float())+self.stock)
        logits=self.assign(h)+self.bias
        mass=logits.softmax(1)
        value=self.value(h)
        modes=mass.transpose(1,2)@value
        peer=(modes[:,None]-mass[:,:,:,None]*value[:,:,None])/(1-mass[:,:,:,None]).clamp_min(.0001)
        message=(logits.softmax(2)[:,:,:,None]*peer).sum(2)
        common=(value.sum(1,keepdim=True)-value)/49
        other_square=(value.square().sum(1,keepdim=True)-value.square())/49
        sd=(other_square-common.square()).clamp_min(1e-7).sqrt()
        out=self.read(torch.cat((h,message,common,sd),2))*v[:,:,None]
        return h,out-out.mean(1,keepdim=True)

    def forward(self,x,v):
        return self.encode_and_predict(x,v)[1]


def train(end,ids):
    return base.train(end,(False,),factory=lambda width:Anchored(width,ids))[0]


def schemes(p,s):
    a,b=s
    near=clock.RATIO*(b[:,:,1]-a[:,:,1])
    full=b[:,:,2]-a[:,:,2]
    far_b=b[:,:,2]-clock.RATIO*b[:,:,1]
    far_p=clock.center(p[:,:,2]-clock.RATIO*p[:,:,1])
    return .18*(full-near),.18*full,.12*(far_b-far_p)


def run():
    folds=[];daily=[];by_end={}
    perm=np.random.default_rng(11743).permutation(50)
    for end,stop in ((8,12),(12,14),(14,16)):
        ids=grouping(end);by_end[end]=ids.tolist()
        nets=[train(end,v) for v in (ids[perm],ids)]
        for begin in range(end,stop,2):
            group=[[[],[],[]],[[],[],[]]]
            for d in (begin,begin+1):
                raw=clock.book.rawday(d);t=np.arange(d%4,len(raw),4)
                s=base.forecasts(raw,t,nets)
                if d==begin:
                    st=t[t<401]
                    for q,sub in zip(s,base.forecasts(raw[:401],st,nets)):
                        np.testing.assert_allclose(q[:len(st)],sub,rtol=0,atol=3e-5)
                y=clock.labels(d).reshape(-1,50,3)[t]
                for j,ref in enumerate(incumbent.REFS):
                    p=ref(d).reshape(-1,50,3)[t]
                    for k,z in enumerate(schemes(p,s)):group[j][k].append((p,y,z))
                daily.append([base.chain.metric.gain(g[-1:]) for g in group[0]])
            folds.append([[base.chain.metric.gain(g) for g in side] for side in group])
            print(begin+1,folds[-1][0],flush=True)
    # Compare the explicit dependency itself on both references. An entire replacement head
    # is informative here only when real groups beat random ones as well.
    passes=lambda k:all(np.mean([r[j][k]['y3'] for r in folds])>0 and
           np.mean([r[j][k]['relative_far'] for r in folds])>0 for j in (0,1)) and (
           min(r[0][k]['y3'] for r in folds[-2:])>0 and sum(d[k]['y3']>0 for d in daily)>=5)
    good=[k for k in (0,1) if passes(k)]
    if good and passes(2):good.append(2)
    pick=max(good,key=lambda k:np.mean([r[0][k]['y3'] for r in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,chosen=pick,groups=by_end,
        methods=['real minus random far','real minus random full','real far replacement']),indent=2))
    if pick is not None:
        ids=grouping(16)
        torch.save(dict(chosen=pick,states=[train(16,v).cpu().state_dict() for v in (ids[perm],ids)]),OUT/'models.pth')
    return pick is not None


def layer():
    parent=incumbent.layer();pkg=torch.load(OUT/'models.pth',weights_only=True,map_location='cpu')
    nets=[]
    for state in pkg['states']:
        net=Anchored(state['enc.0.weight'].shape[1],state['bias'].argmax(1).numpy()).cuda().eval()
        net.load_state_dict(state);nets.append(net)
    def apply(raw,p):
        q=parent(raw,p)
        q[:,:,2]+=schemes(q,base.forecasts(raw,np.arange(len(q)),nets))[pkg['chosen']]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as old
    parent=old.load();change=layer()
    return lambda raw,p:change(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():base.chain.replay.verify(OUT,layer,incumbent.OUT)
