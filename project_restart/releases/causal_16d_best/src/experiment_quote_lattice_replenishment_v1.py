"""Infer two quote moves on a price lattice and test side-specific replenishment as a forward signal."""
import json
import numpy as np
import experiment_repricing_adjusted_book_flow_v1 as flow
import experiment_eight_date_expert_contrasts_v1 as early
import experiment_ordered_book_path_areas_v1 as areas
import experiment_delayed_tail_band_skill_v1 as ref
import experiment_pairwise_book_tree_v1 as pair
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

OUT=ART/'quote_lattice_replenishment_v1'
BID=(41,43,50,52,54,56,79)
ASK=(42,44,51,53,55,57,80)
FIELDS=BID+ASK


def geometry():
    tick=[];depth=[];spacing=[]
    for d in range(4):
        raw=pair.rawday(d);r=np.abs(raw[:,:,0].astype(float)-.00136)
        tick.append([np.quantile(a[a>.018],.07) for a in r.T])
        depth.append(np.arcsinh(raw[d%11::11,:,FIELDS].astype('float32')).reshape(-1,len(FIELDS)))
        u=np.unique(raw[:,:,39].astype(float));spacing.append([u[0],np.median(np.diff(u[:15]))])
    center,high=np.quantile(np.concatenate(depth),[.5,.85],axis=0)
    return dict(tick=np.median(tick,0),spacing=np.median(spacing,0),center=center.astype('float32'),scale=(high-center).clip(.09,None).astype('float32'))


def quotes(raw,g):
    n=len(raw);r=raw[:,:,0].astype(float)-.00136
    spread=1+np.rint((raw[:,:,39].astype(float)-g['spacing'][0])/g['spacing'][1]).astype('int32')
    ds=np.vstack([np.zeros((1,50),int),np.diff(spread,axis=0)])
    q=g['tick'].copy();bid=np.zeros((n,50),np.float32);ask=np.zeros_like(bid);valid=np.zeros((n,50),bool)
    quantum=np.empty_like(bid)
    for t in range(n):
        quantum[t]=q;z=r[t]/q
        half=np.rint(z).astype(int)
        # A quote update is usable only if both prices can be integer ticks and the measured displacement agrees.
        good=(abs(z-half)<.48)&((half-ds[t])%2==0)
        valid[t]=good & (t>0)
        bid[t]=np.where(valid[t],(half-ds[t])*.5,0)
        ask[t]=np.where(valid[t],(half+ds[t])*.5,0)
        single=(abs(z)>.66)&(abs(z)<1.4)
        q+=.11*single*(abs(r[t])-q)
    return spread,quantum,bid,ask,valid


def streams(raw,g):
    sp,q,b,a,ok=quotes(raw,g);n=len(raw)
    x=np.arcsinh(raw[:,:,FIELDS].astype('float32'))
    sizes=.12+np.exp(np.tanh((x-g['center'])/g['scale'])*1.25)
    bu,au=np.split(sizes,2,axis=-1)
    pb=np.concatenate((bu[:1],bu[:-1]));pa=np.concatenate((au[:1],au[:-1]))
    tb=np.arange(n)[:,None];lastb=np.maximum.accumulate(np.where(b!=0,tb,0),0);lasta=np.maximum.accumulate(np.where(a!=0,tb,0),0)
    bs=b[:,:,None];ass=a[:,:,None];live=ok[:,:,None]
    add=np.where(bs>0,bu,np.where(bs<0,-pb,bu-pb))
    offer=np.where(ass<0,au,np.where(ass>0,-pa,au-pa))
    refill=(bu-pb)*(bs==0)-(au-pa)*(ass==0)
    inward=bu*(bs>0)-au*(ass<0)
    outward=pa*(ass>0)-pb*(bs<0)
    # Candidate queue fields are still anonymous. Only the measured side price moves are used to switch the accounting rule.
    f=np.concatenate(((add-offer)*live,refill*live,inward*live,outward*live,
                      (bu-au)/(bu+au),((bu+au)-(pb+pa))*np.sign(bs+ass)*live),axis=2)
    event=np.stack((np.sign(b)+np.sign(a),np.tanh(b/3)+np.tanh(a/3),
                    (b>0).astype(float)-(a<0),(a>0).astype(float)-(b<0),
                    (np.sign(b)+np.sign(a))*np.log1p(sp),
                    np.exp(-(tb-lastb)/16)-np.exp(-(tb-lasta)/16),
                    np.sign(b)*np.exp(-(tb-lasta)/12),np.sign(a)*np.exp(-(tb-lastb)/12)),axis=2)
    return np.concatenate((f,event),axis=2).astype('float32'),sp,q


class Day:
    def __init__(self,raw,g):
        z,self.sp,self.quantum=streams(raw,g)
        self.z=z
        self.cs=np.concatenate((np.zeros_like(z[:1],dtype=float),z.cumsum(0,dtype=float)))
    def design(self,t):
        parts=[self.z[t]]
        for lo,hi in ((1,7),(7,26),(26,85),(85,310)):
            end=(t+1-lo).clip(0);start=(t+1-hi).clip(0);sz=(end-start).clip(1)[:,None,None]
            change=(self.cs[end]-self.cs[start])/np.sqrt(sz)
            parts.append(np.arcsinh(change).astype('float32'))
        z=np.concatenate(parts,axis=2);m=z.mean(1)
        return np.column_stack((m,m[:,:self.z.shape[-1]]*np.exp(-t[:,None]/360))).astype('float32'),z-m[:,None]


def main():
    OUT.mkdir(exist_ok=True);g=geometry();hist=[];daily=[];report={}
    for d in range(8,20):
        if d in (10,12,14,16):
            w=flow.fit(hist);rows=[]
            if d==16:
                gains=np.stack(daily);gate=((gains>0).sum(0)>=4)&(gains.mean(0)>0)&(gains[-2:].mean(0)>0)
                report['eligible']=gate.tolist()
                if not gate.any():break
                w=[a*b for a,b in zip(w,gate)];np.savez(OUT/'model.npz',**g,market=w[0],relative=w[1])
        p=np.load(ART/'individual_kernel_forecast_v1'/f'pred_{d}.npy') if d>=16 else ref.base(d)
        raw=pair.rawday(d);day=Day(raw,g)
        if d>=10:
            a,b=areas.predict(day,p,w);q=(p+a+b).astype('float32')
            if d in (10,16):
                h=227;ca,cb=areas.predict(Day(raw[:h],g),p[:h],w);np.testing.assert_allclose(q[:h],p[:h]+ca+cb,atol=3e-6)
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if d<16:
            t=np.arange(d%7,len(p),7);hist.append(early.moments(p[t],y[t],day.design(t)))
        if d>=10:
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            if d<16:
                old=corr(row[0],row[2]);daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in (a,b)])
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))


def frozen():
    with np.load(OUT/'model.npz') as st:
        geom={k:st[k].copy() for k in ('tick','spacing','center','scale')}
        w=(st['market'].copy(),st['relative'].copy())
    def refine(raw,p):
        a,b=areas.predict(Day(raw,geom),p,w)
        return (p+a+b).astype('float32')
    return refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(ART/'individual_kernel_forecast_v1',OUT,frozen())
    else:main()
