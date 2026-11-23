"""Calibrate cumulative book imbalances on a common price grid and bound depth in each price band."""
import numpy as np
import experiment_reconstructed_depth_flow_v1 as depth
import experiment_execution_volume_clock_v1 as clock
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'price_band_depth_curve_v1'
PARENT=ART/'damped_depth_fusion_v1'


def geometry():
    with np.load(depth.OUT/'model.npz') as st:g={k:st[k].copy() for k in st.files if k not in ('market','relative')}
    data=[];low=[];high=[]
    for d in range(4):
        raw=clock.pair.rawday(d);low.append(raw[:,:,21].min());high.append(raw[:,:,21].max())
        raw=raw[d%4::4];sp,_=clock.inputs(raw,g['spacing']);qs=g['quantum_coef'][0]*raw[:,:,40].astype(float)+g['quantum_coef'][1]
        data.append(np.column_stack((sp.ravel(),qs.ravel(),*(raw[:,:,j].astype(float).ravel() for j in (21,24,25,26,27)))))
    g['top_limits']=np.array([min(low),max(high)],dtype=float)
    z=np.concatenate(data);sp=z[:,0];qs=z[:,1];q=qs/np.maximum(1,sp)
    idsort=np.argsort(qs);ordered=qs[idsort];last=np.flatnonzero(np.r_[np.diff(ordered)!=0,True])
    prior=[];floor=[];cal=[]
    for j in range(4):
        v=z[:,3+j];key,n=np.unique(v,return_counts=True);center=key[n.argmax()]
        empty=abs(v-center)<.00002;prefix=np.cumsum(empty[idsort]);k=last
        wrong=prefix[k]+len(z)-1-k-(prefix[-1]-prefix[k]);split=np.argmin(wrong)
        cutoff=ordered[k[split]]
        good=(sp>0)&(qs+.009<cutoff)&(qs+2*q-.009>cutoff)
        x=z[good,2];a,b=np.polyfit(x,v[good],1)
        prior.append(cutoff);floor.append(center);cal.append((a,b))
    g.update(bands=np.array(prior),empty_code=np.array(floor),band_fit=np.array(cal))
    return g


def curve(raw,g):
    B,A,deep,D=depth.depths(raw,g);sp,_=clock.inputs(raw,g['spacing'])
    qs=g['quantum_coef'][0]*raw[:,:,40].astype(float)+g['quantum_coef'][1]
    q=(qs/np.maximum(sp,1)).clip(.01,10)
    lo,hi=g['top_limits']
    top=((raw[:,:,21].astype(float)-lo)/(hi-lo)).clip(.0001,.9999)
    observed=raw[:,:,24:28].astype(float)
    p=(((observed-g['band_fit'][:,1])/g['band_fit'][:,0]-lo)/(hi-lo)).clip(.0001,.9999)
    count=1+np.floor((g['bands']-qs[:,:,None])/(2*q[:,:,None])+1.e-7)
    count=count.clip(0,200);active=count>0
    # Repeated bands that see only the best quote are exactly the same observation.
    p=np.where(count==1,top[:,:,None],p)
    p=np.concatenate((top[:,:,None],p,(.5+.5*deep)[:,:,None]),axis=2)
    n,s=top.shape;upper=np.ones((n,s,6),np.float32);outer=p[:,:,-1];cap=np.ones_like(top)
    # Each deeper band must contain both side quantities from each inner band.
    for j in reversed(range(5)):
        here=p[:,:,j];on=np.ones_like(top,dtype=bool) if j==0 else active[:,:,j-1]
        fraction=np.minimum(outer/here,(1-outer)/(1-here)).clip(0,1)
        cap=np.where(on,cap*fraction,cap);upper[:,:,j]=np.where(on,cap,0.)
        outer=np.where(on,here,outer)
    return p,count,upper,D,qs,q


def check(g):
    report={}
    for d in range(4,8):
        raw=clock.pair.rawday(d);p,n,_,_,qs,q=curve(raw,g);each=[]
        for j,c in enumerate(g['bands']):
            observed=raw[:,:,24+j].astype(float);only=(qs+.009<c)&(qs+2*q-.009>c)
            err=observed-(g['band_fit'][j,0]*raw[:,:,21].astype(float)+g['band_fit'][j,1])
            empty=abs(observed-g['empty_code'][j])<.00002
            each.append(dict(empty_accuracy=float(np.mean((n[:,:,j]==0)==empty)),single_level_count=int(only.sum()),
                       same_best_median_abs_error=float(np.median(abs(err[only])))))
        report[d+1]=each
    return report


class Day:
    def __init__(self,raw,g):
        p,n,cap,D,qs,q=curve(raw,g);signed=2*p-1
        parts=[signed[:,:,0:1],signed[:,:,-1:]]
        before=p[:,:,0];beforecap=cap[:,:,0];lastcount=np.ones_like(before)
        for j in range(1,6):
            if j<5:
                number=n[:,:,j-1];on=number>0;extra=(number-lastcount).clip(0,50)
            else:
                on=np.ones_like(before,bool);extra=np.zeros_like(before);number=lastcount
            here=p[:,:,j];delta=(here-before)*on
            qb=cap[:,:,j]*here;qa=cap[:,:,j]*(1-here)
            net=((qb-beforecap*before)-(qa-beforecap*(1-before)))*on
            strength=net*D/(1+D)
            diff=np.stack((delta,net,strength,
                          delta/np.sqrt(1+extra),net/(1+extra),net*np.log(q),
                          delta*np.log1p(qs.clip(0)),(2*here-1)*cap[:,:,j]*on,
                          delta*np.log(D+.015).clip(-4,4),
                          (here-before)*np.tanh((2*before-1)*(2*here-1))*on),axis=2)
            parts.append(diff)
            before=np.where(on,here,before);beforecap=np.where(on,cap[:,:,j],beforecap);lastcount=np.where(on,number,lastcount)
        z=np.concatenate(parts,axis=2).astype('float32')
        selected=z[:,:,np.r_[0,1,np.arange(2,z.shape[-1],10),np.arange(3,z.shape[-1],10),np.arange(9,z.shape[-1],10)]]
        t=np.arange(len(z))+1;cs=np.concatenate((np.zeros_like(selected[:1],dtype=float),selected.cumsum(0,dtype=float)))
        pieces=[z]
        for lo,hi in ((0,7),(7,27),(27,87),(87,250)):
            end=(t-lo).clip(0);start=(t-hi).clip(0)
            mean=(cs[end]-cs[start])/(end-start).clip(1)[:,None,None]
            pieces.extend((mean,selected-mean))
        self.x=np.concatenate(pieces,axis=2).astype('float32')

    def design(self,t):
        v=self.x[t];m=v.mean(1)
        return np.column_stack((m,m[:,:22]*np.exp(-t[:,None]/340))).astype('float32'),v-m[:,None]


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(PARENT,OUT,clock.frozen(OUT,Day))
    else:
        g=geometry();clock.evaluate(OUT,PARENT,Day,g,{'held_training_price_band_identity':check(g),'cutoffs':g['bands'].tolist()})
