"""Use estimated execution location within the previous quote and previously matured trade markouts."""
import numpy as np
import experiment_execution_volume_clock_v1 as clock
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'execution_location_markouts_v1'
PARENT=ART/'side_network_execution_fusion_v1'


def geometry():
    with np.load(clock.OUT/'model.npz') as st:g={k:st[k].copy() for k in st.files if k not in ('relative','market')}
    raw=clock.pair.rawday(0)
    grid,count=np.unique(raw[:,:,39].astype(float),return_counts=True)
    # Rare locked quotes are below the first ordinary spread.
    origin=grid[np.flatnonzero(count>raw.shape[0]*50*.001)[0]]
    g['spacing']=np.array([origin,np.median(np.diff(grid[:15]))])
    design=[];targets=[];inactive=[];ratios=[]
    for d in range(4):
        raw=clock.pair.rawday(d);sp,ds=clock.inputs(raw,g['spacing']);r=raw[:,:,0].astype(float)-.00136
        bid=raw[:,:,77]==g['floor'][3];ask=raw[:,:,78]==g['floor'][4]
        half=np.where(bid,-ds,ds);ok=(bid^ask)&(half!=0)&(sp>0);ok[0]=False
        y=r[ok]/half[ok];x=raw[:,:,40].astype(float)[ok]
        target=y*sp[ok];use=(y>.015)&(y<9)&np.isfinite(target)
        design.append(np.column_stack((x[use],np.ones(use.sum()))));targets.append(target[use])
    a=np.concatenate(design);y=np.concatenate(targets);w=np.linalg.lstsq(a,y,rcond=None)[0]
    for _ in range(3):
        e=np.abs(a@w-y);ok=e<np.quantile(e,.90);w=np.linalg.lstsq(a[ok],y[ok],rcond=None)[0]
    g['quantum_coef']=w
    for d in range(4):
        raw=clock.pair.rawday(d);sp,ds=clock.inputs(raw,g['spacing'])
        buy=raw[:,:,54]>g['floor'][1];sell=raw[:,:,55]>g['floor'][2]
        f=raw[:,:,11].astype(float);zero=np.median(f[~(buy|sell)]);inactive.append(zero)
        flat=(buy^sell)&(abs(raw[:,:,0].astype(float)-.00136)<.01)&(ds==0)&(sp>0)
        qsp=w[0]*raw[:,:,40].astype(float)+w[1]
        ratios.append(np.median((f[flat]-zero)/(qsp[flat]*np.where(buy[flat],1.,-1.))))
    g['location']=np.array([np.median(inactive),np.median(ratios)])
    return g


def values(raw,g):
    sp,ds=clock.inputs(raw,g['spacing']);n=len(raw);t=np.arange(n)[:,None]
    # A locked quote has zero spread, so its relative spread cannot identify a quantum.
    valid=sp>0;last=np.maximum.accumulate(np.where(valid,t,0),axis=0)
    q=((g['quantum_coef'][0]*raw[:,:,40].astype(float)+g['quantum_coef'][1])/np.maximum(sp,1)).clip(.01,10)
    q=q[last,np.arange(50)[None,:]];q=np.where(valid[last,np.arange(50)[None,:]],q,1.)
    size=np.maximum(sp,0);old=np.vstack((size[:1],size[:-1]))
    volume=(raw[:,:,8].astype(float)-g['floor'][0]).clip(0,None)/g['rate'];traded=volume>0
    amt=[]
    for j in range(2):
        x=raw[:,:,54+j].astype(float)
        amt.append(np.where(x>g['floor'][j+1],np.interp(x,g[f'keys{j}'],g[f'values{j}']),0.))
    sign=np.divide(amt[0]-amt[1],amt[0]+amt[1],out=np.zeros_like(volume),where=amt[0]+amt[1]>0)
    r=raw[:,:,0].astype(float)-.00136
    curr=(raw[:,:,11].astype(float)-g['location'][0])/g['location'][1]
    prev=curr+r
    cost=prev/q
    excess=(cost-sign*old)*traded
    rest=np.divide(prev,old*q+.01)
    units=np.log1p(volume)
    return r,q,old,curr,prev,sign,rest,units,excess,volume


class Day:
    def __init__(self,raw,g):
        r,q,old,curr,prev,s,rest,units,excess,volume=values(raw,g);n=len(raw);trade=volume>0
        tick=np.arange(n);price=np.cumsum(r,0,dtype=float)
        live=tick[:,None]>0
        sweep=np.arcsinh(excess).clip(-5,5)*live
        context=np.stack((s,s*np.log1p(old),s*np.log(q),s*units,rest.clip(-4,4)*trade,
                          np.tanh(curr)*trade,np.tanh(prev)*trade,sweep,sweep*units,
                          np.tanh(r)*abs(sweep),s*(abs(excess)>.3),s*(abs(excess)>1.1)),axis=2)
        parts=[context];lagged=[]
        for h in (4,17,53):
            ix=np.maximum(tick-h,0);known=(tick>=h)[:,None]&trade[ix]
            move=price-price[ix]
            markout=np.arcsinh((move-curr[ix])/(q[ix]*np.sqrt(h)+.14))*known
            align=s[ix]*markout
            lagged.extend((s[ix]*(align>0)*known,s[ix]*(align<0)*known,
                           markout*np.tanh(units[ix]),markout*np.tanh(abs(sweep[ix])),
                           align*s,align*rest.clip(-3,3)*trade))
        stats=np.concatenate((context,np.stack(lagged,2)),axis=2)
        cs=np.concatenate((np.zeros_like(stats[:1]),stats.cumsum(0,dtype=float)))
        tsim=tick+1
        for lo,hi in ((0,5),(5,24),(24,88),(88,270)):
            end=(tsim-lo).clip(0);start=(tsim-hi).clip(0)
            win=(cs[end]-cs[start])/np.sqrt((end-start).clip(1))[:,None,None]
            parts.append(np.arcsinh(win))
        self.x=np.concatenate(parts,axis=2).astype('float32')

    def design(self,t):
        v=self.x[t];m=v.mean(1)
        return np.column_stack((m,m[:,:30]*np.exp(-t[:,None]/340))).astype('float32'),v-m[:,None]


def audit(g):
    report={}
    for d in range(4,8):
        raw=clock.pair.rawday(d);r,q,old,curr,prev,s,rest,units,excess,vol=values(raw,g)
        one=(abs(s)>.999999)&(vol>0);both=(abs(s)<.99999)&(vol>0)
        report[d+1]={}
        for key,active in [('both',both),('single',one)]:
            active[0]=False
            report[d+1][key]={'count':int(active.sum()),'median_absolute_quote_error_in_half_ticks':float(np.median(abs(excess[active]))),'fraction_within_point04':float(np.mean(abs(excess[active])<.04))}
    return report


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(PARENT,OUT,clock.frozen(OUT,Day))
    else:
        g=geometry();clock.evaluate(OUT,PARENT,Day,g,{'training_execution_location_check':audit(g)})
