"""Recover side-specific quantities from anonymous fields and integrate flow over exact past-volume windows."""
import json
import numpy as np
import experiment_quote_lattice_replenishment_v1 as lattice
import experiment_eight_date_expert_contrasts_v1 as early
import experiment_repricing_adjusted_book_flow_v1 as flow
import experiment_delayed_tail_band_skill_v1 as ref
import experiment_ordered_book_path_areas_v1 as areas
import experiment_pairwise_book_tree_v1 as pair
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

OUT=ART/'execution_volume_clock_v1'
WINDOWS=(4.,16.,56.,180.)


def inputs(raw,base):
    sp=1+np.rint((raw[:,:,39].astype(float)-base[0])/base[1])
    ds=np.vstack((np.zeros((1,50)),np.diff(sp,axis=0)))
    return sp,ds


def geometry():
    base=lattice.geometry()['spacing']
    floor=np.array([float(pair.rawday(0)[:,:,j].min()) for j in (8,54,55,77,78)])
    obs=[[],[]];xq=[];yq=[];rates=[]
    for d in range(4):
        raw=pair.rawday(d);sp,ds=inputs(raw,base)
        total=raw[:,:,8].astype(float)-floor[0];rates.append(np.minimum(total,np.quantile(total,.995,axis=0)).mean(0))
        left=raw[:,:,54].astype(float);right=raw[:,:,55].astype(float)
        for j,(own,other) in enumerate(((left,right),(right,left))):
            take=(own>floor[1+j])&(other==floor[2-j]);obs[j].append(np.column_stack((own[take],total[take])))
        bid=raw[:,:,77]==floor[3];ask=raw[:,:,78]==floor[4]
        only=(bid^ask)&(ds!=0);only[0]=False
        half=np.where(bid,-ds,ds)
        q=(raw[:,:,0].astype(float)-.00136)[only]/half[only];f=raw[:,:,40].astype(float)
        ok=(q>0)&(q<9);xq.append(np.column_stack((f[only]/sp[only],1/sp[only]))[ok]);yq.append(q[ok])
    out={'spacing':base,'floor':floor,'rate':np.mean(rates,0).clip(.002,None),
         'quantum_coef':np.linalg.lstsq(np.concatenate(xq),np.concatenate(yq),rcond=None)[0]}
    for j,pairs in enumerate(obs):
        z=np.concatenate(pairs);key,idx=np.unique(z[:,0],return_inverse=True)
        amount=np.bincount(idx,weights=z[:,1])/np.bincount(idx)
        out[f'keys{j}']=key;out[f'values{j}']=np.maximum.accumulate(amount)
    return out


def decode(raw,g):
    sp,ds=inputs(raw,g['spacing']);floor=g['floor']
    total=(raw[:,:,8].astype(float)-floor[0]).clip(0,None)
    measured=[]
    for j in range(2):
        x=raw[:,:,54+j].astype(float)
        measured.append(np.where(x>floor[j+1],np.interp(x,g[f'keys{j}'],g[f'values{j}']),0))
    both=measured[0]+measured[1]
    left=total*np.divide(measured[0],both,out=np.full_like(both,.5),where=both>0)
    right=total-left
    bid=raw[:,:,77]==floor[3];ask=raw[:,:,78]==floor[4]
    numerator=g['quantum_coef'][0]*raw[:,:,40].astype(float)+g['quantum_coef'][1]
    q=np.divide(numerator,sp,out=np.where(numerator>=0,10.,.01),where=sp!=0).clip(.01,10)
    if g.get('carry_locked_quantum',False):
        valid=sp>0
        last=np.maximum.accumulate(np.where(valid,np.arange(len(raw))[:,None],0),axis=0)
        q=np.where(valid[last,np.arange(50)],q[last,np.arange(50)],1.)
    r=raw[:,:,0].astype(float)-.00136
    # Quote movement times come from the two observed reset fields. Single-side displacements are exact spread changes.
    ratio=r/np.sqrt(q*np.vstack((q[:1],q[:-1])))
    parity=ds.astype(int)%2
    half=2*np.rint((ratio-parity)*.5)+parity
    half=np.where(bid&~ask,-ds,np.where(ask&~bid,ds,np.where(bid&ask,half,0.)))
    b=(half-ds)*.5;a=(half+ds)*.5;b[0]=0;a[0]=0
    return left,right,total,b,a,sp,q,both


def audit(g):
    out={}
    for d in range(4,8):
        raw=pair.rawday(d);left,right,total,b,a,sp,q,est=decode(raw,g)
        ds=np.diff(sp,axis=0);r=raw[1:,:,0].astype(float)-.00136
        bid=raw[1:,:,77]==g['floor'][3];ask=raw[1:,:,78]==g['floor'][4]
        single=(bid^ask)&(ds!=0);tick=r[single]/np.where(bid,-ds,ds)[single]
        both=(left>0)&(right>0);source_tick=q[1:][single]
        out[d+1]=dict(single_side_rows=int(single.sum()),both_quantity_rows=int(both.sum()),
            quantum_correlation=float(np.corrcoef(source_tick,tick)[0,1]),
            quantum_median_relative_error=float(np.median(abs(source_tick/tick-1))),
            both_quantity_correlation=float(np.corrcoef(est[both],total[both])[0,1]),
            both_quantity_median_relative_error=float(np.median(abs(est[both]/total[both]-1))))
    return out


def volume_window(signal,volume,budget):
    n,stocks,width=signal.shape;out=np.empty_like(signal);duration=np.empty((n,stocks),np.float32)
    for s in range(stocks):
        c=np.r_[0.,volume[:,s].cumsum(dtype=float)]
        accum=np.vstack((np.zeros((1,width)),signal[:,s].cumsum(0,dtype=float)))
        target=np.maximum(c[1:]-budget,0)
        end=np.searchsorted(c,target,side='left');start=(end-1).clip(0)
        frac=np.divide(target-c[start],c[end]-c[start],out=np.zeros(n),where=c[end]>c[start])
        back=accum[start]+frac[:,None]*(accum[end]-accum[start])
        used=(c[1:]-target).clip(.25,None)
        out[:,s]=(accum[1:]-back)/used[:,None]
        duration[:,s]=(np.arange(1,n+1)-start-frac*(end-start))
    return out,duration


def frame(raw,g):
    left,right,total,b,a,sp,q,_=decode(raw,g);scale=g['rate']
    left/=scale;right/=scale;volume=left+right
    side=np.divide(left-right,volume,out=np.zeros_like(volume),where=volume>0)
    signed=left-right
    on=(volume>0);both=(left>0)&(right>0)
    bid_lost=(b<0);ask_lost=(a>0)
    flowvec=np.stack((signed,left*ask_lost-right*bid_lost,left*(a==0)-right*(b==0),
                left*(a<0)-right*(b>0),volume*np.sign(a+b),np.sign(a+b)*~on,
                np.sign(a)*~(left>0)+np.sign(b)*~(right>0),np.tanh((b+a)/3),signed/np.sqrt(sp),
                signed*both,volume*(np.sign(b)+np.sign(a))*.5,signed*np.log1p(sp),
                side*volume*(abs(b)+abs(a)).clip(0,5)/3),axis=2).astype('float32')
    now=np.stack((side,np.arcsinh(signed),np.tanh(b/3),np.tanh(a/3),np.log1p(sp),
                  np.log(q),np.log1p(volume),side*both,side*(a+b==0)),axis=2).astype('float32')
    pieces=[now]
    for amount in WINDOWS:
        sums,duration=volume_window(flowvec,volume,amount)
        v=np.arcsinh(sums).astype('float32');speed=np.log((duration+.5)/(amount+.5)).clip(-3.5,3.5)
        pieces.extend((v,v[:,:,:5]*speed[:,:,None],(v[:,:,:5]*np.tanh((a+b)/3)[:,:,None]).astype('float32')))
    return np.concatenate(pieces,axis=2).astype('float32')


class Day:
    def __init__(self,raw,g):self.x=frame(raw,g)
    def design(self,t):
        z=self.x[t];m=z.mean(1)
        return np.column_stack((m,m[:,9:27]*np.exp(-t[:,None]/340))).astype('float32'),z-m[:,None]


def evaluate(out,parent,day_type,g,report):
    out.mkdir(exist_ok=True);hist=[];daily=[]
    for d in range(8,20):
        if d in (10,12,14,16):
            w=flow.fit(hist);rows=[]
            if d==16:
                gs=np.stack(daily);ok=((gs>0).sum(0)>=4)&(gs.mean(0)>0)&(gs[-2:].mean(0)>0)
                report['eligible']=ok.tolist()
                if not ok.any():break
                w=[a*b for a,b in zip(w,ok)];np.savez(out/'model.npz',**g,market=w[0],relative=w[1])
        p=np.load(parent/f'pred_{d}.npy') if d>=16 else ref.base(d);raw=pair.rawday(d);day=day_type(raw,g)
        if d>=10:
            a,b=areas.predict(day,p,w);q=(p+a+b).astype('float32')
            if d in (10,16):
                small=219;aa,bb=areas.predict(day_type(raw[:small],g),p[:small],w)
                np.testing.assert_allclose(q[:small],p[:small]+aa+bb,atol=3.e-6)
            if d>=16:np.save(out/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if d<16:
            t=np.arange(d%6,len(p),6);hist.append(early.moments(p[t],y[t],day.design(t)))
        if d>=10:
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            if d<16:
                old=corr(row[0],row[2]);daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in (a,b)])
        if d in (11,13,15,19):report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (out/'results.json').write_text(json.dumps(report,indent=2))


def main():
    g=geometry()
    evaluate(OUT,lattice.OUT,Day,g,{'training_reconstruction_checks':audit(g)})


def frozen(out=OUT,day_type=Day):
    with np.load(out/'model.npz') as st:
        g={k:st[k].copy() for k in st.files if k not in ('market','relative')}
        w=(st['market'].copy(),st['relative'].copy())
    def apply(raw,p):
        a,b=areas.predict(day_type(raw,g),p,w)
        return (p+a+b).astype('float32')
    return apply


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(lattice.OUT,OUT,frozen())
    else:main()
