"""Estimate slowly changing kernel coefficients using agreement between disjoint blocks within each training day."""
import json
import numpy as np
import experiment_supervised_gradient_kernel_v1 as kernel
import experiment_eight_date_expert_contrasts_v1 as early
import experiment_delayed_tail_band_skill_v1 as ref
import experiment_pairwise_book_tree_v1 as pair
import experiment_causal_price_level_centering_v1 as readout
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

OUT=ART/'forecast_coefficient_dynamics_v1'


def stats(d,geo):
    period=8;z=kernel.sample(d,period);result=[]
    t=np.arange(d%period,len(pair.rawday(d))-61,period)
    # Each retained 180-tick segment is followed by an embargo longer than the longest target.
    block=t//250;usable=(t%250)<180
    for j,((x,y),g) in enumerate(zip(z,geo)):
        select=lambda a:np.repeat(a,50) if j else a
        both=[]
        for half in (0,1):
            ids=np.flatnonzero(select(usable & ((block%2)==half)))
            cross=rhs=0.
            for idx in np.array_split(ids,max(1,len(ids)//4500)):
                p=kernel.fdata(x[idx],g);cross+=p.T@p;rhs+=p.T@y[idx]
            both.append((cross.astype(float)/len(ids),rhs.astype(float)/len(ids)))
        result.append(both)
    return result


def changing(history):
    outputs=[]
    for part,days in enumerate(zip(*history)):
        gram,b=(np.mean(a,axis=0) for a in zip(*[h for d in days for h in d]))
        reg=(.25 if part==0 else .15)*np.diag(gram.diagonal().clip(.006,None));reg[0,0]=.012
        w=np.linalg.solve(gram+reg,b);wh=np.linalg.inv(np.linalg.cholesky(gram+reg))
        gradient=np.array([[wh@(rhs-g@w) for g,rhs in day] for day in days])
        gradient-=gradient.mean(axis=0,keepdims=True)
        beta=np.zeros_like(w)
        for j in range(3):
            a,b=gradient[:,0,:,j],gradient[:,1,:,j]
            cross=(a.T@b+b.T@a)/(2*len(days))
            val,ax=np.linalg.eigh(cross);ax=ax[:,-4:]
            aa=a@ax;bb=b@ax;state=(aa+bb)*.5
            ss=(aa*bb).mean(0).clip(.000001,None)
            noise=.25*((aa-bb)**2).mean(0)
            # Cross-fit each day's two halves to estimate reliability and one-day persistence.
            lag=(aa[:-1]*bb[1:]+bb[:-1]*aa[1:]).mean(0)*.5
            rho=(lag/(ss+noise+.00005)).clip(-.58,.68)*(len(days)-1)/(len(days)+7)
            stable=ss/(ss+noise+.00005)
            posterior=stable*(.72*state[-1]+.28*rho*state[-2])
            beta[:,j]=wh.T@ax@(rho*posterior)
        outputs.append(beta.astype('float32'))
    return outputs


def main():
    OUT.mkdir(exist_ok=True);geo=kernel.geometry();days=[];mom=[];daily=[];report={}
    gate=np.ones((2,3),bool)
    for end,stop in ((8,10),(10,12),(12,14),(14,16),(16,20)):
        days.extend(stats(d,geo) for d in range(len(days),end));w=changing(days);rows=[]
        if end>=10:
            if end==16:
                gs=np.stack(daily);gate=((gs>0).sum(0)>=4)&(gs.mean(0)>0)&(gs[-2:].mean(0)>0)
                report['eligible']=gate.tolist()
                if not gate.any():break
            coef=readout.fit(mom)*gate[:,None,:]
            if end==16:
                pars={f'{i}_{k}':v for i,gg in enumerate(geo) for k,v in gg.items()}
                np.savez(OUT/'model.npz',market=w[0],relative=w[1],coef=coef,**pars)
        for d in range(end,stop):
            p=np.load(kernel.OUT/f'pred_{d}.npy') if d>=16 else ref.base(d)
            raw=pair.rawday(d);delta=kernel.predict(raw,geo,w)
            m=delta.mean(1);r=pair.innovation(p,delta-m[:,None]);z=(m,r)
            if d==end:np.testing.assert_allclose(delta[:135],kernel.predict(raw[:135],geo,w),atol=3e-6)
            if end>=10:
                a=(m@coef[0])[:,None];b=r@coef[1];q=(p+a+b).astype('float32')
                if end==16:np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:mom.append(readout.moments(p,y,z))
            if end>=10:
                row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
                if end<16:
                    old=corr(row[0],row[2]);daily.append([corr((p+x).reshape(-1,3),row[2])-old for x in (a,b)])
        if end>=10:
            report[str(stop)]=measure(rows);print(stop,report[str(stop)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))


def refinement():
    _,first=kernel.refinement()
    geo,w,coef=kernel.load_fit(OUT)
    def refine(raw,p):
        base=first(raw,p);side=kernel.predict(raw,geo,w);m=side.mean(1)
        r=pair.innovation(base,side-m[:,None])
        return (base+(m@coef[0])[:,None]+r@coef[1]).astype('float32')
    return OUT,refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:main()
