"""Propagate the nonlinear return kernel through train-estimated conditional book dynamics."""
import json
import numpy as np
import experiment_supervised_gradient_kernel_v1 as kernel
import experiment_rolling_drift_strength_v1 as previous
import experiment_pairwise_book_tree_v1 as pair
import experiment_phase_future_book_v1 as phase
import experiment_causal_price_level_centering_v1 as readout
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure
from experiment_multiscale_return_net_v1 import SCALE

OUT=ART/'propagated_return_kernel_v1'
LAGS=(5,10,15,20,40)


def transitions(d,geo):
    raw=pair.rawday(d);s,v=pair.state(raw);t=np.arange(d%9,len(raw)-41,9)
    def observations(ids):
        m,r=kernel.panel(s,v,ids)
        return [((z.reshape(-1,z.shape[-1])-g['center'])/g['scale']).clip(-4,4) for z,g in zip((m,r),geo)]
    first=observations(t);out=[]
    for horizon in LAGS:
        later=observations(t+horizon);stats=[]
        for a,b in zip(first,later):
            ma=a.mean(0);mb=b.mean(0)
            stats.append((ma,mb,a.T@a/len(a),a.T@b/len(a),b.T@b/len(a)))
        out.append(stats)
    return out


def fit_transition(history,geo):
    operators=[]
    for lag,all_h in zip(LAGS,zip(*history)):
        local=[]
        for i,days in enumerate(zip(*all_h)):
            ma,mb,xx,xy,yy=(np.mean(a,axis=0,dtype=float) for a in zip(*days))
            xx-=np.outer(ma,ma);xy-=np.outer(ma,mb);yy-=np.outer(mb,mb)
            penalty=(.48 if i==0 else .25)*np.diag(xx.diagonal().clip(.005,None))
            diag=np.diag((xy.diagonal()/xx.diagonal().clip(.005,None)).clip(-.25,.995))
            op=np.linalg.solve(xx+penalty,xy+penalty@diag);bias=mb-ma@op
            error=yy-op.T@xy-xy.T@op+op.T@xx@op
            omega=geo[i]['omega']
            variance=(omega*(error@omega)).sum(0).clip(0,15)
            local.append(tuple(z.astype('float32') for z in (op,bias,np.exp(-.5*variance))))
        operators.append(local)
    return operators


def forward(z,g,w,op):
    trans,bias,atten=op
    expected=(z@trans+bias).clip(-4,4);proj=expected@g['omega']
    # For an approximately Gaussian innovation, its characteristic function attenuates each frequency.
    return w[0]+expected@w[1:1+z.shape[1]]+(np.sin(proj)*atten)@w[1+z.shape[1]:-len(atten)]+(np.cos(proj)*atten)@w[-len(atten):]


def predict(raw,p,geo,w,operators):
    s,v=pair.state(raw);out=np.empty_like(p)
    for lo in range(0,len(p),72):
        t=np.arange(lo,min(lo+72,len(p)));m,r=kernel.panel(s,v,t)
        z=[((a.reshape(-1,a.shape[-1])-g['center'])/g['scale']).clip(-4,4) for a,g in zip((m,r),geo)]
        path=[]
        for lag,op in zip(LAGS,operators):
            market=forward(z[0],geo[0],w[0],op[0])
            side=forward(z[1],geo[1],w[1],op[1]).reshape(len(t),50,3)
            side*=1+np.exp(-lag/100)*((.45+.55*v[t,:,None])-1)
            path.append(market[:,None]+side-side.mean(1,keepdims=True))
        # A 20-step interval is four 5-step intervals. A 60-step interval is three 20-step intervals.
        y2=(p[t,:,0]+sum(a[:,:,0] for a in path[:3]))*(SCALE[1]/SCALE[0])
        y3=(p[t,:,1]+path[3][:,:,1]+path[4][:,:,1])*(SCALE[2]/SCALE[1])
        out[t]=np.stack((path[0][:,:,0],y2,y3),axis=-1)
    return out


def main():
    OUT.mkdir(exist_ok=True);geo=kernel.geometry();trans=[];stats=[];hist=[];good=np.ones((2,3),bool);report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        for d in range(len(trans),end):
            trans.append(transitions(d,geo));stats.append(kernel.statistics(d,geo))
        operators=fit_transition(trans,geo);w=kernel.fit(stats);rows=[];daily=[]
        if end>=14:
            coef=readout.fit(hist)*good[:,None,:]
            if end==16:
                pars={f'{i}_{k}':v for i,g in enumerate(geo) for k,v in g.items()}
                for h,op in enumerate(operators):
                    for i,f in enumerate(op):
                        for j,a in enumerate(f):pars[f'op{h}_{i}_{j}']=a
                np.savez(OUT/'model.npz',market=w[0],relative=w[1],coef=coef,**pars)
        for d in range(end,stop):
            base=previous.OUT if d>=16 else (pair.OUT if d>=14 else phase.ref.OUT)
            p=np.load(base/f'pred_{d}.npy');raw=pair.rawday(d);n=predict(raw,p,geo,w,operators)
            if d==end:np.testing.assert_allclose(n[:154],predict(raw[:154],p[:154],geo,w,operators),atol=4e-6)
            z=readout.basis(p,n)
            if end>=14:
                a=(z[0]@coef[0])[:,None];b=z[1]@coef[1];q=(p+a+b).astype('float32')
                if d>=16:np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:hist.append(readout.moments(p,y,z))
            if end>=14:
                row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
                if end==14:
                    old=corr(row[0],row[2]);daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in (a,b)])
        if end>=14:
            report[str(stop)]=measure(rows)
            if end==14:good=np.min(daily,axis=0)>0;report['eligible']=good.tolist()
            (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report,flush=True)
            if not good.any():break


def refinement():
    _,first=previous.refinement()
    geo,w,coef=kernel.load_fit(OUT)
    with np.load(OUT/'model.npz') as st:
        operators=[[tuple(st[f'op{h}_{i}_{j}'].copy() for j in range(3)) for i in range(2)] for h in range(len(LAGS))]
    def refine(raw,p):
        base=first(raw,p);m,r=readout.basis(base,predict(raw,base,geo,w,operators))
        return (base+(m@coef[0])[:,None]+r@coef[1]).astype('float32')
    return OUT,refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:main()
