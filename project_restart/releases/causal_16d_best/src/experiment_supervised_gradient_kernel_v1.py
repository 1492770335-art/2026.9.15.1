"""Fit a nonlinear return kernel whose input metric is learned from pilot-model gradients on early dates."""
import json
import numpy as np
import experiment_pairwise_book_tree_v1 as pair
import experiment_causal_price_level_centering_v1 as readout
import experiment_phase_future_book_v1 as phase
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

OUT=ART/'supervised_gradient_kernel_v1'
SOURCES=(3,7,11,15,18,21,23,28,29,30)


def panel(s,v,t):
    z=s[t];m=z.mean(1,keepdims=True);rr=z-m
    selected=np.take(z,SOURCES,axis=-1)
    scale=.45+.55*v[t]
    clock=np.exp(-t[:,None]/380)
    shared=np.column_stack((m[:,0],np.tanh(1.7*selected).mean(1),selected.std(1),clock,scale.mean(1)))
    field=np.take(m[:,:,0:31],SOURCES,axis=-1)
    market=np.broadcast_to(field,(len(t),50,len(SOURCES)))
    relative=np.concatenate((rr,market,np.tanh(1.5*selected),np.log(scale)[:,:,None],
                             np.broadcast_to(clock[:,:,None],scale.shape+(1,))),axis=-1)
    return shared.astype('float32'),relative.astype('float32')


def sample(d,period):
    raw=pair.rawday(d);s,v=pair.state(raw)
    t=np.arange(d%period,len(s)-61,period)
    xs=panel(s,v,t);y=labels(d).reshape(-1,50,3)[t];m=y.mean(1)
    ys=(m,(y-m[:,None])/(.45+.55*v[t,:,None]))
    return [(x.reshape(-1,x.shape[-1]),a.reshape(-1,3).clip(-5,5)) for x,a in zip(xs,ys)]


def fdata(x,g):
    xx=((x-g['center'])/g['scale']).clip(-4,4)
    proj=xx@g['omega']
    return np.column_stack((np.ones(len(x),np.float32),xx,np.sin(proj),np.cos(proj))).astype('float32')


def pilot(samples,probe,rng):
    x=np.concatenate([a for a,b in samples]);y=np.concatenate([b for a,b in samples])
    center=x.mean(0);scale=x.std(0).clip(.045,None)
    width=x.shape[1]
    omega=(rng.normal(size=(width,84))/np.sqrt(width)*1.5).astype('float32')
    g=dict(center=center,scale=scale,omega=omega)
    p=fdata(x,g);gram=p.T@p/len(p);b=p.T@y/len(p)
    w=np.linalg.solve(gram+.21*np.diag(gram.diagonal().clip(.01,None)),b)
    ww=w[1+width:]
    xx=((probe-center)/scale).clip(-4,4)
    proj=xx@omega
    gradients=np.stack([w[1:1+width,j]+(np.cos(proj)*ww[:84,j]-np.sin(proj)*ww[84:,j])@omega.T for j in range(3)])
    gradients/=np.sqrt(np.mean(gradients**2,axis=(1,2),keepdims=True)).clip(.0001,None)
    grad=gradients.reshape(-1,width)
    values,vectors=np.linalg.eigh(grad.T@grad/len(grad))
    values=values.clip(0,None);values=.68*values/(values.sum()+1e-9)+.32/width
    transform=(vectors*np.sqrt(values))@vectors.T
    draw=rng.normal(size=(width,148))
    om=(transform@draw)*np.tile([.58,1.15,1.85,2.65],37)
    return dict(center=center,scale=scale,omega=om.astype('float32'))


def geometry():
    rng=np.random.default_rng(44631)
    samples=[sample(d,23) for d in range(4)]
    probe=[sample(d,38) for d in range(4,8)]
    out=[]
    for i in range(2):
        x=np.concatenate([day[i][0] for day in probe])
        select=rng.choice(len(x),min(len(x),7500),replace=False)
        out.append(pilot([day[i] for day in samples],x[select],rng))
    return out


def statistics(d,geometry):
    parts=sample(d,8);stats=[]
    for (x,y),g in zip(parts,geometry):
        gram=rhs=0.
        for k in range(0,len(x),5000):
            pp=fdata(x[k:k+5000],g)
            gram+=pp.T@pp;rhs+=pp.T@y[k:k+5000]
        stats.append((gram.astype(float)/len(x),rhs.astype(float)/len(x)))
    return stats


def fit(history):
    weights=[]
    for i,stats in enumerate(zip(*history)):
        g,b=(np.mean(v,axis=0) for v in zip(*stats))
        # Penalize the nonlinear map more strongly for the shared market series.
        alpha=(.23,.12)[i];reg=alpha*np.diag(g.diagonal().clip(.006,None));reg[0,0]=.009
        weights.append(np.linalg.solve(g+reg,b).astype('float32'))
    return weights


def predict(raw,geo,w):
    s,v=pair.state(raw);out=np.empty((len(raw),50,3),np.float32)
    for lo in range(0,len(raw),76):
        t=np.arange(lo,min(lo+76,len(raw)));m,r=panel(s,v,t)
        common=fdata(m,geo[0])@w[0]
        side=(fdata(r.reshape(-1,r.shape[-1]),geo[1])@w[1]).reshape(len(t),50,3)*(.45+.55*v[t,:,None])
        out[t]=common[:,None]+side-side.mean(1,keepdims=True)
    return out


def main():
    OUT.mkdir(exist_ok=True);geo=geometry();history=[];hist=[];report={};good=np.ones((2,3),bool)
    for end,stop in ((12,14),(14,16),(16,20)):
        history.extend(statistics(d,geo) for d in range(len(history),end));w=fit(history)
        if end>=14:
            coef=readout.fit(hist)*good[:,None,:]
            if end==16:
                pars={f'{i}_{key}':v for i,z in enumerate(geo) for key,v in z.items()}
                np.savez(OUT/'model.npz',market=w[0],relative=w[1],coef=coef,**pars)
        rows=[];daily=[]
        for d in range(end,stop):
            raw=pair.rawday(d);n=predict(raw,geo,w)
            if d==end:np.testing.assert_allclose(n[:233],predict(raw[:233],geo,w),atol=4e-6)
            np.save(OUT/f'side_{d}.npy',n)
            base=ART/'conditional_tail_payouts_v1' if d>=16 else (pair.OUT if d>=14 else phase.ref.OUT)
            p=np.load(base/f'pred_{d}.npy');z=readout.basis(p,n)
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


def load_fit(root):
    with np.load(root/'model.npz') as st:
        geo=[{k:st[f'{i}_{k}'].copy() for k in ('center','scale','omega')} for i in range(2)]
        return geo,[st['market'].copy(),st['relative'].copy()],st['coef'].copy()


def refinement():
    import experiment_conditional_tail_payouts_v1 as prev
    _,first=prev.refinement()
    geo,w,coef=load_fit(OUT)
    def refine(raw,p):
        base=first(raw,p);m,r=readout.basis(base,predict(raw,geo,w))
        return (base+(m@coef[0])[:,None]+r@coef[1]).astype('float32')
    return OUT,refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:main()
