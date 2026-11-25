"""Test whether long-horizon expert revisions can also correct the two shorter-horizon predictors."""
import sys
import json
import numpy as np
import torch
import experiment_y3_joint_stage_readout_v1 as stack

parent,history,trial=stack.parent,stack.history,stack.trial
OUT=trial.clock.ART/'joint_horizon_stage_transfer_v1'
N=stack.N
ARMS=('keep current prediction','uniform revision','family ridge','all revisions','pair-date sign agreement',
      'stock-relative only','market only','positive projection across dates','fixed half all','fixed half last two')


def features(z):
    common=z.mean(1,keepdims=True)
    return np.concatenate((z-common,np.broadcast_to(common,z.shape)),2)


def moments(day):
    raw=trial.clock.book.rawday(day)
    p,z=stack.panels(day,raw)
    y=trial.clock.labels(day).reshape(-1,50,3)
    grams=[]
    for ref,changes in zip(p,z):
        a=np.concatenate((np.ones_like(ref[:,:,:1]),ref,features(changes),y),2).reshape(y.size//3,-1).astype(float)
        grams.append(a.T@a)
    return np.stack(grams)


def fit(history_grams,target):
    dates=np.asarray(history_grams).mean(1)
    rows=np.arange(2*N)+3
    def equations(g):
        c=stack.covariance(g)
        # Estimate extra forecast directions while accounting for coverage by all three original predictions.
        h=c[:3,:3]
        cross=c[np.ix_(rows,np.arange(3))]
        proj=np.linalg.solve(h+.03*np.diag(h.diagonal()),cross.T)
        desired=c[:,3+2*N+target]*(c[target,target]/c[target,3+2*N+target])
        return c[np.ix_(rows,rows)]-cross@proj,(desired[rows]-c[rows,target])-proj.T@(desired[:3]-c[:3,target])
    g,b=equations(dates.sum(0))
    db=np.stack([equations(v)[1] for v in dates])
    eye=np.eye(2*N)
    family=np.zeros((N,len(stack.FAMILIES)))
    for i,(a,u) in enumerate(zip((0,*stack.FAMILIES[:-1]),stack.FAMILIES)):
        family[a:u,i]=1
    def solve(basis,mode):
        mat=basis.T@g@basis
        rhs=basis.T@b
        day=db@basis
        noise=day.std(0)*np.sqrt(len(dates))
        rhs*=abs(rhs)/(abs(rhs)+noise+1e-8)
        if mode=='sign':
            rhs*=np.maximum(0,np.sign(day[::2].sum(0))*np.sign(day[1::2].sum(0)))
        w=np.linalg.solve(mat+np.diag(mat.diagonal().clip(.0001)*1.35),rhs)
        if mode=='positive':
            w=np.maximum(w,0)
        return basis@w.clip(-1.,1.)
    duplicate=lambda a:np.tile(a,(2,1))
    sets=(np.ones((2*N,1)),duplicate(family),duplicate(np.eye(N)),duplicate(np.eye(N)),
          eye[:,:N],eye[:,N:],duplicate(family))
    ms=('all','all','all','sign','all','all','positive')
    ws=[np.zeros(2*N)]+[solve(basis,mode) for basis,mode in zip(sets,ms)]
    last=np.zeros(N)
    last[-2:]=.5
    ws.extend((np.full(2*N,.5),np.tile(last,2)))
    return np.asarray(ws)


def gains(g,w,target):
    c=stack.covariance(g)
    coeff=np.zeros((len(w),3+2*N))
    coeff[:,target]=1
    coeff[:,3:]=w
    vari=c[3+2*N+target,3+2*N+target]
    return (coeff@c[:-3,3+2*N+target])/np.sqrt(np.einsum('ij,jk,ik->i',coeff,c[:-3,:-3],coeff)*vari)-c[target,3+2*N+target]/np.sqrt(c[target,target]*vari)


def train():
    OUT.mkdir(exist_ok=True)
    days=[moments(d) for d in range(8,16)]
    choices=[]
    weights=[]
    reports=[]
    for target in range(3):
        folds=[]
        daily=[]
        for end in (2,4,6):
            w=fit(days[:end],target)
            folds.append([gains(g,w,target) for g in sum(days[end:end+2])])
            daily.extend([[gains(g,w,target) for g in pair] for pair in days[end:end+2]])
        f=np.array(folds)
        mean=f.mean(0)
        eligible=(f>0).sum(0).min(0)>=2
        eligible&=(mean.min(0)>0)&(f[-1].min(0)>0)&((np.array(daily)>0).sum(0).min(0)>=4)
        # A transfer to a shorter horizon must replicate and outperform the fixed positive mixture.
        if target<2:
            ids=[i for i in range(1,8) if eligible[i] and (mean[:,i]>mean[:,8]).all()]
            if not ids and eligible[8]:
                ids=[8]
        else:
            # Choosing between previous positive controls is confined to training dates.
            ids=[i for i in range(1,8) if eligible[i] and (mean[:,i,None]>mean[:,[8,9]]).all()]
            if not ids:
                ids=[i for i in (8,9) if eligible[i]]
        chosen=max(ids,key=lambda i:mean[:,i].min()) if ids else 0
        choices.append(chosen)
        weights.append(fit(days,target)[chosen])
        reports.append(dict(mean=mean.tolist(),folds=f.tolist(),daily=np.array(daily).tolist()))
        print('target',target+1,'selected',ARMS[chosen],'mean gains',np.round(mean[:,chosen],6),flush=True)
    report=dict(arms=ARMS,selected=choices,training_days=list(range(9,17)),evaluation_days=list(range(11,17)),targets=reports)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    np.savez(OUT/'model.npz',w=np.stack(weights),pick=choices)
    return choices


def apply_readout(p,z,w):
    out=p.copy()
    for k in range(3):
        if np.any(w[k]):
            # Reuse precisely the summation and float32 rounding of the Y3 readout.
            delta=stack.adjusted(np.zeros_like(p),z,w[k])[:,:,2]
            out[:,:,k]+=delta
    return out


def compose(directory):
    w=np.load(directory/'model.npz')['w']
    start=stack.early.incumbent.layer()
    updates=[stack.module(s).correction() for s in stack.STAGES]
    def predict(raw,p):
        q,z=stack.revisions(raw,start(raw,p),updates)
        return apply_readout(q,z,w)
    return predict


def layer():
    return compose(OUT)


def verify_cache(directory=OUT):
    trace=stack.verifier.trace
    prior=json.loads((parent.OUT/'reload_verification.json').read_text())
    prev=None
    for name in ('raw_replay','reload_verification'):
        w=np.load(directory/'model.npz')['w']
        rows=[]
        days={}
        all_stats=[]
        for day in range(16,20):
            raw=trace.replay.original.features(trace.replay.original.DATA,day)
            with np.load(stack.OUT/'components'/f'day_{day}.npz') as f:
                assert str(f['raw_hash'])==trace.replay.digest(raw)
                p,h,z,hz=[f[k] for k in ('full','head','z','hz')]
                assert f['hashes'].tolist()==[trace.replay.digest(a) for a in (p,h,z,hz)]
            q=apply_readout(p,z,w)
            h=apply_readout(h,hz,w)
            err=float(abs(q[:len(h)]-h).max())
            assert err<1e-4 and np.isfinite(q).all()
            y=trace.replay.labels(day)
            row=(p.reshape(-1,3),q.reshape(-1,3),y)
            stat=trace.older.measure([row])
            assert abs(stat['baseline_score']-prior['daily'][str(day+1)]['score'])<1e-10
            days[str(day+1)]=dict(score=stat['score'],gain=stat['gain'],prefix_error=err)
            rows.append(row)
            stats=[]
            for k in range(3):
                old=p.copy()
                old[:,:,2]=p[:,:,k]
                truth=y.reshape(p.shape).copy()
                truth[:,:,2]=truth[:,:,k]
                stats.append(stack.verifier.audit.blockstats(old,truth,[q[:,:,k].astype(float)-p[:,:,k]]))
            all_stats.append(np.stack(stats,axis=1))
        def gain(g):
            return float(stack.verifier.audit.pooled(g)[:,0]@np.array([.2,.3,.5]))
        rng=np.random.default_rng(3819)
        diffs=[]
        for i in range(1600):
            sampled=[all_stats[d][rng.integers(len(all_stats[d]),size=len(all_stats[d]))].sum(0) for d in rng.integers(4,size=4)]
            diffs.append(gain(sum(sampled)))
        pooled=trace.older.measure(rows)
        assert pooled['rows']==936250 and abs(pooled['gain']-gain(sum(a.sum(0) for a in all_stats)))<1e-10
        report=dict(parent=parent.OUT.name,pooled=pooled,daily=days,paired_score_percentiles=np.quantile(diffs,[.025,.5,.975]).tolist())
        if prev is not None:
            assert report==prev
        (directory/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(directory.name,name,pooled,flush=True)
        prev=report


if __name__=='__main__':
    torch.set_num_threads(2)
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ('train','all'):
        train()
    if mode in ('replay','all'):
        if not all((stack.OUT/'components'/f'day_{day}.npz').exists() for day in range(16,20)):
            stack.cache_components()
        verify_cache()
