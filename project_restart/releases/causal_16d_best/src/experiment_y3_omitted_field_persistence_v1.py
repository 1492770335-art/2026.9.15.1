"""Extract forecastable states from the 84 fields outside the path backbone, conditional on the original 16."""
import json
import sys

import numpy as np

import experiment_y3_oriented_book_paths_v1 as path

history,parent,base=path.history,path.parent,path.base
OUT=parent.OUT.parent/'y3_omitted_field_persistence_v1'
CORE=np.array(path.FIELDS)
OTHER=np.setdiff1d(np.arange(100),CORE)
K=10
ARMS=('unchanged','core fields','instantaneous remainder','wrong stock future','next five',
      'next twenty','distant forty','distant above first five','distant replicated on alternate dates',
      'distant complement to short states','joint near and distant','distant nonlinear')


def views(raw,t,future=False):
    core=[]
    current=[]
    target=[]
    for lo,hi in base.segments(len(raw)):
        ix=t[(t>=lo)&(t<hi)]-lo
        if not len(ix):
            continue
        v=np.tanh(.7*np.arcsinh(raw[lo:hi].astype(np.float32)))
        # Treat the categorical flags as signed, bounded fields in all views.
        v[:,:,83:]=raw[lo:hi,:,83:].astype(np.float32)-1.
        c=np.concatenate((np.zeros_like(v[:1]),v.cumsum(0,dtype=float)),0)
        right=ix+1
        def past(h):
            left=(right-h).clip(0)
            return (c[right]-c[left])/(right-left)[:,None,None]
        fast,slow=past(6),past(80)
        cur=v[ix]
        core.append(np.concatenate((cur[:,:,CORE],fast[:,:,CORE],slow[:,:,CORE]),2))
        now=cur[:,:,OTHER]-slow[:,:,OTHER]
        current.append(now)
        if future:
            assert right.max()+60<=len(c)-1
            bank=[]
            for a,b in ((0,5),(0,20),(20,60)):
                later=((c[right+b]-c[right+a])/(b-a))[:,:,OTHER]
                bank.append(later-slow[:,:,OTHER])
            target.append(np.concatenate(bank+[np.roll(bank[2],13,1)],2))
    def pack(rows):
        x=np.concatenate(rows).astype(np.float32)
        return x-x.mean(1,keepdims=True)
    return pack(core),pack(current),(pack(target) if future else None)


def stats(day):
    raw=history.trial.clock.book.rawday(day)
    t=np.concatenate([np.arange(a+day%14,b-60,14) for a,b in base.segments(len(raw))])
    c,x,y=views(raw,t,True)
    feat=np.concatenate((c,x),2).reshape(-1,c.shape[2]+x.shape[2]).astype(float)
    response=y.reshape(len(feat),-1).astype(float)
    return feat.T@feat/len(feat),feat.T@response/len(feat),np.square(response).mean(0)


def axes(rows):
    g,b,energy=[np.mean(v,0) for v in zip(*rows)]
    n=3*len(CORE)
    k=len(OTHER)
    w=np.linalg.solve(g[:n,:n]+.04*np.diag(g.diagonal()[:n].clip(.001)),g[:n,n:])
    cond=np.vstack((-w,np.eye(k)))
    cov=cond.T@g@cond
    root=np.linalg.cholesky(cov+.10*np.diag(cov.diagonal().clip(.001)))
    wh=np.linalg.inv(root)
    pieces=[]
    for j in range(4):
        sl=slice(j*k,(j+1)*k)
        cross=wh@cond.T@b[:,sl]/np.sqrt(energy[sl].clip(.002))
        u,s,_=np.linalg.svd(cross,full_matrices=False)
        pieces.append(u[:,:K])
    high=pieces[2]
    early=pieces[0]
    # Choose remote modes insensitive to the dominant predictable short-term modes.
    remote=high-early@(early.T@high)
    u,s,_=np.linalg.svd(remote,full_matrices=False)
    pieces.append(u[:,:K]*s)
    def operator(subset):
        cross=np.mean([m[1] for m in subset],0)[:,2*k:3*k]
        return wh@cond.T@cross/np.sqrt(energy[2*k:3*k].clip(.002))
    a,d=operator(rows[::2]),operator(rows[1::2])
    val,v=np.linalg.eigh(.5*(a@d.T+d@a.T))
    pieces.append(v[:,-K:][:,::-1])
    # All modes are linear maps of observations available at the forecast origin.
    _,v=np.linalg.eigh(cov)
    vals=np.stack([wh.T@q for q in pieces]+[v[:,-K:]/np.sqrt(np.linalg.eigvalsh(cov)[-K:].clip(.001))],2)
    cvals,cvec=np.linalg.eigh(g[:n,:n])
    return w,vals,cvec[:,-K:]/np.sqrt(cvals[-K:].clip(.001))


def design(raw,t,state):
    w,bank,cw=state
    c,x,_=views(raw,t)
    rem=x-c@w
    vals=(rem@bank.reshape(len(OTHER),-1)).reshape(len(t),50,K,-1).astype(np.float32).clip(-6,6)
    near,medium,far,wrong,unique,stable,instant=np.moveaxis(vals,3,0)
    core=(c@cw).astype(np.float32).clip(-6,6)
    def expand(v):
        return np.concatenate((v,np.tanh(v),v*np.tanh(np.abs(v)-.5)),2)
    # A difference of coordinate indices from unrelated eigenbases has no invariant meaning.
    # Remove the early subspace before reading out the remaining far states.
    both=np.concatenate((near[:,:,:K//2],far[:,:,:K//2]),2)
    fields=(core,instant,wrong,near,medium,far,unique,stable,unique,both,far)
    out=[]
    for j,v in enumerate(fields):
        z=expand(v)
        if j not in (7,8,10):
            z[:,:,K:]=0
        out.append(z-z.mean(1,keepdims=True))
    return np.stack(out,2)


def moments(x,p,y):
    n=x.shape[-1]
    xs=x.transpose(2,0,1,3).reshape(x.shape[2],-1,n).astype(float)
    targets=[]
    for k in (0,1):
        a=p[:,:,2]-k*base.RATIO*p[:,:,1]
        b=y[:,:,2]-k*base.RATIO*y[:,:,1]
        a-=a.mean(1,keepdims=True)
        b-=b.mean(1,keepdims=True)
        beta=np.sum(a*b,dtype=float)/max(np.sum(a*a,dtype=float),1e-7)
        targets.append((b/beta.clip(.35,2.8)-a).ravel())
    return xs.transpose(0,2,1)@xs/xs.shape[1],xs.transpose(0,2,1)@np.stack(targets,1)/xs.shape[1]


def calibrate(raws,refs,targets,state,days):
    rows=[]
    for raw,pair,y,d in zip(raws,refs,targets,days):
        t=np.arange(d%11,len(raw),11)
        x=design(raw,t,state)
        obs=[moments(x,p[t],y[t]) for p in pair]
        rows.append(tuple(np.mean(v,0) for v in zip(*obs)))
    gs,bs=[np.stack(v) for v in zip(*rows)]
    mu=bs.mean(0)
    mu*=abs(mu)/(abs(mu)+bs.std(0)/np.sqrt(len(rows))+.00001)
    a,b=mu[:,:,0],mu[:,:,1]
    z=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    return .28*np.stack([np.linalg.solve(m+1.8*np.diag(m.diagonal().clip(.001)),v) for m,v in zip(gs.mean(0),z)])


def read_day(d):
    raw,y,p=parent.records(d)
    return raw,(history.references(d,raw) if d>=8 else p),y


def fitted(d,rows):
    st=axes(rows[:d])
    # Recompute supervised coefficients in the *same* basis used on the next fold.
    raws,refs,targets=zip(*(read_day(day) for day in range(4,d)))
    return st,calibrate(raws,refs,targets,st,range(4,d))


def forecast(raw,st,w):
    out=[]
    # Restrict the size of the dense field design. Eighty earlier observations suffice for every feature.
    for lo,hi in base.segments(len(raw)):
        for a in range(lo,hi,180):
            start=max(lo,a-79)
            end=min(a+180,hi)
            bounded=raw[start:end]
            x=design(bounded,np.arange(a-start,end-start),st)
            out.append(np.einsum('tsgf,gf->tsg',x,w).astype(np.float32))
    return np.concatenate(out)


def train():
    OUT.mkdir(exist_ok=True)
    rows=[stats(day) for day in range(16)]
    folds,daily=[],[]
    for end in (8,12):
        st,w=fitted(end,rows)
        group=[]
        for d in range(end,end+4):
            raw,ps,y=read_day(d)
            z=forecast(raw,st,w)
            if d==8:
                for n in (433,2427):
                    np.testing.assert_allclose(z[:n],forecast(raw[:n],st,w),atol=1e-6,rtol=0)
            ms=[base.records(p,y,z) for p in ps]
            group.append(ms)
            daily.append([base.score(m,np.ones(len(ARMS)-1))[0] for m in ms])
        folds.append([base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(group,0)])
    f,ds=np.array(folds),np.array(daily)
    ok=[i for i in range(4,len(ARMS)) if f[:,:,:,i].min()>0 and (ds[:,:,i]>0).sum(0).min()>=5 and
        (f[:,:,0,i].mean(0)>f[:,:,0,1:4].mean(0).max(1)).all()]
    pick=max(ok,key=lambda i:f[:,:,0,i].min()) if ok else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),daily=ds.tolist(),windows=[[9,12],[13,16]]),indent=2))
    print('selected',ARMS[pick],f[:,0,0].round(6).tolist(),flush=True)
    if pick:
        (w,bank,cw),coef=fitted(16,rows)
        np.savez(OUT/'model.npz',conditional=w,bank=bank,core=cw,coef=coef,pick=pick-1)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        st=f['conditional'],f['bank'],f['core']
        w,pick=f['coef'],int(f['pick'])
    return lambda raw:forecast(raw,st,w)[:,:,pick]


def layer():
    before,update=parent.layer(),correction()
    def apply(raw,initial):
        p=before(raw,initial).copy()
        p[:,:,2]+=update(raw)
        return p
    return apply


def replay_with():
    update=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        p[:,:,2]+=update(raw)
        h[:,:,2]+=update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
