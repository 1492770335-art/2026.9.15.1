"""Predict the excess payoff of the recurring tick-2401 price change using only pre-boundary information."""
import json
import numpy as np
import experiment_response_template_training_v1 as parent_model
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'recurring_boundary_payoff_v1'
book=parent_model.block.paths.book
BOUNDARY=2401
CONTROLS=(1561,1801,2041,2161,2641,2881,3121,3361)
H=np.array([5,20,60])
UNIT=np.array([.558942,.272164,.160943])
EDGES=np.array([0,1,2,4,12])


def frame(raw,t):
    def field(k):return np.arcsinh(raw[t,:,k].astype('float32'))
    r=raw[:,:,0].astype(float)-.00136
    cs=np.vstack((np.zeros(50),r.cumsum(0)))
    end=t+1
    flow=[field(0),field(10),field(11),field(14),field(21),field(26),field(38),
          field(41)-field(42),field(50)-field(51),field(54)-field(55),field(77)-field(78)]
    for width in (5,20,60,180):
        flow.append((cs[end]-cs[(end-width).clip(0)])/np.sqrt(width))
    x=np.tanh(np.stack(flow,2))
    m=x.mean(1,keepdims=True)
    spread=np.tanh(field(7))[:,:,None]
    loc=x-m
    local=np.concatenate((loc,loc*spread,loc*m),axis=2)
    common=np.concatenate((m,(x*spread).mean(1,keepdims=True)),axis=2)
    return local,common


def samples(raw,boundary):
    offsets=np.array([1,2,4,7,11,16,24,35,48,60])
    t=boundary-offsets
    r=raw[:,:,0].astype(float)-.00136
    target=np.stack([r[boundary+lo:boundary+hi].sum(0) for lo,hi in zip(EDGES[:-1],EDGES[1:])],axis=1)
    m=target.mean(0,keepdims=True)
    result=[]
    for x,y in zip(frame(raw,t),(target-m,m)):
        # History length changes how informative the last visible book is about the event.
        attenuation=np.exp(-offsets/15)[:,None,None]
        a=np.concatenate((x,x*attenuation),axis=2).reshape(-1,2*x.shape[2])
        b=np.broadcast_to(y,(len(t),*y.shape)).reshape(-1,4)
        result.append((a.mean(0),b.mean(0),a.T@a/len(a),a.T@b/len(a)))
    return result


def records(day):
    raw=book.rawday(day)
    placebo=[samples(raw,b) for b in CONTROLS]
    avg=[[np.mean(v,axis=0) for v in zip(*side)] for side in zip(*placebo)]
    return avg,samples(raw,BOUNDARY)


def fit(days):
    fitted=[]
    for regime in zip(*days):
        parts=[]
        for history in zip(*regime):
            centers,ym,grams,rhs=map(np.stack,zip(*history))
            x=centers.mean(0)
            z=ym.mean(0)
            # The sign of a common jump has very few independent days. Estimate only state-dependent changes.
            g=grams.mean(0)-np.outer(x,x)
            b=rhs.mean(0)-np.outer(x,z)
            reg=2.*np.diag(grams.mean(0).diagonal().clip(.001))+.002*np.eye(len(x))
            w=np.linalg.solve(g+reg,b)
            parts.append((x,w))
        fitted.append(parts)
    return fitted


def changes(raw,t,pars):
    result=[np.zeros((len(t),50,3),np.float32),np.zeros((len(t),1,3),np.float32)]
    use=(t<BOUNDARY)&(t+60>=BOUNDARY)
    if not use.any():return result
    sel=t[use]
    offset=BOUNDARY-sel
    for i,(x,out) in enumerate(zip(frame(raw,sel),result)):
        a=np.concatenate((x,x*np.exp(-offset/15)[:,None,None]),axis=2)
        p=[(a-c[i][0])@c[i][1] for c in pars]
        excess=(p[1]-p[0]).clip(-4.5,4.5)
        if i==0:excess-=excess.mean(1,keepdims=True)
        coverage=(H[None,:,None]-offset[:,None,None]+1-EDGES[None,None,:-1])/np.diff(EDGES)
        out[use]=.30*np.einsum('tsj,thj->tsh',excess,coverage.clip(0,1))*UNIT
    return result


def load():
    parent=parent_model.load()
    with np.load(OUT/'model.npz') as st:
        pars=[[(st[f'c{i}{j}'],st[f'w{i}{j}']) for j in range(2)] for i in range(2)]
        keep=st['mask']
    def apply(raw,p):
        q=parent(raw,p).copy()
        for v,k in zip(changes(raw,np.arange(len(raw)),pars),keep):
            q+=v*k
        return q
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    days,daily,folds,affected=[],[],[],[]
    for end in (8,10,12,14):
        days.extend(records(d) for d in range(len(days),end))
        pars=fit(days)
        groups=[[],[]]
        event_rows=[[],[]]
        for d in (end,end+1):
            raw=book.rawday(d)
            t=np.arange(len(raw))
            p=np.load(parent_model.block.segment.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r').copy()
            p[:,:,1]=np.load(parent_model.linear.earlier.OUT/f'y2_day_{d}.npy',mmap_mode='r')
            y=labels(d).reshape(p.shape)
            z=changes(raw,t,pars)
            for n in (BOUNDARY-23,BOUNDARY,BOUNDARY+6):
                for pre,full in zip(changes(raw[:n],t[:n],pars),z):
                    np.testing.assert_allclose(pre,full[:n],rtol=0,atol=2e-6)
            for v,g,e in zip(z,groups,event_rows):
                q=p+v
                g.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
                e.append(tuple(a[BOUNDARY-60:BOUNDARY].reshape(-1,3) for a in (p,q,y)))
            daily.append([parent_model.linear.gains(g[-1:]) for g in groups])
        folds.append([parent_model.linear.gains(g) for g in groups])
        affected.append([parent_model.linear.gains(e) for e in event_rows])
        print(end+1,'all-day local/market IC gains',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.asarray(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,affected=affected,retained=keep.tolist()),indent=2))
    if keep.any():
        days.extend(records(d) for d in range(len(days),16))
        st=dict(mask=keep)
        for i,parts in enumerate(fit(days)):
            for j,(c,w) in enumerate(parts):st.update({f'c{i}{j}':c,f'w{i}{j}':w})
        np.savez(OUT/'model.npz',**st)
        base=json.loads((parent_model.OUT/'reload_verification.json').read_text())['pooled']['score']
        for name in ('raw_replay','reload_verification'):
            r=parent_model.linear.best.online.best.roll.gls.best.walk.replay(load())
            if name=='reload_verification':
                prev=json.loads((OUT/'raw_replay.json').read_text())
                np.testing.assert_allclose(r['pooled']['candidate'],prev['pooled']['candidate'],atol=1e-10,rtol=0)
            (OUT/f'{name}.json').write_text(json.dumps(r,indent=2))
            print(name,r['pooled']['score'],flush=True)
            if r['pooled']['score']<=base:break


if __name__=='__main__':main()
