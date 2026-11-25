"""Predict long-horizon errors from current features and earlier, fully settled short-forecast errors."""
import argparse
import json
import numpy as np

import experiment_compounded_return_audit_v1 as tape
import experiment_y3_peer_response_fingerprints_v1 as peers
from experiment_y3_settled_factor_confidence_v1 import historical_geometry
from y3_portfolio_walkforward import historical_panels,parent,scores,replay

OUT=parent.ART/'y3_settled_stock_gradient_v1'
PRIMARY_OUT=parent.ART/'y3_settled_stock_gradient_primary_v1'
BLOCK=96
D=6
ARMS=('incumbent','misassigned response','five-tick gradient','twenty-tick gradient','joint gradient',
      'agreement across alternating blocks','diagonal only','direction mixing','replicated across training dates')


def chunks(raw,p,geo):
    k,coeff,missing=geo
    for lo,hi in parent.base.segments(len(raw)):
        a=raw[lo:hi];q=p[lo:hi]
        field=np.tanh(.8*np.arcsinh(a[:,:,[26,38,41,42]].astype(np.float32)))
        flow=field[:,:,::2]-field[:,:,1::2]
        x=np.concatenate((q,flow,np.tanh(a[:,:,0:1].astype(np.float32)-.00136)),2)
        x-=x.mean(1,keepdims=True)
        x=np.tanh(x/(.08+np.sqrt((x*x).mean(1,keepdims=True)))).astype(np.float32)
        path=tape.compound(a,k)
        errors=np.vstack((np.zeros(50,int),(~tape.present(a,missing)).cumsum(0)))
        for start in range(0,len(a),BLOCK):
            stop=min(start+BLOCK,len(a))
            profiles=[]
            for j,h in enumerate((5,20)):
                t=np.arange(max(0,start-960),start-h,h)
                if len(t)==0:
                    profiles.append(np.zeros((2,50,D),np.float32))
                    continue
                valid=errors[t+h+1]==errors[t]
                actual=coeff[j,0]+coeff[j,1]*tape.span(path,t,h,k)
                err=actual-q[t,:,j]
                err=(err-(err*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)).clip(-3,3)
                # Estimate response on disjoint blocks. No origin extends into the block being scored.
                aligned=[]
                for part in (0,1):
                    weight=(np.exp(-(start-t-h)/480)*((t//BLOCK)%2==part))[:,None]*valid
                    val=x[t]
                    numer=(val*(weight*err)[:,:,None]).sum(0)
                    denom=((val*val)*weight[:,:,None]).sum(0)
                    beta=numer/(denom+4)
                    aligned.append(beta-beta.mean(0))
                profiles.append(np.stack(aligned))
            yield np.arange(lo+start,lo+stop),x[start:stop],np.stack(profiles).astype(np.float32)


def evidence(v):
    avg=v.mean(1)
    a,b=v[:,0],v[:,1]
    stable=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    return avg,stable


def basis(x,g):
    return (x[:,:,:,None]*g[None,:,None,:]).reshape(len(x),50,-1)


def record(raw,p,y,d,geo):
    a,b=parent.exposure.bands(p),parent.exposure.bands(y)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    unit=((a*b).sum((0,1))/(a*a).sum((0,1)).clip(.001)).clip(.45,2.5)
    e=(b/unit-a)@np.array([.62,.38])
    cells=[]
    for t,x,g in chunks(raw,p,geo):
        keep=t%7==d%7
        for _,end in parent.base.segments(len(raw)):
            keep &= (t<end-60)|(t>=end)
        if not keep.any():continue
        avg,stable=evidence(g)
        out=[]
        for h in (*avg,stable.mean(0),np.roll(avg.mean(0),17,axis=0)):
            z=basis(x[keep],h)
            z-=z.mean(1,keepdims=True)
            z=z.reshape(-1,D*D).astype(float)
            out.append((z.T@z,z.T@e[t[keep]].ravel()))
        cells.append(out)
    return [(np.sum(gs,0)/len(y)/50*7,np.sum(bs,0)/len(y)/50*7)
            for gs,bs in (zip(*byhead) for byhead in zip(*cells))]


def fit(history):
    bank=[]
    for byhead in zip(*history):
        gs,bs=[np.stack(v) for v in zip(*byhead)]
        def solve(g,b,idx):
            gram=g[:,idx][:,:,idx].mean(0)
            mean=b[:,idx].mean(0)
            mean*=abs(mean)/(abs(mean)+b[:,idx].std(0)/np.sqrt(len(b))+.00007)
            w=np.zeros(D*D)
            w[idx]=np.linalg.solve(gram+1.8*np.diag(gram.diagonal().clip(.00035)),mean)
            return w.reshape(D,D)
        ids=np.arange(D*D)
        a,b=[solve(gs[j::2],bs[j::2],ids) for j in (0,1)]
        bank.append(np.stack((solve(gs,bs,ids),solve(gs,bs,np.arange(0,D*D,D+1)),
                              np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0))))
    return np.stack(bank).astype(np.float32)


def shifts(raw,p,geo,state):
    allrows=[]
    for t,x,g in chunks(raw,p,geo):
        avg,stable=evidence(g)
        vectors=[*avg,stable.mean(0),np.roll(avg.mean(0),17,axis=0)]
        fitted=[]
        for v,w in zip(vectors,state):
            response=np.einsum('kij,sj->sik',w,v)
            fitted.append(np.einsum('tsi,sik->tsk',x,response))
        near,medium,agree,wrong=fitted
        mix=.5*(near+medium)
        z=np.stack((wrong[:,:,0],near[:,:,0],medium[:,:,0],mix[:,:,0],agree[:,:,0],
                    mix[:,:,1],mix[:,:,0]-mix[:,:,1],mix[:,:,2]),2)
        allrows.append(.38*(z-z.mean(1,keepdims=True)))
    return np.concatenate(allrows)


def train(primary=False):
    history,donor,profiles,days,frozen=[],[],[],[],[]
    geo=historical_geometry()
    for d,raw,ps,y in historical_panels():
        if d>=8:
            if d%2==0:
                w=fit(history);peer=peers.fit(donor,profiles)
            if d==8:anchor=w
            qs=[];rr=[];ff=[]
            for i,p in enumerate(ps):
                q=p.copy();q[:,:,2]+=peers.signals(raw,p,peer)[:,:,4]
                qs.append(q);z=shifts(raw,q,geo,w)
                if d==8 and i==0:
                    for n in (37,491,2471):np.testing.assert_allclose(z[:n],shifts(raw[:n],q[:n],geo,w),atol=2e-6,rtol=0)
                rr.append(parent.base.records(q,y,z))
                ff.append(parent.base.records(q,y,shifts(raw,q,geo,anchor)))
            days.append(rr);frozen.append(ff)
        else:qs=ps
        history.append(record(raw,np.mean(qs[:2],0),y,d,geo))
        obs=[peers.observed(raw,p,y,d) for p in ps[:2]]
        donor.append([tuple(np.mean(a,0) for a in zip(*s)) for s in zip(*obs)])
        profiles.append(peers.fingerprint(raw))
        print('processed day',d+1,flush=True)
    def halves(rows):return np.stack([scores(v,len(ARMS)) for v in (rows[:4],rows[4:])])
    h,f=halves(days),halves(frozen)
    daily=np.stack([scores([r],len(ARMS)) for r in days])
    pick=max(range(2,len(ARMS)),key=lambda i:h[0,:,0,i].min())
    keep=(h[:,:,0,pick].min()>0 and f[:,-1,0,pick].min()>0 and
          (daily[:,-1,0,pick]>0).sum()>=5 and (h[:,:,0,pick]-h[:,:,0,1]).mean()>0)
    if primary:
        # This exploratory check keeps the first-window winner and assesses the official
        # endpoint on the chronological reference; older architectures serve as sensitivity checks.
        keep=(pick==3 and h[:,-1,0,pick].min()>0 and f[:,-1,0,pick].min()>0 and
              (daily[:,-1,0,pick]>0).sum()>=6 and h[:,:,0,pick].mean(1).min()>0)
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,windows=[[9,12],[13,16]],first_window_choice=pick,
             selected=pick if keep else 0,halves=h.tolist(),frozen=f.tolist(),daily=daily.tolist()),indent=2))
    if keep:np.savez(OUT/'model.npz',pick=pick,w=fit(history))
    print('selected',pick if keep else 0,'strict',h[:,-1,0].round(6).tolist(),flush=True)


def correction(directory=None):
    if directory is None:
        directory=OUT
    prior=peers.correction();geo=historical_geometry()
    with np.load(directory/'model.npz') as f:pick,w=int(f['pick']),f['w']
    def update(raw,p):
        prev=prior(raw,p)
        q=p.copy();q[:,:,2]+=prev
        return prev+shifts(raw,q,geo,w)[:,:,pick-1]
    return update


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),nargs='?',default='train')
    parser.add_argument('--upstream-dir')
    parser.add_argument('--primary',action='store_true')
    args=parser.parse_args()
    if args.primary:OUT=PRIMARY_OUT
    if args.mode=='train':train(args.primary)
    else:
        if args.mode=='screen' and not args.upstream_dir:parser.error('supply the upstream directory')
        replay(OUT,correction,args.upstream_dir if args.mode=='screen' else None)
