"""Query previously settled stock errors using the learned twenty-step bilinear response as attention scores."""
import argparse
import json
import numpy as np

import experiment_compounded_return_audit_v1 as tape
import experiment_y3_settled_stock_gradient_v1 as gradient
import experiment_y3_peer_response_fingerprints_v1 as peers
from y3_portfolio_walkforward import historical_panels,parent,scores,replay

OUT=parent.ART/'y3_settled_error_attention_v1'
ARMS=('current settled-gradient model','linear similarity','signed softmax','positive similarity',
      'softmax minus linear','median-centered similarity','reassigned errors','agreement of separate history blocks')


def lookup(raw,p,w,geo):
    k,coef,template=geo
    rms=np.sqrt(np.square(w).sum()).clip(.0001)
    mat=w/rms
    result=[]
    for lo,hi in parent.base.segments(len(raw)):
        r,q=raw[lo:hi],p[lo:hi]
        f=np.tanh(.8*np.arcsinh(r[:,:,[26,38,41,42]].astype(np.float32)))
        x=np.concatenate((q,f[:,:,::2]-f[:,:,1::2],np.tanh(r[:,:,0:1].astype(np.float32)-.00136)),2)
        x-=x.mean(1,keepdims=True)
        x=np.tanh(x/(.08+np.sqrt((x*x).mean(1,keepdims=True))))
        price=tape.compound(r,k)
        missing=np.vstack((np.zeros(50,int),(~tape.present(r,template)).cumsum(0)))
        for start in range(0,len(r),gradient.BLOCK):
            stop=min(start+gradient.BLOCK,len(r))
            t=np.arange(max(0,start-960),start-20,20)
            if len(t)<4:
                result.append(np.zeros((stop-start,50,len(ARMS)-1),np.float32))
                continue
            valid=missing[t+21]==missing[t]
            err=coef[1,0]+coef[1,1]*tape.span(price,t,20,k)-q[t,:,1]
            err=(err-(err*valid).sum(1,keepdims=True)/valid.sum(1,keepdims=True).clip(1)).clip(-3,3).T
            weight=(np.exp(-(start-20-t)/480)[:,None]*valid).T
            score=(x[start:stop].transpose(1,0,2)@mat)@x[t].transpose(1,2,0)
            def read(similar,wt,values):
                wt=wt[:,None,:]
                total=wt.sum(2,keepdims=True).clip(.00001)
                centered=similar-(wt*similar).sum(2,keepdims=True)/total
                return ((centered*wt)@values[:,:,None])[:,:,0]/(total[:,:,0]+4)
            base=read(score,weight,err)
            def soft(wt):
                def tilt(s):
                    z=np.exp(np.clip(s,-5,5))
                    mass=wt[:,None,:]*z
                    return (mass@err[:,:,None])[:,:,0]/mass.sum(2).clip(.001)
                n=wt.sum(1,keepdims=True)
                return .5*(tilt(2*score)-tilt(-2*score))*n/(n+5)/2
            attention=soft(weight)
            positive=read(np.maximum(score,0),weight,err)
            centered_score=np.tanh((score-np.median(score,axis=2,keepdims=True))/
                             (.25+np.median(abs(score),axis=2,keepdims=True)))
            even,odd=[soft(weight*((t//96)%2==i)) for i in (0,1)]
            replicated=np.where(even*odd>0,np.sign(even)*np.minimum(abs(even),abs(odd)),0)
            z=np.stack((base,attention,positive,attention-base,read(centered_score,weight,err),
                        read(score,weight,np.roll(err,13,axis=0)),replicated),2).transpose(1,0,2)
            result.append((.18*rms*(z-z.mean(1,keepdims=True))).astype(np.float32))
    return np.concatenate(result)


def train():
    hist,donor,profiles,days,locked=[],[],[],[],[]
    geo=gradient.historical_geometry()
    for d,raw,ps,y in historical_panels():
        if d>=8:
            if d%2==0:
                w=gradient.fit(hist);peer=peers.fit(donor,profiles)
            if d==8:anchor=w
            qs=[];rr=[];ff=[]
            for i,p in enumerate(ps):
                q=p.copy();q[:,:,2]+=peers.signals(raw,p,peer)[:,:,4]
                qs.append(q)
                r=q.copy();r[:,:,2]+=gradient.shifts(raw,q,geo,w)[:,:,2]
                z=lookup(raw,q,w[1,0],geo)
                if d==8 and i==0:
                    for n in (243,2473):np.testing.assert_allclose(z[:n],lookup(raw[:n],q[:n],w[1,0],geo),atol=1e-6,rtol=0)
                rr.append(parent.base.records(r,y,z))
                ff.append(parent.base.records(r,y,lookup(raw,q,anchor[1,0],geo)))
            days.append(rr);locked.append(ff)
        else:qs=ps
        hist.append(gradient.record(raw,np.mean(qs[:2],0),y,d,geo))
        obs=[peers.observed(raw,p,y,d) for p in ps[:2]]
        donor.append([tuple(np.mean(a,0) for a in zip(*s)) for s in zip(*obs)])
        profiles.append(peers.fingerprint(raw))
        print('processed day',d+1,flush=True)
    def halves(rows):return np.stack([scores(v,len(ARMS)) for v in (rows[:4],rows[4:])])
    h,f=halves(days),halves(locked)
    dd=np.stack([scores([v],len(ARMS)) for v in days])
    pick=max((2,3,4,5,7),key=lambda i:h[0,:,0,i].min())
    keep=(h[:,:,0,pick].min()>0 and f[:,-1,0,pick].min()>0 and (dd[:,-1,0,pick]>0).sum()>=5
          and (h[:,:,0,pick]-h[:,:,0,6]).mean()>0)
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,windows=[[9,12],[13,16]],first_window_choice=pick,
             selected=pick if keep else 0,halves=h.tolist(),frozen=f.tolist(),daily=dd.tolist()),indent=2))
    if keep:np.savez(OUT/'model.npz',pick=pick,w=gradient.fit(hist)[1,0])
    print('selected',pick if keep else 0,'strict',h[:,-1,0].round(6).tolist(),flush=True)


def correction():
    primary=gradient.correction(gradient.PRIMARY_OUT);first=peers.correction()
    geo=gradient.historical_geometry()
    with np.load(OUT/'model.npz') as f:w,pick=f['w'],int(f['pick'])
    def update(raw,p):
        q=p.copy();q[:,:,2]+=first(raw,p)
        return primary(raw,p)+lookup(raw,q,w,geo)[:,:,pick-1]
    return update


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),nargs='?',default='train')
    parser.add_argument('--upstream-dir')
    args=parser.parse_args()
    if args.mode=='train':train()
    else:
        if args.mode=='screen' and not args.upstream_dir:parser.error('supply the upstream directory')
        replay(OUT,correction,args.upstream_dir if args.mode=='screen' else None)
