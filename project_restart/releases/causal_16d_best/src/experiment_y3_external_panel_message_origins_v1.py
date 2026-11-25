"""Transfer conditional forecasts of aggressive fills and passive withdrawals learned from synchronized US tapes."""
import hashlib
import json
import sys
import zipfile

import numpy as np

import experiment_lobster_event_replication_v1 as source
import joint_conditional_wait_references as history
import y3_four_day_neural_references as warm

parent=history.incumbent
base=parent.base
OUT=base.OUT.parent/'y3_external_panel_message_origins_v1'
ARMS=('unchanged','future price only','rotated message recipient','individual fill forecast',
      'peer contribution to future fills','peer contribution to withdrawals','peer fill minus withdrawal',
      'price-independent fill response','near versus late peer fill','signed execution spread across stocks',
      'market aggressive flow','market fill versus passive withdrawal','peer execution with confirming quote',
      'peer fill with opposing current price')
HORIZONS=((1,5),(6,20),(21,60))


def source_day():
    ends=np.arange(34205,57601,5)
    prices,imbalances,amounts,hashes=[],[],[],{}
    for name in source.SYMBOLS:
        file=source.trial.OUT/'lobster_samples'/f'{name}.zip'
        hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
        with zipfile.ZipFile(file) as ar:
            def read(kind,cols):
                member,=[k for k in ar.namelist() if kind in k]
                with ar.open(member) as s:
                    return np.loadtxt(s,delimiter=',',usecols=cols,unpack=True)
            tm,kind,vol,side=read('_message_',(0,1,3,5))
            ask,va,bid,vb=read('_orderbook_',(0,1,2,3))
        assert len(tm)==len(ask) and (np.diff(tm)>=0).all()
        ok=(bid>0)&(ask>=bid)&(ask<1e9)&(va>0)&(vb>0)
        ix=np.searchsorted(tm[ok],ends,side='right')-1
        assert ix.min()>=0
        mid=np.log(.5*(ask[ok][ix]+bid[ok][ix]))
        prices.append(5000*np.diff(mid,prepend=mid[0]))
        imbalances.append(((vb[ok]-va[ok])/(vb[ok]+va[ok]))[ix])
        label=[]
        for codes in ((4,5),(2,3)):
            active=np.isin(kind,codes)&ok
            weight=np.sqrt(vol[active]/100).clip(0,10)
            when=np.searchsorted(ends,tm[active],side='left')
            inside=when<len(ends)
            for w in (-side[active]*weight,weight):
                label.append(np.bincount(when[inside],weights=w[inside],minlength=len(ends)))
        amounts.append(np.stack(label,1))
    return np.stack(prices,1),np.stack(imbalances,1),np.stack(amounts,1),hashes


def inputs(ret,imb):
    n,ns=ret.shape
    t=np.arange(1,n+1)
    left=(t-80).clip(0)
    def cs(a):
        return np.vstack((np.zeros(ns),a.cumsum(0)))
    rv=cs(ret*ret)
    vol=np.sqrt((rv[t]-rv[left]+8)/(t-left+8)[:,None]).clip(.12,20)
    p=cs(ret)
    b=cs(imb)
    acc=[]
    for span in (1,6,24,80):
        start=(t-span).clip(0)
        signed=np.tanh((p[t]-p[start])/(vol*np.sqrt(span)))
        queue=(b[t]-b[start])/(t-start)[:,None]
        acc.extend((signed,queue,np.tanh(3*(imb-queue)),np.sign(signed)*queue**2))
    return np.stack(acc,2).astype('float32'),vol


def features(a):
    t,n,k=a.shape
    grouped=a.reshape(t,-1,5,k)
    others=((grouped.sum(2,keepdims=True)-grouped)*.25).reshape(t,n,k)
    broad=((np.sign(grouped).sum(2,keepdims=True)-np.sign(grouped))*.25).reshape(t,n,k)
    own=np.concatenate((a,a*np.abs(a)),2)
    return np.concatenate((own,others,broad,others*np.abs(a),a*abs(others),
                           others*np.tanh(a*others),np.tanh(2*(a-others))-(a-others)),2)


def learn():
    ret,imb,amt,hashes=source_day()
    a,vol=inputs(ret,imb)
    x=features(a)
    t=np.arange(80,len(ret)-60)
    cum=np.concatenate((np.zeros_like(amt[:1]),amt.cumsum(0)))
    price=np.vstack((np.zeros(ret.shape[1]),ret.cumsum(0)))
    targets=[]
    for first,last in HORIZONS:
        sums=cum[t+last+1]-cum[t+first]
        targets.append(np.stack(((price[t+last+1]-price[t+first])/(np.sqrt(last-first+1)*vol[t]),
                                 sums[:,:,0]/(sums[:,:,1]+2),sums[:,:,2]/(sums[:,:,3]+2)),axis=2))
    y=np.stack(targets,2).clip(-5,5)
    w0=a.shape[2]*2
    def fit(rows):
        xx=x[t[rows]].reshape(-1,x.shape[2]).astype(float)
        yy=y[rows].reshape(-1,9)
        wrong=np.roll(y[rows],2,axis=1).reshape(-1,9)
        g=xx.T@xx/len(xx)
        xy=xx.T@np.column_stack((yy,wrong))/len(xx)
        reg=.6*np.diag(g.diagonal().clip(.002))
        full=np.linalg.solve(g+reg,xy)
        own=np.zeros_like(full)
        own[:w0]=np.linalg.solve((g+reg)[:w0,:w0],xy[:w0])
        return np.stack((full,own)).astype('float32')
    cut=len(ret)*2//3
    hold=t>=cut
    frozen=fit(t+60<cut)
    z=x[t[hold]]@frozen[0,:,:9]
    yy=y[hold].reshape(*z.shape)
    ic=[float(np.corrcoef(z[:,:,i].ravel(), yy[:,:,i].ravel())[0,1]) for i in range(9)]
    return fit(np.ones(len(t),bool)),dict(source_date='2012-06-21',tick_seconds=5,
        held_ticks=int(hold.sum()),first_training_last_origin=int(t[t+60<cut][-1]),first_validation_origin=int(t[hold][0]),
        target_order=['price','aggressive fills','withdrawals'],source_holdout_ic=ic,source_sha256=hashes)


def design(raw,w,orders,lim):
    ret=base.raw_returns(raw)
    imb=(2*(raw[:,:,21].astype(float)-lim[0])/(lim[1]-lim[0])-1).clip(-1,1)
    result=np.empty((*ret.shape,len(ARMS)-1),np.float32)
    for lo,hi in base.segments(len(ret)):
        a,_=inputs(ret[lo:hi],imb[lo:hi])
        for start in range(0,len(a),144):
            stop=min(start+144,len(a))
            vec=np.zeros((stop-start,50,18),np.float32)
            local=np.zeros_like(vec)
            for order in orders:
                xx=features(a[start:stop,order])
                vec[:,order]+=xx@w[0]
                local[:,order]+=xx@w[1]
            vec/=len(orders)
            local/=len(orders)
            extra=vec-local
            flow,cancel=extra[:,:,7],extra[:,:,8]
            price=extra[:,:,6]
            direction=np.sign(a[start:stop,:,0])
            # flow*tanh(2*flow*imbalance) is even in flow; the current imbalance supplies its sign.
            vals=(vec[:,:,6],extra[:,:,16],local[:,:,7],flow,cancel,flow-cancel,flow-price,
                  extra[:,:,4]-flow,flow*(abs(flow)>abs(cancel)),flow,flow-cancel,
                  flow*np.tanh(2*flow*imb[lo+start:lo+stop]),flow*(flow*direction<0))
            out=np.stack(vals,2)
            mean=out.mean(1,keepdims=True)
            out-=mean
            out[:,:,9:11]=mean[:,:,9:11]
            result[lo+start:lo+stop]=out
    return result


def imbalance_bounds():
    slices=[history.trial.clock.book.rawday(d)[:,:,21] for d in range(4)]
    return min(float(v.min()) for v in slices),max(float(v.max()) for v in slices)


def train():
    OUT.mkdir(exist_ok=True)
    w,diagnostic=learn()
    lim=imbalance_bounds()
    rng=np.random.default_rng(17331)
    orders=np.stack([rng.permutation(50) for _ in range(3)])
    rows,folds=[],[]
    for d in range(4,16):
        if d in (8,12):
            coef=base.calibrate(rows)
            fold=[]
        raw=history.trial.clock.book.rawday(d)
        z=design(raw,w,orders,lim)
        if d==4:
            np.testing.assert_allclose(z[:113],design(raw[:113],w,orders,lim),rtol=0,atol=2e-6)
        y=history.trial.clock.labels(d).reshape(-1,50,3)
        ps=history.references(d,raw) if d>=8 else warm.read(d,raw)
        rec=[base.records(p,y,z) for p in ps]
        rows.append(rec)
        if d>=8:
            fold.append(rec)
            if d in (11,15):
                folds.append([base.score(v,coef) for v in np.sum(fold,axis=0)])
    f=np.asarray(folds)
    good=[i for i in range(4,len(ARMS)) if f[:,:,0,i].min()>0 and
          all((f[:,:,0,i].mean(0)>f[:,:,0,j].mean(0)).all() for j in (1,2,3))]
    pick=max(good,key=lambda i:f[:,:,0,i].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,folds=f.tolist(),source=diagnostic),indent=2))
    print('selected',ARMS[pick],'external holdout',diagnostic['source_holdout_ic'],flush=True)
    if pick:
        np.savez(OUT/'model.npz',w=w,lim=lim,orders=orders,pick=pick-1,coef=base.calibrate(rows)[pick-1])
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        w,orders,lim,idx,coef=f['w'],f['orders'],f['lim'],int(f['pick']),float(f['coef'])
    return lambda raw:(design(raw,w,orders,lim)[:,:,idx]*coef).astype('float32')


def layer():
    before,delta=parent.layer(),correction()
    def predict(raw,initial):
        p=before(raw,initial).copy()
        p[:,:,2]+=delta(raw)
        return p
    return predict


def replay_with():
    delta=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        p[:,:,2]+=delta(raw)
        h[:,:,2]+=delta(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
