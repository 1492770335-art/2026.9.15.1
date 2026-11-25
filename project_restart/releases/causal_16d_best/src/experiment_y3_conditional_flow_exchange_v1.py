"""Compare each stock's flow magnitude with matched contemporaneous and lagged alternative recipients."""
import json
import sys

import numpy as np

import y3_panel_message_origin_references as history

parent,base=history.incumbent,history.base
OUT=parent.OUT.parent/'y3_conditional_flow_exchange_v1'
ARMS=('unchanged','own magnitude','single displaced magnitude','all-other gate',
      'own minus other names','own minus similar activity','own minus similar activity and imbalance',
      'own minus similar signed imbalance','own minus historical magnitude','within-stock versus peer contrast',
      'excess above matching recipients only','deficit below matching recipients only',
      'conditional versus unconditional benchmark','group-assignment control')
CONTROLS=(1,2,3,13)


def source_state():
    with np.load(parent.OUT/'model.npz') as f:
        return dict(w=(f['w'][0,:,7]-f['w'][1,:,7]),orders=f['orders'],lim=f['lim'],groups=base.pace()[0])


def flow(raw,st):
    lo,hi=st['lim']
    b=(2*(raw[:,:,21].astype(float)-lo)/(hi-lo)-1).clip(-1,1)
    f=np.zeros(raw.shape[:2],np.float32)
    for left,right in base.segments(len(raw)):
        a,_=parent.inputs(base.raw_returns(raw[left:right]),b[left:right])
        for order in st['orders']:
            for i in range(0,len(a),144):
                stop=min(i+144,len(a))
                f[left+i:left+stop,order]+=(parent.features(a[i:stop,order])@st['w'])/len(st['orders'])
    return np.abs(f),b


def design(raw,st):
    f,b=flow(raw,st)
    n=len(raw)
    out=np.empty((n,50,len(ARMS)-1),np.float32)
    gid=st['groups']
    pair=gid[:,None]==gid[None,:]
    others=~np.eye(50,dtype=bool)
    for lo,hi in base.segments(n):
        for i in range(lo,hi,96):
            t=np.arange(i,min(hi,i+96))
            a=f[t]
            z=b[t]
            all_gates=a[:,None,:]*np.tanh(2*z[:,:,None]*a[:,None,:])
            actual=all_gates.diagonal(axis1=1,axis2=2)
            global_gate=(all_gates*others).sum(2)/49
            def matched(weight):
                return ((all_gates*weight).sum(2)+2*global_gate)/(weight.sum(-1)+2)
            group=matched((others&pair)[None])
            dist=abs(abs(z)[:,:,None]-abs(z)[:,None,:])
            close=matched((others&pair)[None]*np.exp(-4*dist))
            signed=matched(others[None]*np.exp(-4*abs(z[:,:,None]-z[:,None,:])))
            wrong=np.roll(a,13,axis=1)
            history_gate=np.mean([v*np.tanh(2*z*v) for v in [f[np.maximum(lo,t-k)] for k in (12,24,48)]],axis=0)
            sign=np.sign(z)
            diff=(actual-close)*sign
            fake=matched((np.roll(pair,13,axis=0)&others)[None]*np.exp(-4*dist))
            bank=np.stack((actual,wrong*np.tanh(2*z*wrong),global_gate,actual-global_gate,actual-group,
                           actual-close,actual-signed,actual-history_gate,history_gate-close,diff.clip(0)*sign,
                           diff.clip(None,0)*sign,close-global_gate,actual-fake),2)
            out[t]=bank-bank.mean(1,keepdims=True)
    return out


def train():
    OUT.mkdir(exist_ok=True)
    st=source_state()
    rows,folds,days=[],[],[]
    for d in range(4,16):
        if d in (8,12):
            w=base.calibrate(rows)
            fold=[]
        raw=history.trial.clock.book.rawday(d)
        z=design(raw,st)
        if d==4:
            for n in (131,2440):
                np.testing.assert_allclose(design(raw[:n],st),z[:n],atol=8e-7,rtol=0)
        y=history.trial.clock.labels(d).reshape(-1,50,3)
        pp=history.references(d,raw) if d>=8 else parent.warm.read(d,raw)
        r=[base.records(p,y,z) for p in pp]
        rows.append(r)
        if d>=8:
            days.append([base.score(m,w) for m in r]);fold.append(r)
            if d in (11,15):
                folds.append([base.score(m,w) for m in np.sum(fold,axis=0)])
    f=np.asarray(folds)
    daily=np.asarray(days)
    ok=[i for i in range(4,13) if f[:,:,0,i].min()>0 and (daily[:,:,0,i]>0).sum(0).min()>=5 and
        (f[:,:,0,i].mean(0)>f[:,:,0,1].mean(0)).all()]
    pick=max(ok,key=lambda i:f[:,:,0,i].min()) if ok else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=f.tolist(),daily=daily.tolist()),indent=2))
    print('selected',ARMS[pick],f[:,:,0].round(7),flush=True)
    if pick:
        np.savez(OUT/'model.npz',**st,coef=base.calibrate(rows)[pick-1],pick=pick-1)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f: st=dict(f)
    return lambda raw:(design(raw,st)[:,:,int(st['pick'])]*float(st['coef'])).astype('float32')


def layer():
    before,delta=parent.layer(),correction()
    def predict(raw,initial):
        q=before(raw,initial).copy()
        q[:,:,2]+=delta(raw)
        return q
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
