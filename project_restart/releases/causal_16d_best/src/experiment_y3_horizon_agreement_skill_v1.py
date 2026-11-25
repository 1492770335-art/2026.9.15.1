"""Price a peer horizon state separately for stocks whose short and long forecasts agree or diverge."""
import json
import sys
import numpy as np
import torch
import experiment_y3_conditional_forecast_skill_v1 as prior
import y3_horizon_geometry_training as study
import joint_sibling_horizon_skill_references as history

parent,trial,stack = history.incumbent,history.trial,history.stack
OUT = trial.clock.ART/'y3_horizon_agreement_skill_v1'
ARMS = ('unchanged','pooled horizon state','peer agreement contrast','receiver agreement','matched sender and receiver',
        'combined pooled and conditional gradients','crossed agreement','most confident senders','wrong receiver identity')
CANDIDATES = (2,3,4,5,6,7)
CONTROLS = (1,8)


def geometry(x,weight):
    g = (x*weight[:,:,None]).transpose(0,2,1)@x/weight.sum(1)[:,None,None].clip(1)
    slow = study.ema(g,128)
    fast = .8*study.ema(g,6)+.2*slow
    a,b = [np.sqrt(m.diagonal(axis1=1,axis2=2)+.003) for m in (fast,slow)]
    corr = fast/a[:,:,None]/a[:,None,:]-slow/b[:,:,None]/b[:,None,:]
    return np.column_stack((np.log(a/b),corr[:,[0,0,1],[1,2,2]]))


def views(z,side):
    peer,own = z[:,side::2],z[:,1-side::2]
    def score(x):
        sd = np.sqrt((x*x).mean(1,keepdims=True)+.003)
        v = x/sd
        return np.tanh(.65*(v[:,:,0]+v[:,:,1])*v[:,:,2])
    sender,receiver = score(peer),score(own)
    u = geometry(peer,np.ones_like(sender))
    pos,neg = [geometry(peer,1+sign*.85*sender) for sign in (1,-1)]
    return own,np.column_stack((np.ones(len(z)),u,pos-neg)),receiver


def record(p,y):
    z = trial.clock.center(p.astype(float))
    yy = trial.clock.center(y[:,:,2].astype(float))
    beta = np.clip((z[:,:,2]*yy).sum()/(z[:,:,2]**2).sum(),.4,2.5)
    err = (yy/beta-z[:,:,2]).clip(-4,4)
    stats = []
    for side in (0,1):
        own,s,agree = views(z,side)
        prod = own*err[:,1-side::2,None]
        target = np.column_stack((prod.mean(1),(prod*agree[:,:,None]).mean(1)))
        stats.append((s.T@s/len(s),s.T@target/len(s),(own.transpose(0,2,1)@own).mean(0)/25))
    return tuple(np.mean(v,axis=0) for v in zip(*stats))


def fit(rows):
    gs,bs,rs = [np.stack(v) for v in zip(*rows)]
    out = dict(risk=rs.mean(0))
    for name,ids in (('ordinary',np.arange(7)),('both',np.arange(13))):
        g = gs.mean(0)[np.ix_(ids,ids)]
        b = bs[:,ids].mean(0)
        b *= abs(b)/(abs(b)+bs[:,ids].std(0)/np.sqrt(len(rows))+1.e-8)
        mat = np.zeros((13,6))
        mat[ids] = np.linalg.solve(g+1.1*np.diag(g.diagonal().clip(.0002)),b)
        out[name] = mat
    return out


def signals(p,st):
    z = trial.clock.center(p.astype(float))
    out = np.zeros((*p.shape[:2],len(ARMS)),np.float32)
    risk = st['risk']
    inv = np.linalg.inv(risk+.8*np.diag(risk.diagonal().clip(.001)))
    for side in (0,1):
        own,s,agree = views(z,side)
        def apply(mu):
            w = mu@inv
            size = np.sqrt((w*(w@risk)).sum(1)).clip(1e-8)
            w *= (.55*np.minimum(1.,.18*np.sqrt(risk[2,2])/size))[:,None]
            return (own*w[:,None]).sum(2)
        b = st['both']
        base = apply(s@st['ordinary'][:,:3])
        sender = apply(s[:,7:]@b[7:,:3])
        receiver = apply(s[:,:7]@b[:7,3:])*agree
        interaction = apply(s[:,7:]@b[7:,3:])*agree
        avg = apply(s@b[:,:3])
        bank = (base,sender,receiver,interaction,avg+receiver+interaction,sender-interaction,
                avg+sender,np.roll(interaction,7,axis=1))
        out[:,1-side::2,1:] = np.stack(bank,axis=2)
    return out


def train():
    OUT.mkdir(exist_ok=True)
    rows,folds,daily = [],[],[]
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        refs = history.references(d,raw)
        if d>=10:
            if d%2==0:
                st,group = fit(rows),[]
            ms = []
            for p in refs:
                q = p[:,:,2:]+signals(p,st)
                v = np.concatenate((np.ones_like(p[:,:,:1]),q,y[:,:,2:]),axis=2).reshape(-1,len(ARMS)+2).astype(float)
                ms.append(v.T@v)
            group.append(ms)
            daily.append([prior.gains(m) for m in ms])
            if d%2:
                folds.append([prior.gains(m) for m in np.sum(group,axis=0)])
        rows.append(tuple(np.mean(v,axis=0) for v in zip(*[record(p,y) for p in refs])))
    f,ds = np.asarray(folds),np.asarray(daily)
    avg = f.mean(0)
    keep = [i for i in CANDIDATES if (avg[:,i]>0).all() and (f[-1,:,i]>0).all() and
            (f[:,:,i]>0).sum(0).min()>=2 and (ds[:,:,i]>0).sum(0).min()>=4 and all((avg[:,i]>avg[:,j]).all() for j in CONTROLS)]
    pick = max(keep,key=lambda i:avg[:,i].min()) if keep else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,mean=avg.tolist(),folds=f.tolist(),daily=ds.tolist())))
    print(ARMS[pick],avg.round(6),flush=True)
    if pick:
        np.savez(OUT/'model.npz',pick=pick,**fit(rows))
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        st = dict(f)
    return lambda raw,p:signals(p,st)[:,:,int(st['pick'])]


def replay_with():
    apply = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        q,head = p.copy(),h.copy()
        q[:,:,2] += apply(raw,p)
        head[:,:,2] += apply(raw[:n],h)
        return q,head,n
    return stack.verifier,read


def layer():
    return history.study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        parent.increments.replay.replay(sys.modules[__name__])
