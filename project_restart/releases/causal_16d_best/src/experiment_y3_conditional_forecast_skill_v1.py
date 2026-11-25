"""Regress the realized cross-sectional forecast-error covariance on the opposite basket's observable state."""
import json
import sys
import numpy as np
import torch
import y3_horizon_geometry_training as study
import y3_horizon_transport_references as history

trial,stack,parent = history.trial,history.stack,history.incumbent
OUT = trial.clock.ART/'y3_conditional_forecast_skill_v1'
ARMS = ('unchanged','static skill','variance-conditioned skill','correlation-conditioned skill','joint skill',
        'joint skill with local risk','joint skill with diagonal risk','date-half agreement','incorrect horizon states')
CANDIDATES = (2,3,4,5,6,7)
CONTROLS = (1,8)


def states(p,side):
    source = p[:,side::2]
    z = p[:,1-side::2]
    g = source.transpose(0,2,1)@source/25
    slow = study.ema(g,128)
    fast = .80*study.ema(g,6)+.20*slow
    def features(g):
        sd = np.sqrt(g.diagonal(axis1=1,axis2=2)+.003)
        c = g/(sd[:,:,None]*sd[:,None,:])
        return np.log(sd),c[:,[0,0,1],[1,2,2]]
    a,b = features(fast)
    c,d = features(slow)
    diff = np.column_stack((np.ones(len(p)),a-c,b-d))
    return z,diff,fast


def record(p,y):
    z = trial.clock.center(p.astype(float))
    yy = trial.clock.center(y[:,:,2].astype(float))
    slope = np.clip(np.sum(z[:,:,2]*yy)/np.sum(z[:,:,2]**2),.4,2.5)
    err = (yy/slope-z[:,:,2]).clip(-4,4)
    records = []
    for side in (0,1):
        own,s,g = states(z,side)
        outcome = (own*err[:,1-side::2,None]).mean(1)
        scale = s.T@s/len(s)
        rhs = s.T@outcome/len(s)
        records.append((scale,rhs,g.mean(0)))
    return tuple(np.mean(v,axis=0) for v in zip(*records))


def fit(rows):
    grams,rhs,risks = [np.stack(z) for z in zip(*rows)]
    out = dict(risk=risks.mean(0))
    for name,ix in (('base',[0]),('spread',[0,1,2,3]),('corr',[0,4,5,6]),('all',list(range(7)))):
        def estimate(gg,rr):
            a = gg.mean(0)[np.ix_(ix,ix)]
            b = rr[:,ix].mean(0)
            se = rr[:,ix].std(0)/max(1,np.sqrt(len(gg)))
            b *= abs(b)/(abs(b)+se+1.e-8)
            w = np.zeros((7,3))
            w[ix] = np.linalg.solve(a+.9*np.diag(a.diagonal().clip(.0002)),b)
            return w
        out[name] = estimate(grams,rhs)
        if name=='all':
            a,b = [estimate(grams[i::2],rhs[i::2]) for i in (0,1)]
            out['rep'] = np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
    return out


def signals(p,st):
    z = trial.clock.center(p.astype(float))
    bank = np.zeros((*p.shape[:2],len(ARMS)),np.float32)
    for side in (0,1):
        own,s,g = states(z,side)
        risk = st['risk']
        reg = .8*np.diag(risk.diagonal().clip(.001))
        normal = np.linalg.inv(risk+reg)
        def static(name):
            return (s@st[name])@normal
        full = static('all')
        mu = s@st['all']
        local = np.linalg.solve(.5*(risk+g)+reg,mu[:,:,None])[:,:,0]
        diagonal = mu/(.5*(risk.diagonal()+g.diagonal(axis1=1,axis2=2))+reg.diagonal())
        wrong = (s[:,[0,3,1,2,6,4,5]]@st['all'])@normal
        for j,coef in enumerate((static('base'),static('spread'),static('corr'),full,local,diagonal,static('rep'),wrong),1):
            norm = np.sqrt(np.einsum('ni,ij,nj->n',coef,risk,coef)).clip(1e-8)
            coeff = coef*(.55*np.minimum(1.,.18*np.sqrt(risk[2,2])/norm))[:,None]
            bank[:,1-side::2,j] = (own*coeff[:,None]).sum(2)
    return bank


def stats(p,y,st):
    u = p[:,:,2:]+signals(p,st)
    v = np.concatenate((np.ones_like(p[:,:,:1]),u,y[:,:,2:]),axis=2)
    v = v.reshape(-1,v.shape[-1]).astype(float)
    return v.T@v


def gains(m):
    c = stack.covariance(m)
    ic = c[:-1,-1]/np.sqrt(c.diagonal()[:-1]*c[-1,-1])
    return ic-ic[0]


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
            ms = [stats(p,y,st) for p in refs]
            group.append(ms)
            daily.append([gains(m) for m in ms])
            if d%2:
                folds.append([gains(m) for m in np.sum(group,axis=0)])
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
        q,part = p.copy(),h.copy()
        q[:,:,2] += apply(raw,p)
        part[:,:,2] += apply(raw[:n],h)
        return q,part,n
    return stack.verifier,read


def layer():
    return history.study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        history.previous.incumbent.replay.replay(sys.modules[__name__])
