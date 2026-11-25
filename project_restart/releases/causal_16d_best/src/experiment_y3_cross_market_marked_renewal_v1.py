"""Transfer price-event directions from external markets while pricing ticks 21–60 on the domestic event clock."""
from functools import lru_cache
import hashlib
import json
import sys
import zipfile

import numpy as np
import torch
from scipy.signal import lfilter

import experiment_external_event_prior_v1 as source
import experiment_joint_factor_order_calibration_v1 as replay
import joint_revision_timescale_references as history
import y3_early_score_neighborhoods as early

parent,trial = history.incumbent,history.trial
RATIO = .160943/.272164
OUT = trial.clock.ART/'y3_cross_market_marked_renewal_v1'
M,A,D = 16,31,5
ARMS = ('unchanged','domestic event chain','external marginal signs','shuffled external motifs','FI directional chain',
        'US directional chain','cross-market replicated evidence','FI conditional versus domestic',
        'clock and drift interaction','persistent drift above state-free chain','near-to-far external change')
CANDIDATES = (4,5,6,7,8,9,10)
CONTROLS = (1,2,3)
CUT = .022


def segments(n):
    return [(lo,hi) for lo,hi in ((0,min(n,2401)),(2401,n)) if hi>lo]


def events(x,threshold):
    at = np.flatnonzero(np.abs(x)>threshold)
    if not len(at):
        return None
    sign = np.sign(x[at]).astype(int)
    same = sign==np.r_[sign[0],sign[:-1]]
    t = np.arange(len(at))
    motif = np.zeros(len(at),int)
    for lag in (3,2,1,0):
        motif = motif*2+same[np.maximum(t-lag,0)]
    slow = lfilter([.075],[1,-.925],sign)
    bias = np.digitize(slow*sign,[-.13,.13])
    return at,sign,motif,bias


def delay(age,rate):
    return np.digitize((age+1)*rate,(.3,.7,1.4,2.8))


def mark_counts(x,threshold):
    e = events(x,threshold)
    shape = (M,3,D,2)
    if e is None or len(e[0])<8:
        return np.zeros(shape)
    at,s,m,b = e
    ids = np.arange(4,len(s)-1)
    lag = delay(np.diff(at)[ids]-1,len(at)/len(x))
    k = (((m[ids]*3+b[ids])*D+lag)*2+(s[ids]==s[ids+1]))
    return np.bincount(k,minlength=np.prod(shape)).reshape(shape)


@lru_cache(maxsize=1)
def external():
    archive = source.fi.ARCHIVE
    manifest = json.loads((source.fi.OUT/'external_data.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
    fin,usa = [],[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted({a['file'] for a in manifest['sequences']}):
            with z.open(name) as stream:
                ask = np.fromstring(stream.readline(),sep=' ')
                stream.readline()
                bid = np.fromstring(stream.readline(),sep=' ')
            mid = np.log(.5*(ask+bid))
            for row in (r for r in manifest['sequences'] if r['file']==name):
                p = mid[row['first']:row['first']+row['rows']]
                fin.append(mark_counts(np.diff(p,prepend=p[0]),1.e-9))
    for sym in ('AAPL','AMZN','GOOG','INTC','MSFT'):
        with zipfile.ZipFile(source.OUT/'lobster_samples'/f'{sym}.zip') as z:
            member, = [n for n in z.namelist() if n.endswith('_orderbook_1.csv')]
            with z.open(member) as stream:
                ask,va,bid,vb = np.loadtxt(stream,delimiter=',',unpack=True)
        valid = (ask>=bid)&(bid>0)&(va>0)&(vb>0)&(ask<1.e9)
        for ids in np.split(np.arange(len(ask)),np.flatnonzero(valid[1:]!=valid[:-1])+1):
            if valid[ids[0]] and len(ids)>8:
                p = np.log(.5*(ask[ids]+bid[ids]))
                usa.append(mark_counts(np.diff(p,prepend=p[0]),1.e-9))
    return np.stack((np.sum(fin,0),np.sum(usa,0)))


def raw_returns(raw):
    return raw[:,:,0].astype(float)-.00136


@lru_cache(maxsize=1)
def pace():
    rate = np.mean([(np.abs(raw_returns(trial.clock.book.rawday(d)))>CUT).mean(0) for d in range(4)],0)
    idx = np.argsort(rate)
    groups = np.empty(50,int)
    for i,ids in enumerate(np.array_split(idx,4)):
        groups[ids] = i
    return groups,np.array([np.mean(rate[groups==i]) for i in range(4)])


def grid(x):
    e = events(x,CUT)
    if e is None:
        return np.zeros((4,len(x)),int)
    at,s,m,b = e
    t = np.arange(len(x))
    ix = np.searchsorted(at,t,side='right')-1
    j = ix.clip(0)
    return np.stack((m[j],b[j],np.minimum(A,t-at[j]).clip(0),s[j]*(ix>=4)))


def moment(raw):
    stocks,_ = pace()
    shape = (4,M,3,A+1)
    count = np.zeros((*shape,5))
    for lo,hi in segments(len(raw)):
        ret = raw_returns(raw[lo:hi])
        for s,r in enumerate(ret.T):
            m,b,a,sign = grid(r)
            valid = sign[:-1]!=0
            k = ((m[:-1]*3+b[:-1])*(A+1)+a[:-1])[valid]
            signed = (r[1:]*sign[:-1])[valid]
            yes,no = signed>CUT,signed<-CUT
            for i,v in enumerate((np.ones(len(k)),yes,no,signed.clip(0,6)*yes,(-signed).clip(0,6)*no)):
                count[stocks[s],:,:,:,i] += np.bincount(k,weights=v,minlength=M*3*(A+1)).reshape(M,3,A+1)
    return count


def table(history_rows):
    count = np.sum(history_rows,axis=0)
    stocks,rates = pace()
    opportunity,yes,no,same_size,rev_size = np.moveaxis(count,4,0)
    events = yes+no
    eh = events.sum((1,2),keepdims=True)
    oh = opportunity.sum((1,2),keepdims=True)
    hazard = (events+60*(eh+6*rates[:,None,None,None])/(oh+6))/(opportunity+60)
    sm = ((same_size.sum((1,2),keepdims=True)+1)/(yes.sum((1,2),keepdims=True)+1))
    rm = ((rev_size.sum((1,2),keepdims=True)+1)/(no.sum((1,2),keepdims=True)+1))
    mags = ((same_size+25*sm)/(yes+25),(rev_size+25*rm)/(no+25))
    dc = np.zeros((M,3,D,2))
    ag = np.arange(A+1)
    bins = np.stack([delay(ag,r) for r in rates])
    for i in range(4):
        for j in range(D):
            take = bins[i]==j
            dc[:,:,j,0] += no[i][:,:,take].sum(-1)
            dc[:,:,j,1] += yes[i][:,:,take].sum(-1)
    def prob(c):
        cond = (c.sum(2,keepdims=True)[:,:,:,1]+4)/(c.sum((2,3),keepdims=False)[:,:,None]+8)
        return (c[:,:,:,1]+30*cond)/(c.sum(3)+30)
    home = prob(dc)
    fi,us = [prob(v) for v in external()]
    delta = np.where((fi-home)*(us-home)>0,np.sign(fi-home)*np.minimum(abs(fi-home),abs(us-home)),0.)
    stable = home+.5*delta
    neutral = np.broadcast_to(fi.mean((0,1)),fi.shape)
    kernels = (home,.6*home+.4*neutral,.6*home+.4*np.roll(fi,5,axis=0),.6*home+.4*fi,.6*home+.4*us,
               stable, np.broadcast_to(stable.mean(1,keepdims=True),stable.shape))
    m = np.arange(M)[:,None]
    b = np.arange(3)[None,:]
    vals = []
    for kernel in kernels:
        c = np.stack([kernel[:,:,ids] for ids in bins])
        prev = np.zeros_like(hazard)
        for k in range(1,61):
            same = prev[:,(2*m+1)%M,b,0][:,:,:,None]
            reverse = prev[:,(2*m)%M,2-b,0][:,:,:,None]
            prev = ((1-hazard)*prev[:,:,:,np.minimum(ag+1,A)]+hazard*
                    (c*(mags[0]+same)-(1-c)*(mags[1]+reverse)))
            if k==20:
                near = prev.copy()
        vals.append((prev-near,near))
    return np.stack(vals).astype('float32')


def signals(raw,w,stocks):
    out = np.zeros((*raw.shape[:2],len(ARMS)-1),np.float32)
    for lo,hi in segments(len(raw)):
        ret = raw_returns(raw[lo:hi])
        for s,r in enumerate(ret.T):
            m,b,a,sign = grid(r)
            z = w[:,:,stocks[s],m,b,a]
            home,marg,wrong,fi,us,shared,no_bias = z[:,0]
            near = z[:,1]
            out[lo:hi,s] = np.stack((home,marg,wrong,fi,us,shared,fi-home,shared-no_bias,
                                    (fi-home)-(no_bias-home),shared-.5*near[5]),axis=1)*sign[:,None]
    return out-out.mean(1,keepdims=True)


def records(p,y,z):
    tail = p[:,:,2]-RATIO*p[:,:,1]
    far = y[:,:,2]-RATIO*y[:,:,1]
    v = np.concatenate((np.ones_like(p[:,:,:1]),p[:,:,2:],tail[:,:,None],z,y[:,:,2:],far[:,:,None]),2)
    v = v.reshape(-1,v.shape[2]).astype(float)
    return v.T@v


def covariance(m):
    return (m[1:,1:]-np.outer(m[0,1:],m[0,1:])/m[0,0])/m[0,0]


def calibrate(rows):
    cs = np.stack([np.mean([covariance(m) for m in d],0) for d in rows])
    out = []
    for k in range(2):
        cov = cs[:,2:-2,k]
        beta = (cs[:,k,k-2]/cs[:,k,k]).clip(.25,3)
        grad = cs[:,2:-2,k-2]/beta[:,None]-cov
        v = cs.diagonal(axis1=1,axis2=2)[:,2:-2]-cov**2/cs[:,k:k+1,k]
        mu = grad.mean(0)
        mu *= abs(mu)/(abs(mu)+grad.std(0)/np.sqrt(len(rows))+1.e-8)
        scale = v.mean(0).clip(1.e-6)
        out.append(.5*mu/scale)
    full,far = out
    return np.where(full*far>0,np.sign(full)*np.minimum(abs(full),abs(far)),0.)


def score(m,w):
    c = covariance(m)
    results = []
    for k in range(2):
        cov = c[2:-2,k]
        vv = c[k,k]+2*w*cov+w*w*c.diagonal()[2:-2]
        r = (c[k,k-2]+w*c[2:-2,k-2])/np.sqrt(vv*c[k-2,k-2])
        r0 = c[k,k-2]/np.sqrt(c[k,k]*c[k-2,k-2])
        results.append(np.r_[0.,r-r0])
    return results


def screen(rows):
    folds,daily = [],[]
    for n in (6,8,10):
        w = calibrate(rows[:n])
        pair = rows[n:n+2]
        folds.append([score(m,w) for m in np.sum(pair,axis=0)])
        daily.extend([[score(m,w) for m in day] for day in pair])
    f,d = np.asarray(folds),np.asarray(daily)
    avg = f.mean(0)
    good = [i for i in CANDIDATES if avg[:,: ,i].min()>0 and f[-1,:,:,i].min()>0 and
            (f[:,:,:,i]>0).sum(0).min()>=2 and (d[:,:,:,i]>0).sum(0).min()>=4 and
            all((avg[:,0,i]>avg[:,0,j]).all() for j in CONTROLS)]
    pick = max(good,key=lambda i:avg[:,0,i].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,metrics=['Y3','21-60'],
                                                        mean=avg.tolist(),folds=f.tolist(),daily=d.tolist()),indent=2))
    print(ARMS[pick],avg.round(6),flush=True)
    return pick,calibrate(rows)


def historical(d,raw):
    if d>=8:
        return history.references(d,raw)
    return [early.load(d,raw)]*2


def train():
    OUT.mkdir(exist_ok=True)
    hist,rows = [],[]
    stocks,_ = pace()
    for d in range(16):
        raw = trial.clock.book.rawday(d)
        if d>=4:
            if d%2==0:
                w = table(hist)
            z = signals(raw,w,stocks)
            y = trial.clock.labels(d).reshape(-1,50,3)
            rows.append([records(p,y,z) for p in historical(d,raw)])
        hist.append(moment(raw))
    pick,coefs = screen(rows)
    if pick:
        np.savez(OUT/'model.npz',w=table(hist),stocks=stocks,pick=pick-1,scale=coefs[pick-1])
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        w,stocks,pick,scale = f['w'],f['stocks'],int(f['pick']),float(f['scale'])
    return lambda raw,p:(signals(raw,w,stocks)[:,:,pick]*scale).astype('float32')


def layer():
    earlier,adjust = parent.layer(),correction()
    def predict(raw,base):
        p = earlier(raw,base)
        q = p.copy()
        q[:,:,2] += adjust(raw,p)
        return q
    return predict


def replay_with():
    adjust = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        q,head = p.copy(),h.copy()
        q[:,:,2] += adjust(raw,p)
        head[:,:,2] += adjust(raw[:n],h)
        return q,head,n
    return history.stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay.replay(sys.modules[__name__])
