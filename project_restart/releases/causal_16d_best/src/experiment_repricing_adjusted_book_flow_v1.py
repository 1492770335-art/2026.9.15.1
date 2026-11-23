"""Separate mechanical book changes at price updates from unexpected replenishment within a price spell."""
import json
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART,CACHE,corr,measure,labels
import experiment_eight_date_expert_contrasts_v1 as earlier

OUT=ART/'repricing_adjusted_book_flow_v1'
FIELDS=[10,11,13,14,20,23,26,29,34,36,38,41,42,50,51,52,53,54,55,60,64,72,74]
CUT=np.array([-1.2,-.55,-.16,.16,.55,1.2],dtype='float32')
STEP=np.array([-1.6,-.5,-.04,.04,.5,1.6],dtype='float32')
STOCK=np.arange(50)
F=len(FIELDS)


def rawday(d):
    return np.load(CACHE/f'day_{d:02d}_x.npy',mmap_mode='r')


def frame(raw):
    return np.arcsinh(raw[:,:,FIELDS].astype('float32'))


def fit_book():
    moves=[]
    for d in range(8):
        r=np.abs(rawday(d)[1:,:,0].astype('float32')-.00136)
        moves.append([np.quantile(a[a>.018],.30) for a in r.T])
    tick=np.median(moves,0).clip(.02,6)
    total=np.zeros((F,50,49));counts=np.zeros_like(total)
    for d in range(8):
        raw=rawday(d);z=frame(raw);change=np.diff(z,axis=0)
        move=np.searchsorted(STEP,(raw[1:,:,0].astype('float32')-.00136)/tick)
        for j in range(F):
            bucket=49*STOCK+7*move+np.searchsorted(CUT,z[:-1,:,j])
            counts[j]+=np.bincount(bucket.ravel(),minlength=2450).reshape(50,49)
            total[j]+=np.bincount(bucket.ravel(),weights=change[:,:,j].clip(-4,4).ravel(),minlength=2450).reshape(50,49)
    prior=total.sum(1,keepdims=True)/(counts.sum(1,keepdims=True)+10)
    table=(total+85*prior)/(counts+85)
    return tick,table.astype('float32')


class Day:
    def __init__(self,raw,geom):
        tick,table=geom;n=len(raw)
        z=frame(raw);r=raw[:,:,0].astype('float32')-.00136
        move=np.searchsorted(STEP,r/tick)
        self.shock=np.zeros_like(z)
        for j in range(F):
            bins=7*move[1:]+np.searchsorted(CUT,z[:-1,:,j])
            self.shock[1:,:,j]=np.diff(z[:,:,j],axis=0)-table[j,STOCK,bins]
        self.shock=self.shock.clip(-4,4)
        self.prefix=np.concatenate((np.zeros_like(z[:1],dtype=float),self.shock.cumsum(0,dtype=float)))
        t=np.arange(n)[:,None];event=abs(r)>.018
        self.last=np.maximum.accumulate(np.where(event,t,0),axis=0)
        prev=np.vstack((np.zeros((1,50),int),self.last[:-1]))
        self.before=prev[self.last,STOCK]
        self.move=np.sign(r).astype('float32')
        self.move[~event]=0.
        self.scale=np.sqrt((6+trailing(r*r,96))/(6+np.minimum(t+1,96))).clip(.45,4)

    def design(self,t):
        start=self.last[t];prev=self.before[t]
        age=t[:,None]-start
        here=self.prefix[t+1]-self.prefix[start+1,STOCK]
        prev_flow=self.prefix[start,STOCK]-self.prefix[np.minimum(prev+1,start),STOCK]
        cached=self.shock[start,STOCK]
        rev=self.move[start,STOCK]*self.move[prev,STOCK]
        at=np.arcsinh(here/np.sqrt(1+age[:,:,None]))
        tail=np.arcsinh(prev_flow/np.sqrt((start-prev+1)[:,:,None]))
        decay=np.exp(-age/18)[:,:,None]
        terms=[self.shock[t],at,cached*decay,tail*decay,
               (cached*rev[:,:,None])*decay,at*np.log(self.scale[t])[:,:,None]]
        for w in (10,65):
            size=np.minimum(t+1,w)
            change=(self.prefix[t+1]-self.prefix[t+1-size])/np.sqrt(size[:,None,None])
            terms.append(np.arcsinh(change))
        z=np.concatenate(terms,axis=2).astype('float32')
        m=z.mean(1)
        return np.column_stack((m,m[:,:F]*np.exp(-t[:,None]/360))).astype('float32'),z-m[:,None]


def trailing(x,w):
    c=np.vstack([np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)])
    end=np.arange(1,len(c))
    return c[end]-c[(end-w).clip(0)]


def fit(hist):
    out=[]
    for strength,part in zip((2.8,1.35),zip(*hist)):
        g,b=(np.mean(z,0) for z in zip(*part))
        reg=strength*np.diag(g.diagonal().clip(.00002,None))
        daily=np.stack([np.linalg.solve(a+reg,c) for a,c in part])
        out.append(.68*(.6*np.linalg.solve(g+reg,b)+.4*np.median(daily,0))*abs(np.sign(daily).mean(0)))
    return out


def predict(day,p,w):
    a=np.empty((len(p),1,3),np.float32);b=np.empty_like(p)
    for start in range(0,len(p),111):
        t=np.arange(start,min(start+111,len(p)));m,r=day.design(t)
        a[t,0]=m@w[0];b[t]=r@w[1]
    return a,b


def main():
    OUT.mkdir(exist_ok=True)
    hist=[];report={};daily=[];geom=fit_book()
    for d in range(8,20):
        if d in (10,12,14,16):
            w=fit(hist);rows=[]
            if d==16:
                a=np.stack(daily)
                good=((a>0).sum(0)>=4)&(a.mean(0)>0)&(a[-2:].mean(0)>0)
                report['eligible']=good.tolist()
                if not good.any():break
                w=[v*g for v,g in zip(w,good)]
                np.savez(OUT/'model.npz',tick=geom[0],table=geom[1],market=w[0],relative=w[1])
        raw=rawday(d);p=earlier.baseline(d);day=Day(raw,geom)
        if d>=10:
            a,b=predict(day,p,w);q=(p+a+b).astype('float32')
            if d in (10,16):
                ah,bh=predict(Day(raw[:207],geom),p[:207],w)
                np.testing.assert_allclose(q[:207],p[:207]+ah+bh,atol=3e-6)
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if d<16:
            t=np.arange(d%7,len(p),7);m,r=day.design(t)
            hist.append(earlier.moments(p[t],y[t],(m,r)))
        if d>=10:
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            if d<16:
                old=corr(row[0],row[2])
                daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in (a,b)])
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

