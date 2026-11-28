"""Learn which revisions of a fixed-expiry return and which final settlement errors repeat across dates."""
import sys
import numpy as np
import experiment_dense_market_field_screen_v1 as source

strip,rel,onset,REFERENCE,read_p=source.strip,source.rel,source.onset,source.REFERENCE,source.read_p
OUT=REFERENCE.parent/'dense_terminal_vintage_ladder_v1'
ARMS=('earlier_revisions','per_stage_reliability','terminal_and_stage_agree')
CONTROL=4


def base_scale(end):
    return np.mean([strip.daily_scale(read_p(d),onset.settled.load(d,'y')) for d in range(4,end)],axis=0)


def panels(raw,grid,p,units):
    k,coef,bad=strip.physical()
    for a,b in rel.fields.parent.sessions(grid):
        x=raw[a:b]
        ok=~np.all(x==bad,axis=2)
        cash=np.log1p(k*(x[...,0].astype(float)-.00136))*coef[2]/k
        price=(cash*ok).cumsum(0)
        gaps=np.vstack((np.zeros((1,50),int),(~ok).cumsum(0)))
        q=p[a:b].astype(float)*units
        t=np.arange(len(x))
        cols=[np.ones_like(q[...,0]),*np.moveaxis(q,2,0)]
        for h in (5,20):
            current=q[...,int(h==20)]
            vintages=[]
            for lag in range(5,61-h,5):
                at=(t-lag).clip(0)
                r=h+lag
                if r<=20:
                    old=((20-r)*q[at,:,0]+(r-5)*q[at,:,1])/15
                else:
                    old=((60-r)*q[at,:,1]+(r-20)*q[at,:,2])/40
                valid=(t>=lag)[:,None]&(gaps[t+1]==gaps[at])
                vintages.append(np.where(valid,old-price+price[at],current))
            prev=np.stack(vintages,axis=-1)
            ls=np.sort(prev,axis=-1)
            lower,upper=ls[...,len(vintages)//4],ls[...,(-1-len(vintages)//4)]
            mid=ls[...,len(vintages)//2]
            recent=prev[...,:3].mean(-1)
            early=prev[...,-3:].mean(-1)
            trend=recent-early
            displacement=current-mid
            scale=.03+upper-lower
            outside=(current-upper).clip(0)+(current-lower).clip(max=0)
            position=np.tanh(displacement/scale)
            cols.extend((current-early,current-recent,trend,displacement,outside,position,
                         displacement*np.tanh(trend*displacement/(scale*scale)),
                         position*abs(q[...,2]-q[...,1]),position*abs(trend)))
        z=np.stack(cols,axis=-1)
        yield a,b,z,gaps


def moments(raw,grid,p,y,units):
    _,coef,_=strip.physical()
    # Labels occur only in the fitted right-hand sides. The telescope sums to the
    # original terminal forecast error and contains no approximate interpolated label.
    error=y.astype(float)*(coef[2]/coef)-p.astype(float)*units
    days=[]
    for a,b,z,gaps in panels(raw,grid,p,units):
        t=np.arange(55,b-a-60,4)
        valid=gaps[t+61]==gaps[t-55]
        ee=error[a:b]
        st=np.stack((ee[t,:,1]-ee[t+15,:,0],ee[t+15,:,0],np.zeros_like(ee[t,:,0]),
                     ee[t,:,2]-ee[t+40,:,1],ee[t+40,:,1]-ee[t+55,:,0],ee[t+55,:,0]),axis=-1)
        np.testing.assert_allclose(st[...,3:].sum(-1),ee[t,:,2],atol=1e-12)
        allv=np.concatenate((z[t],st),axis=-1)
        cnt=valid.sum(1)
        m=(allv*valid[...,None]).sum(1)/cnt[:,None].clip(1)
        v=(allv-m[:,None])[valid]
        common=m[cnt>0]
        days.append((common.T@common,len(common),v.T@v,len(v)))
    return np.stack((sum(r[0] for r in days)/sum(r[1] for r in days),
                     sum(r[2] for r in days)/sum(r[3] for r in days)))


def fit(rows):
    n=rows.shape[-1]-6
    out=np.zeros((2,n,3,2))
    for panel in range(2):
        days=rows[:,panel]
        g=days.mean(0)
        v=np.eye(n)[:,CONTROL:]
        v[:CONTROL]=-np.linalg.solve(g[:CONTROL,:CONTROL]+.0001*np.eye(CONTROL),g[:CONTROL,CONTROL:n])
        cov=v.T@g[:n,:n]@v
        sd=np.sqrt(cov.diagonal().clip(1e-8))
        eig,u=np.linalg.eigh(cov/np.outer(sd,sd))
        ax=v@(u/(sd[:,None]*np.sqrt(eig.clip(0)+.8)))
        pay=ax.T@days[:,:n,n:]
        def reliable(x):
            m=x.mean(0)
            return m*m*m/(m*m+x.var(0)/len(x)+1e-8)
        for h in range(2):
            r=pay[:,:,3*h:3*h+3]
            early=reliable(r[:,:,:h+1].sum(2))
            pieces=reliable(r).sum(1)
            whole=reliable(r.sum(2))
            agree=np.where(pieces*whole>0,np.sign(pieces)*np.minimum(abs(pieces),abs(whole)),0.)
            out[panel,:,:,h]=ax@np.stack((early,pieces,agree),axis=1)
    return out


def train(end):
    _,coef,_=strip.physical()
    units=base_scale(end)*coef[2]/coef
    rows=[]
    for d in range(4,end):
        raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
        rows.append(moments(raw,grid,read_p(d),y,units))
    OUT.mkdir(exist_ok=True)
    np.save(OUT/f'gram_{end:02d}.npy',np.stack(rows))
    return np.r_[units,(fit(np.stack(rows))/units[None,None,None,1:]).ravel()]


def signals(raw,grid,p,bank):
    w=bank[3:].reshape(2,-1,3,2)
    out=np.empty((*p.shape[:2],3,2))
    for a,b,z,_ in panels(raw,grid,p,bank[:3]):
        m=z.mean(1,keepdims=True)
        out[a:b]=np.einsum('tnk,kjh->tnjh',z-m,w[1])+np.einsum('tnk,kjh->tnjh',m,w[0])
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
