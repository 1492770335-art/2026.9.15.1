"""Decompose short book-transition rewards into endpoint, oriented cycle and symmetric edge rewards."""
import sys
import numpy as np

import experiment_dense_vintage_crossing_transfer_v1 as ref

strip,rel,onset,REFERENCE,read_p=ref.strip,ref.rel,ref.onset,ref.REFERENCE,ref.read_p
OUT=REFERENCE.parent/'dense_transition_cash_decomposition_v1'
ARMS=('endpoint_change','oriented_cycle_cash','symmetric_transition_cash')
BID=(41,50,54,77)
ASK=(42,51,55,78)
S=15
J=len(BID)


def book(raw):
    bid=np.arcsinh(raw[:,:,BID].astype(float))
    ask=np.arcsinh(raw[:,:,ASK].astype(float))
    return bid-ask,(bid+ask)*.5


def cuts():
    sampled=[book(onset.settled.load(d,'x')[d::9]) for d in range(4)]
    pressure,size=[np.concatenate(v).reshape(-1,J) for v in zip(*sampled)]
    return np.r_[np.quantile(pressure,[.15,.35,.65,.85],axis=0).ravel(),
                 np.quantile(size,[.30,.70],axis=0).ravel()]


def states(raw,rule):
    pressure,size=book(raw)
    ps,vs=rule[:4*J].reshape(4,J),rule[4*J:].reshape(2,J)
    return np.stack([np.searchsorted(ps[:,j],pressure[...,j],side='right')*3+
                     np.searchsorted(vs[:,j],size[...,j],side='right') for j in range(J)],axis=-1)


def observations(raw,grid,rule):
    k,c,bad=strip.physical()
    for a,b in rel.fields.parent.sessions(grid):
        x=raw[a:b]
        ok=~np.all(x==bad,axis=2)
        inc=np.log1p(k*(x[...,0].astype(float)-.00136))*c[2]/k
        inc=(inc-(inc*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1))*ok
        yield a,b,states(x,rule),inc.cumsum(0),np.vstack((np.zeros((1,50),int),(~ok).cumsum(0)))


def estimate(rule,days):
    counts=np.zeros((J,S,S))
    paid=np.zeros_like(counts)
    for d in days:
        raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
        for a,b,s,price,gap in observations(raw,grid,rule):
            t=np.arange(d%3,b-a-5,3)
            valid=gap[t+6]==gap[t]
            ret=(price[t+5]-price[t])[valid]
            current,nxt=s[t][valid],s[t+5][valid]
            for j in range(J):
                ij=current[:,j]*S+nxt[:,j]
                counts[j]+=np.bincount(ij,minlength=S*S).reshape(S,S)
                paid[j]+=np.bincount(ij,weights=ret,minlength=S*S).reshape(S,S)
    return counts,paid


def laws(dynamics,rewards,return_edges=False):
    values=[]
    for count,n,gross in zip(dynamics[0],*rewards):
        transition=(count+.3)/(count.sum(1,keepdims=True)+.3*S)
        # Symmetric conductances give an undirected least-squares projection; the
        # potential differences telescope exactly on every realized path.
        r=gross/(n+18)
        odd=.5*(r-r.T)
        even=.5*(r+r.T)
        w=2*n*n.T/(n+n.T+1)
        L=np.diag(w.sum(1))-w
        v=np.linalg.lstsq(L,-(w*odd).sum(1),rcond=1e-9)[0]
        potential=v[None,:]-v[:,None]
        cycle=odd-potential
        np.testing.assert_allclose(potential+cycle+even,r,atol=1e-12)
        np.testing.assert_allclose((w*cycle).sum(1),0,atol=1e-6)
        edge=np.stack((potential,cycle,even),-1)
        if return_edges:
            values.append(edge)
            continue
        one=(transition[...,None]*edge).sum(1)
        ys=[]
        for h in (4,12):
            # Use only the 2nd--4th or 5th--12th five-tick payoffs.
            first=1 if h==4 else 4
            val=sum(np.linalg.matrix_power(transition,j)@one for j in range(first,h))
            np.testing.assert_allclose(val[:,0],np.linalg.matrix_power(transition,h)@v-np.linalg.matrix_power(transition,first)@v,atol=1e-10)
            ys.append(val)
        values.append(np.stack(ys,-1))
    return np.stack(values)


def geometry():
    rule=cuts()
    left,right=[estimate(rule,days) for days in ((0,2),(1,3))]
    return rule,.5*(laws(left,right)+laws(right,left))


def features(raw,grid,p,rule,values):
    for a,b in rel.fields.parent.sessions(grid):
        state=states(raw[a:b],rule)
        table=np.stack([values[j][state[...,j]] for j in range(J)],axis=2)
        yield a,b,strip.center(p[a:b].astype(float)),strip.center(table)


def collect():
    OUT.mkdir(exist_ok=True)
    rule,values=geometry()
    np.savez(OUT/'geometry.npz',rule=rule,values=values)
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
        p=read_p(d)
        mats=np.zeros((3,2,3+J+1,3+J+1))
        n=0
        for a,b,q,z in features(raw,grid,p,rule,values):
            t=np.arange(d%4,b-a-60,4)
            truth=strip.center(y[a+t].astype(float))
            for j in range(3):
                for h in range(2):
                    row=np.concatenate((q[t],z[t,:,:,j,h],truth[...,h+1:h+2]),axis=-1).reshape(-1,J+4)
                    mats[j,h]+=row.T@row
            n+=len(t)*50
        np.save(OUT/f'gram_{d:02d}.npy',mats/n)


def train(end):
    gs=np.stack([np.load(OUT/f'gram_{d:02d}.npy') for d in range(4,end)])
    with np.load(OUT/'geometry.npz') as f:
        rule=f['rule'];values=f['values']
    bank=[]
    for j in range(3):
        for h in range(2):
            rows=gs[:,j,h];g=rows.mean(0)
            ax=np.eye(J+3)[:,3:]
            ax[:3]=-np.linalg.solve(g[:3,:3]+.01*np.diag(g.diagonal()[:3]),g[:3,3:-1])
            cov=ax.T@g[:-1,:-1]@ax
            b=h+1
            beta=rows[:,b,-1]/rows[:,b,b].clip(1e-8)
            rhs=(rows[:,:-1,-1]/beta[:,None].clip(.12)-rows[:,:-1,b])@ax
            avg=rhs.mean(0)
            stable=avg**3/(avg*avg+rhs.var(0)/len(rows)+1e-11)
            bank.append(ax@np.linalg.solve(cov+.7*np.diag(cov.diagonal().clip(1e-9))+1e-10*np.eye(J),stable))
    return np.r_[rule,values.ravel(),np.stack(bank).ravel()]


def signals(raw,grid,p,bank):
    cut=6*J
    size=J*S*3*2
    values=bank[cut:cut+size].reshape(J,S,3,2)
    weights=bank[cut+size:].reshape(3,2,J+3)
    out=np.empty((*p.shape[:2],3,2))
    for a,b,q,z in features(raw,grid,p,bank[:cut],values):
        for j in range(3):
            for h in range(2):
                w=weights[j,h]
                out[a:b,:,j,h]=q@w[:3]+z[:,:,:,j,h]@w[3:]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    collect()
    strip.evaluation.rolling(sys.modules[__name__])
