"""Forecast from the order in which signed book pressure and unsigned activity change."""
import argparse
import json
import numpy as np

from y3_settled_walkforward_panels import panels, parent, attention
from y3_portfolio_walkforward import scores, replay

OUT = parent.ART/'y3_ordered_book_area_v1'
ARMS = ('incumbent','endpoints and products','own ordered area','slow ordered area','market ordered area',
        'stock and market area','replicated across dates','reassigned stock')
WINDOWS = (12, 48, 144)
PAIR_COUNT = 6


def signature(raw):
    # Pair odd (directional) and even (activity) coordinates. Reversing the
    # path negates its signed area and leaves the product of increments intact.
    q = np.tanh(.8*np.arcsinh(raw[:,:,[26,38,41,42,54,55]].astype('float32')))
    flow = q[:,:,::2]-q[:,:,1::2]
    size = (q[:,:,::2]+q[:,:,1::2])*.5
    ret = np.tanh(raw[:,:,0].astype('float32')-.00136)
    a = np.stack((flow[:,:,0], flow[:,:,1], flow[:,:,2], ret, flow[:,:,1], flow[:,:,0]),2)
    b = np.stack((size[:,:,0], size[:,:,1], size[:,:,2], abs(flow[:,:,0]),
                  abs(ret), abs(flow[:,:,2])),2)
    n = len(raw)
    t = np.arange(n)
    step_area = a[:-1]*b[1:]-b[:-1]*a[1:]
    summed = np.concatenate((np.zeros_like(a[:1],dtype=float), step_area.cumsum(0,dtype=float)))
    endpoint, area = [], []
    for h in WINDOWS:
        start = np.maximum(0,t-h)
        da,db = a-a[start], b-b[start]
        # Close each open polyline by a straight segment back to the start.
        twice = summed[t]-summed[start]+a*b[start]-b*a[start]
        elapsed=np.maximum(t-start,1)[:,None,None]
        endpoint.extend((da, da*db))
        area.append(np.tanh(twice/np.sqrt(elapsed)))
    return np.concatenate((*endpoint,*area),2).astype('float32')


def gram(raw, ps, y, d):
    stock,market=[],[]
    for lo,hi in parent.base.segments(len(raw)):
        t = np.arange(d%7,hi-lo-60,7)
        x = signature(raw[lo:hi])[t]
        p = np.mean(ps[:2,lo+t],0)
        v = np.concatenate((x,p,parent.exposure.bands(y[lo+t])),2)
        m = v.mean(1)
        s = (v-m[:,None]).reshape(-1,v.shape[-1]).astype(float)
        m = m.astype(float)
        stock.append(s.T@s/len(s));market.append(m.T@m/len(m))
    return np.mean(stock,0),np.mean(market,0)


def fit_moments(mats):
    mats=np.stack(mats)
    n=mats.shape[-1]-5
    edge=2*len(WINDOWS)*PAIR_COUNT
    # Remove responses explained by the incumbent and endpoint/product controls.
    def solve(rows):
        g=rows.mean(0)
        def reg(v): return np.diag(v.diagonal().clip(.00002))
        ref=g[n:n+3,n:n+3]
        baseline=np.linalg.solve(ref+.015*reg(ref),g[n:n+3,-2:])
        unit=(g[n+2,-2]/max(g[n+2,n+2],.0001)).clip(.45,2.5)
        rhs=(rows[:,:n,-2:]-rows[:,:n,n:n+3]@baseline)@np.array([.65,.35])/unit
        mean=rhs.mean(0)
        mean*=abs(mean)/(abs(mean)+rhs.std(0)/np.sqrt(len(rows))+.00002)
        first=g[:edge,:edge]
        inv=np.linalg.solve(first+reg(first),g[:edge,edge:n])
        bank=[]
        w=np.zeros(n);w[:edge]=np.linalg.solve(first+reg(first),mean[:edge])
        bank.append(w)
        for ids in (np.arange(n-edge),np.arange(PAIR_COUNT,n-edge)):
            idx=edge+ids
            coeff=inv[:,ids]
            cov=g[np.ix_(idx,idx)]-g[idx,:edge]@coeff-coeff.T@g[:edge,idx]+coeff.T@first@coeff
            gain=mean[idx]-coeff.T@mean[:edge]
            v=np.linalg.solve(cov+1.5*reg(cov),gain)
            w=np.zeros(n);w[idx]=v;w[:edge]=-coeff@v
            bank.append(w)
        return np.stack(bank,axis=1)
    w=solve(mats)
    a,b=solve(mats[::2])[:,1],solve(mats[1::2])[:,1]
    stable=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    return np.column_stack((w,stable)).astype('float32')


def fit(hist):
    return tuple(fit_moments(v) for v in zip(*hist))


def signals(raw,state):
    sw,mw=state
    parts=[]
    for lo,hi in parent.base.segments(len(raw)):
        x=signature(raw[lo:hi])
        m=x.mean(1,keepdims=True)
        st=(x-m)@sw
        mk=m@mw
        broad=np.broadcast_to(mk[:,:,1],st.shape[:2])
        parts.append(.32*np.stack((st[:,:,0]+mk[:,:,0],st[:,:,1],st[:,:,2],broad,
                                   st[:,:,1]+broad,st[:,:,3]+mk[:,:,3],np.roll(st[:,:,1],17,axis=1)),2))
    return np.concatenate(parts)


def train():
    hist,days,locked=[],[],[]
    for d,raw,ps,y in panels(refresh=True):
        if d>=8:
            if d%2==0:state=fit(hist)
            if d==8:anchor=state
            z=signals(raw,state);v=signals(raw,anchor)
            if d==8:
                for cut in (7,127,2461):
                    np.testing.assert_allclose(z[:cut],signals(raw[:cut],state),rtol=0,atol=2e-6)
            days.append([parent.base.records(p,y,z) for p in ps])
            locked.append([parent.base.records(p,y,v) for p in ps])
        hist.append(gram(raw,ps,y,d))
        print('trained through',d+1,flush=True)
    def halves(rows):return np.stack([scores(v,len(ARMS)) for v in (rows[:4],rows[4:])])
    h,f=halves(days),halves(locked)
    daily=np.stack([scores([r],len(ARMS)) for r in days])
    j=max(range(2,7),key=lambda i:h[0,:,0,i].min())
    keep=(h[:,-1,0,j].min()>0 and h[:,:,0,j].mean(1).min()>0 and f[:,-1,0,j].min()>0 and
          (daily[:,-1,0,j]>0).sum()>=5 and (h[:,:,0,j]-h[:,:,0,1]).mean()>0)
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,windows=[[9,12],[13,16]],choice=j,
        selected=j if keep else 0,halves=h.tolist(),frozen=f.tolist(),daily=daily.tolist()),indent=2))
    if keep:
        sw,mw=fit(hist)
        np.savez(OUT/'model.npz',sw=sw,mw=mw,pick=j)
    print('choice',j,'selected',j if keep else 0,'strict Y3',h[:,-1,0].round(6).tolist(),flush=True)


def correction():
    prior=attention.correction()
    with np.load(OUT/'model.npz') as f:state=(f['sw'],f['mw']);j=int(f['pick'])
    return lambda raw,p: prior(raw,p)+signals(raw,state)[:,:,j-1]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),default='train',nargs='?')
    parser.add_argument('--upstream-dir')
    args=parser.parse_args()
    if args.mode=='train':train()
    elif args.mode=='screen' and not args.upstream_dir:parser.error('missing --upstream-dir')
    else:replay(OUT,correction,args.upstream_dir)
