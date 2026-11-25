"""Transfer stock-specific response slopes using prior-date order-book behavior to select peers."""
import json
import numpy as np
from y3_portfolio_walkforward import historical_panels, parent, clock, scores

OUT=parent.ART/'y3_peer_response_fingerprints_v1'
ARMS=('incumbent','reassigned donors','own stock','similar stocks','similar above reassigned',
      'replicated neighbor response','agreement of self and peers','opposite end of distance ranking')


def fingerprint(raw):
    r=np.tanh(raw[:,:,0].astype(float)-.00136)
    q=np.tanh(.8*np.arcsinh(raw[:,:,[26,38,41,42,54,55,77,78]].astype(float)))
    flows=q[:,:,::2]-q[:,:,1::2]
    def corr(a,b):
        a=a-a.mean(0);b=b-b.mean(0)
        return (a*b).mean(0)/np.sqrt((a*a).mean(0)*(b*b).mean(0)+1e-7)
    fields=[np.mean(abs(r),0),np.mean(abs(r)>.05,0),corr(r[1:],r[:-1])]
    for f in flows.transpose(2,0,1):
        fields.extend((abs(f).mean(0),corr(f[4:],f[:-4]),corr(f[:-1],r[1:])))
    value=np.column_stack(fields)
    return (value-value.mean(0))/(value.std(0)+.001)


def features(raw,p):
    market=clock.design(raw,p)
    return market[:,[1,2,3,4,5,6,7,9,10,11,12,13,15,16]]


def observed(raw,p,y,d):
    rows=[]
    for lo,hi in parent.base.segments(len(raw)):
        t=np.arange(d%7,hi-lo-60,7)
        z=features(raw[lo:hi],p[lo:hi])[t]
        a,b=parent.exposure.bands(p[t+lo]),parent.exposure.bands(y[t+lo])
        a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
        unit=((a*b).sum((0,1))/(a*a).sum((0,1)).clip(.001)).clip(.45,2.5)
        err=(b/unit-a)@np.array([.62,.38])
        rows.append((z.T@z/len(z),z.T@err/len(z)))
    return rows


def neighbors(profiles, reverse=False):
    m=np.mean(profiles,0)
    dist=np.mean((m[:,None]-m[None,:])**2,2)
    if reverse:dist=-dist
    np.fill_diagonal(dist,np.inf)
    nearest=dist.argsort(axis=1)[:,:10]
    rankweights=np.linspace(1,.3,10)
    weights=np.zeros((50,50))
    np.put_along_axis(weights,nearest,np.broadcast_to(rankweights/rankweights.sum(),nearest.shape),axis=1)
    return weights


def fit(history,profiles):
    donor=neighbors(profiles);far=neighbors(profiles,True)
    result={}
    for side,records in enumerate(zip(*history)):
        gs,bs=[np.stack(v) for v in zip(*records)]
        def solve(g,b):
            gram=g.mean(0)
            rhs=b.mean(0)
            rhs*=abs(rhs)/(abs(rhs)+b.std(0)/np.sqrt(len(b))+.0002)
            return np.linalg.solve(gram+1.8*np.diag(gram.diagonal().clip(.0005)),rhs)
        own=solve(gs,bs)
        near=own@donor.T
        wrong=own@np.roll(donor,13,axis=1).T
        rep=[solve(gs[i::2],bs[i::2])@neighbors(profiles[i::2]).T for i in (0,1)]
        def agree(a,b):return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
        result[f'w{side}']=np.stack((wrong,own,near,near-wrong,agree(*rep),agree(near,own),own@far.T),axis=2)
    return result


def signals(raw,p,state):
    out=[]
    for side,(lo,hi) in enumerate(parent.base.segments(len(raw))):
        x=features(raw[lo:hi],p[lo:hi])
        z=np.einsum('tc,csk->tsk',x,state[f'w{side}'])
        z-=z.mean(1,keepdims=True)
        out.append(.30*z)
    return np.concatenate(out).astype('float32')


def run():
    history=[];profiles=[];days=[];fixed=[]
    for d,raw,ps,y in historical_panels():
        if d>=8:
            if d%2==0:model=fit(history,profiles)
            if d==8:anchor=model
            rows=[];locked=[]
            for i,p in enumerate(ps):
                z=signals(raw,p,model)
                if d==8 and i==0:
                    np.testing.assert_allclose(z[:2471],signals(raw[:2471],p[:2471],model),atol=3e-7,rtol=0)
                rows.append(parent.base.records(p,y,z))
                locked.append(parent.base.records(p,y,signals(raw,p,anchor)))
            days.append(rows);fixed.append(locked)
        vals=[observed(raw,p,y,d) for p in ps[:2]]
        history.append([tuple(np.mean(v,0) for v in zip(*s)) for s in zip(*vals)])
        profiles.append(fingerprint(raw))
    def halves(data):return np.stack([scores(v,len(ARMS)) for v in (data[:4],data[4:])])
    h=halves(days);f=halves(fixed)
    daily=np.stack([scores([r],len(ARMS)) for r in days])
    pick=max((3,4,5,6),key=lambda i:h[0,:,:,i].min())
    ok=(h[:,:,:,pick].min()>0 and (h[:,:,0,pick]>h[:,:,0,1]).all()
        and f[:,-1,0,pick].min()>0 and (daily[:,:,0,pick]>0).sum(0).min()>=5)
    OUT.mkdir(exist_ok=True)
    result=dict(arms=ARMS,windows=[[9,12],[13,16]],first_window_choice=pick,selected=pick if ok else 0,
                halves=h.tolist(),day8_locked=f.tolist(),daily=daily.tolist())
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    if ok:np.savez(OUT/'model.npz',pick=pick,**fit(history,profiles))
    print('selected',result['selected'],'strict Y3',h[:,-1,0].round(6).tolist(),flush=True)

def correction():
    with np.load(OUT/'model.npz') as f:state=dict(f)
    return lambda raw,p:signals(raw,p,state)[:,:,int(state['pick'])-1]


if __name__=='__main__':
    import argparse
    from y3_portfolio_walkforward import replay
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),nargs='?',default='train')
    parser.add_argument('--upstream-dir')
    args=parser.parse_args()
    if args.mode=='train':run()
    else:
        if args.mode=='screen' and not args.upstream_dir:parser.error('supply --upstream-dir for cached screening')
        replay(OUT,correction,args.upstream_dir if args.mode=='screen' else None)
