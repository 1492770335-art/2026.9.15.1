"""Forecast each ten-tick forward return and test curve information independent of existing forecasts."""
import json
import numpy as np
import torch
import experiment_y3_split_view_sector_filter_v1 as incumbent
import experiment_future_path_prototypes_v1 as paths
import experiment_y3_settled_factor_confidence_v1 as settled

clock=incumbent.clock
metric=incumbent.incumbent.base.chain.metric
OUT=clock.ART/'y3_partial_forward_curve_v1'
REFS=incumbent.arrival.sector.incumbent.REFS
EDGES=np.arange(20,61,10)


def target(y,t):
    k,coeff,_=settled.historical_geometry()
    # Compose each adjacent pair of official five-tick labels in physical return units.
    near=(y[t,:,1]-coeff[1,0])/coeff[1,1]
    growth=1+k*near
    blocks=[]
    for start in EDGES[:-1]:
        a=(y[t+start,:,0]-coeff[0,0])/coeff[0,1]
        b=(y[t+start+5,:,0]-coeff[0,0])/coeff[0,1]
        move=a+b+k*a*b
        blocks.append(growth*coeff[2,1]*move)
        growth=growth*(1+k*move)
    path=np.stack(blocks,2)
    path+=(coeff[2,0]-clock.RATIO*coeff[1,0])/4
    actual=y[t,:,2]-clock.RATIO*y[t,:,1]
    gap=actual-path.sum(2)
    path+=gap[:,:,None]/4
    return clock.center(path),float(np.sqrt(np.mean(gap**2)))


def design(raw,t,net,day=None):
    x,v=paths.frozen.encoding(raw,t,net,day,fixed_batch=True)
    x=x.float().cpu().numpy();v=v.cpu().numpy()
    h=x[:,:,:96]
    rel=clock.center(h)
    z=np.concatenate((rel,np.tanh(2*rel),clock.center(x[:,:,96:99]),
                      rel*np.tanh(2*x[:,:,97:98]),rel*np.tanh(x[:,:,98:99]-clock.RATIO*x[:,:,97:98])),2)
    return z.astype('float32'),v


def gram(x,y):
    x=x.reshape(-1,x.shape[-1]);y=y.reshape(len(x),-1)
    return (x.T@x/len(x)).astype(float),(x.T@y/len(x)).astype(float)


def solve(a,b,r):
    return np.linalg.solve(a+r*np.diag(a.diagonal().clip(.00005))+.00001*np.eye(len(a)),b)


def fit(end,net):
    history=[]
    for d in range(end):
        raw=clock.book.rawday(d);t=np.arange(d%8,len(raw)-61,8)
        x,v=design(raw,t,net,d)
        path,_=target(clock.labels(d).reshape(-1,50,3),t)
        history.append(gram(x,(path/v[:,:,None]).clip(-3,3).astype('float32')))
    a,b=[np.mean(s,0) for s in zip(*history)]
    return solve(a,b,.6).astype('float32')


def predict(raw,t,net,w):
    x,v=design(raw,t,net)
    return clock.center((x@w)*v[:,:,None])


def basis(p,bands):
    ref=clock.center(p)
    slopes=np.stack((bands[:,:,:2].sum(2)-bands[:,:,2:].sum(2),
                     bands[:,:,0]-bands[:,:,1]-bands[:,:,2]+bands[:,:,3]),2)
    return np.concatenate((ref,bands.sum(2,keepdims=True),slopes),2)


def calibrate(hist):
    a,b=[np.mean(v,0) for v in zip(*hist)]
    # Remove existing horizon exposure before estimating the incremental shape payoff.
    proj=solve(a[:4,:4],a[:4,4:],.025)
    direction=np.vstack((-proj,np.eye(2)))
    day=np.array([direction.T@y for g,y in hist])
    mean=day.mean(0);se=day.std(0)/np.sqrt(len(day))
    coeff=solve(direction.T@a@direction,mean,.6)
    return (.4*direction@(coeff*abs(mean)/(abs(mean)+se+.00002))).ravel()


def diagnostics(p,bands,path):
    x=clock.center(p).reshape(-1,3).astype(float)
    y=path.reshape(-1,4).astype(float)
    s=bands.reshape(-1,4).astype(float)
    def residual(z):return z-x@np.linalg.lstsq(x,z,rcond=None)[0]
    yr,sr=residual(y),residual(s)
    def cols(a,b):return ((a*b).mean(0)/np.sqrt((a*a).mean(0)*(b*b).mean(0)).clip(1e-10)).tolist()
    return dict(model_ic=cols(s,y),partial_ic=cols(sr,yr),incumbent_far_ic=cols(
        np.broadcast_to((x[:,2]-clock.RATIO*x[:,1])[:,None],y.shape),y))


def run():
    history=[];folds=[];daily=[];per_day={};geo={}
    for end,stop in ((8,12),(12,14),(14,16)):
        net=paths.decoder.net(end)
        w=fit(end,net)
        for begin in range(end,stop,2):
            coef=calibrate(history) if history else None;groups=[[],[]]
            for day in (begin,begin+1):
                raw=clock.book.rawday(day);t=np.arange(day%4,len(raw),4)
                bands=predict(raw,t,net,w)
                if day==begin:
                    sub=t[t<413]
                    np.testing.assert_allclose(predict(raw[:413],sub,net,w),bands[:len(sub)],rtol=0,atol=3e-5)
                y=clock.labels(day).reshape(-1,50,3)
                valid=t<len(raw)-61;truth,gap=target(y,t[valid]);geo[day+1]=gap
                for j,ref in enumerate(REFS):
                    p=ref(day).reshape(y.shape)[t]
                    x=basis(p,bands)
                    if j==0:
                        per_day[day+1]=diagnostics(p[valid],bands[valid],truth)
                        a=clock.center(p[valid,:,2]);b=clock.center(y[t[valid],:,2])
                        scale=((a*b).mean()/(a*a).mean().clip(.0001)).clip(.45,2)
                        history.append(gram(x[valid],(b/scale-a).clip(-5,5)[:,:,None]))
                    if coef is not None:groups[j].append((p,y[t],x@coef))
                if coef is not None:daily.append(metric.gain(groups[0][-1:]))
            if coef is not None:folds.append([metric.gain(g) for g in groups])
            print(begin+1,per_day[begin+1]['partial_ic'],folds[-1] if coef is not None else 'calibration',flush=True)
    keep=bool(all(sum(f[j]['y3']>0 for f in folds)>=2 and np.mean([f[j]['y3'] for f in folds])>0 and
             folds[-1][j]['y3']>0 for j in (0,1)) and sum(d['y3']>0 for d in daily)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(selected=keep,folds=folds,daily=daily,
        segment_diagnostics=per_day,reconstruction_rmse_before_reconciliation=geo,segments=EDGES.tolist()),indent=2))
    if keep:np.savez(OUT/'model.npz',w=fit(16,paths.decoder.net(16)),coef=calibrate(history))
    return keep


def layer():
    parent=incumbent.layer();net=paths.decoder.net(16)
    with np.load(OUT/'model.npz') as s:w,coef=s['w'],s['coef']
    def apply(raw,p):
        q=parent(raw,p)
        for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+399)//400)):
            q[t,:,2]+=basis(q[t],predict(raw,t,net,w))@coef
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,act=ancestor.load(),layer()
    return lambda raw,p:act(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():incumbent.incumbent.base.chain.replay.verify(OUT,layer,incumbent.OUT)
