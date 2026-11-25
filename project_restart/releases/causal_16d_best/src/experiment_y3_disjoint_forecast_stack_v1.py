"""Use chronologically out-of-fold direct interval heads to recalibrate the retained Y3 forecasts."""
import json
import sys

import numpy as np
import torch

import experiment_y3_disjoint_peer_gru_v1 as net
import y3_disjoint_validation as trial

OUT=trial.clock.ART/'y3_disjoint_forecast_stack_v1'
ARMS=['reference calibration','new forecasts only','joint linear calibration','conditional calibration',
      'joint shuffled forecasts','conditional shuffled forecasts','increment over reference calibration']
CACHE=OUT/'historical_heads'
state=net.state


@torch.inference_mode()
def heads(z,weights):
    m=net.PeerNet(z.shape[-1]).cuda().eval()
    m.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
    out=[]
    for t in range(0,len(z),48):
        ids=np.arange(t,t+48).clip(max=len(z)-1)
        x=torch.from_numpy(z[ids]).cuda()
        out.append(m(x)[:min(48,len(z)-t)].cpu().numpy())
    return np.concatenate(out)


def design(p,heads):
    r=trial.curve.settled.historical_geometry()[1][:,1]
    v=trial.clock.center(p)*np.array([r[2]/r[0],r[2]/r[1],1],dtype='float32')
    v[:,:,2]-=v[:,:,1];v[:,:,1]-=v[:,:,0]
    def interaction(h):
        normalize=lambda z:z/(.025+np.sqrt((z*z).mean(1,keepdims=True)))
        match=np.tanh(normalize(h)*normalize(v[:,:,2:]))
        return trial.clock.center(np.concatenate((h,h*match,v[:,:,2:]*match),2))
    wrong=np.roll(heads,17,axis=1)
    return np.concatenate((v,interaction(heads),interaction(wrong)),2)


GROUPS=[[0,1,2],[3,4],[0,1,2,3,4],list(range(9)),[0,1,2,9,10],[0,1,2,9,10,11,12,13,14]]


def moments(x,p,y):
    z=x.reshape(-1,x.shape[-1]).astype(float)
    b=p[:,:,2].astype(float).ravel()
    truth=y[:,:,2].astype(float).ravel()
    b-=b.mean();truth-=truth.mean()
    k=(b@truth)/max(1e-9,b@b)
    error=truth/k-b
    return z.T@z/len(z),z.T@error/len(z)


def calibrate(history):
    gs,bs=[np.stack(v) for v in zip(*history)]
    g=gs.mean(0);b=bs.mean(0)
    noise=bs.std(0)/np.sqrt(len(bs))
    rhs=b*abs(b)/(abs(b)+noise+1e-6)
    weights=[]
    for ids in GROUPS:
        s=g[np.ix_(ids,ids)]
        w=np.zeros(len(g))
        w[ids]=trial.curve.solve(s,rhs[ids],1.).clip(-1.2,1.2)
        weights.append(w)
    weights.append(weights[3]-weights[0])
    return (.4*np.column_stack(weights)).astype('float32')


def changes_with_reference(z,m,p):
    w={k[4:]:v for k,v in m.items() if k.startswith('net/')}
    x=design(p,heads(z,w))
    return list(np.moveaxis(x @ m['calibration'],2,0))


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    def improves(a,b=None,focus=False):
        z,s=f[...,a],d[...,a]
        if b is not None:
            z,s=z-f[...,b],s-d[...,b]
        # Three disjoint two-day folds after two earlier days initialize the calibration.
        return (z[:,:,0]>0).all() and ((s[:,0]>0).sum()>=4) and (focus or (z[:,:,1:].mean(0).min()>0))
    eligible=[a for a,b in [(2,4),(3,5),(6,5)] if improves(a) and improves(a,0,True) and improves(a,b,True)]
    if improves(0):
        eligible.append(0)
    return int(max(eligible,key=lambda i:f[:,0,0,i].mean())) if eligible else None


def run():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    train=[];history=[];folds=[];daily=[]
    for d in range(16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        z=state(raw)
        if d>=8:
            path=CACHE/f'day_{d:02d}.npz'
            if d%2==0:
                if not path.exists() or not (CACHE/f'day_{d+1:02d}.npz').exists():
                    weights,_=net.fit(train)
                if d>=10:
                    coeff=calibrate(history);stats=[[],[]]
            if path.exists():
                with np.load(path) as stored:
                    assert int(stored['train_end'])==d-d%2 and stored['raw_hash'].item()==trial.previous.prior.digest(raw)
                    h=stored['h']
            else:
                h=heads(z,weights)
                np.testing.assert_allclose(h[:367],heads(state(raw[:367]),weights),atol=3e-5,rtol=0)
                np.savez(path,h=h,train_end=d-d%2,raw_hash=trial.previous.prior.digest(raw))
            moments_today=[]
            for j,p in enumerate(trial.refs(d,raw)):
                x=design(p,h)
                if d>=10:
                    delta=list(np.moveaxis(x@coeff,2,0))
                    summaries=[a.sum(0) for a in trial.statistics(p,y,delta)]
                    stats[j].append(summaries)
                    if j==0:
                        daily.append([trial.scoring.audit.pooled(a).tolist() for a in summaries])
                moments_today.append(moments(x,p,y))
            history.append([np.mean(v,0) for v in zip(*moments_today)])
            if d>=10 and d%2:
                folds.append([[trial.scoring.audit.pooled(a).tolist() for a in np.sum(col,axis=0)] for col in stats])
                print('fold',d+1,'full IC changes',folds[-1][0][0],flush=True)
        train.append(net.observe(z,y,d))
    pick=select(folds,daily)
    report=dict(arms=ARMS,selected=pick,folds=folds,daily=daily,calibration_warmup_days=[9,10],assessment_days=list(range(11,17)))
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if pick is not None:
        weights,_=net.fit(train)
        np.savez(OUT/'model.npz',pick=pick,calibration=calibrate(history),**{'net/'+k:v for k,v in weights.items()})
        p=trial.previous
        trial.scoring.outer(sys.modules[__name__],reference=trial.parent,preceding=(p.prior.prior.incumbent,p.prior.incumbent,p.incumbent))


def correction():
    return trial.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
