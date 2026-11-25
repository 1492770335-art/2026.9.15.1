"""Test the fixed same-clock market readout against full-session, shifted-clock and replicated-date controls."""
import argparse
import json
import numpy as np
import experiment_y3_persistent_backbone_exposure_v1 as parent
from y3_marginal_replay_v1 import evaluate

BLOCK=192
OUT=parent.OUT.parent/'y3_session_continuation_ridge_v1'
ARMS=('incumbent','full-session coefficient','shift by half a session','same-clock continuation',
      'same clock above full session','same-clock coefficient replicated on disjoint dates')
CHOSEN=3


def rolling(values, width):
    end = np.arange(1, len(values)+1)
    start = (end-width).clip(0)
    sums = np.concatenate((np.zeros_like(values[:1]), values.cumsum(0, dtype=float)))
    return (sums[end]-sums[start])/(end-start)[:, None]


def design(raw,p):
    r=raw[:,:,0].astype(float)-.00136
    q=np.tanh(.8*np.arcsinh(raw[:,:,[26,38,41,42,54,55,77,78]].astype(np.float32)))
    f=np.column_stack((np.tanh(r).mean(1),np.tanh(3*r).mean(1),(q[:,:,::2]-q[:,:,1::2]).mean(1)))
    return np.column_stack((np.ones(len(raw)),p.mean(1,dtype=float),rolling(p.mean(1,dtype=float),48),
                            rolling(f,6),rolling(f,32)))


def record(raw,p,y):
    a=parent.exposure.bands(p)
    b=parent.exposure.bands(y)
    av=a-a.mean((0,1));bv=b-b.mean((0,1))
    slope=((av*bv).sum((0,1))/(av*av).sum((0,1)).clip(.001)).clip(.45,2.5)
    residual=(b/slope-a).mean(1)
    out=[]
    for lo,hi in parent.base.segments(len(p)):
        x=design(raw[lo:hi],p[lo:hi])
        g=[];c=[]
        for start in range(0,len(x),BLOCK):
            t=np.arange(start,min(len(x)-60,start+6*BLOCK),4)
            z=x[t]
            g.append(z.T@z/max(1,len(z)))
            c.append(z.T@residual[lo+t]/max(1,len(z)))
        out.append((np.stack(g),np.stack(c)))
    return out


def params(history):
    result={}
    for side,records in enumerate(zip(*history)):
        n=min(len(g) for g,c in records)
        gs,cs=[np.stack([v[:n] for v in a]) for a in zip(*records)]
        def solve(g,c):
            penalty=np.eye(g.shape[-1])*2*g.diagonal(axis1=-2,axis2=-1).clip(.002)[...,None,:]
            return np.linalg.solve(g+penalty,c)
        g,c=gs.mean(0),cs.mean(0)
        full=solve(g.mean(0),c.mean(0))
        clock=solve(g,c)
        a,b=[solve(gs[i::2].mean(0),cs[i::2].mean(0)) for i in (0,1)]
        stable=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
        result[f'w{side}']=np.stack([np.broadcast_to(full,clock.shape),np.roll(clock,n//2,axis=0),
                                     clock,clock-full,stable],axis=1)
    return result


def signals(raw,p,model):
    parts=[]
    for j,(lo,hi) in enumerate(parent.base.segments(len(raw))):
        x=design(raw[lo:hi],p[lo:hi])
        w=model[f'w{j}']
        idx=np.minimum(np.arange(len(x))//BLOCK,len(w)-1)
        pred=np.einsum('tc,tkch->tkh',x,w[idx])
        parts.append(.23*(.62*pred[:,:,0]+.38*pred[:,:,1]))
    z=np.concatenate(parts).astype(np.float32)
    return np.broadcast_to(z[:,None],(len(z),50,z.shape[1]))


def run():
    hist=[];obs=[];days=[];fixed=[]
    for d in range(4,16):
        raw,refs,y=parent.history.training_day(d)
        refs=np.array(refs,copy=True)
        today=[parent.exposure.observations(p,y) for p in refs]
        if d>=8:
            if d%2==0:rolling=params(hist);w=parent.exposure.fit(obs)[6]
            if d==8:anchor=params(hist)
            refs[:,:,:,2]+=parent.apply_market(*parent.historical_heads(d),w)
            strict=np.load(parent.ART/'cutoff_reference_panel_v1'/f'panel_{d}.npy')
            pair=[];frozen=[]
            for j,p in enumerate([*refs,strict]):
                z=signals(raw,p,rolling)
                if d==8 and j==0:
                    for n in (157,2519):
                        np.testing.assert_allclose(z[:n],signals(raw[:n],p[:n],rolling),atol=3e-7,rtol=0)
                pair.append(parent.base.records(p,y,z))
                frozen.append(parent.base.records(p,y,signals(raw,p,anchor)))
            days.append(pair);fixed.append(frozen)
        obs.append(tuple(np.mean(v,0) for v in zip(*today)))
        records=[record(raw,p,y) for p in refs]
        hist.append([tuple(np.mean(v,0) for v in zip(*s)) for s in zip(*records)])
    def score(ds):return np.array([parent.base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(ds,axis=0)])
    h=np.stack([score(days[:4]),score(days[4:])])
    f=np.stack([score(fixed[:4]),score(fixed[4:])])
    day_scores=np.stack([score([r]) for r in days])
    # Preselected baseline from the preceding discovery. Inspect the locked day-8 fit separately.
    ok=h[:,:,:,CHOSEN].min()>0 and f[:,-1,0,CHOSEN].min()>0
    OUT.mkdir(exist_ok=True)
    report=dict(arms=ARMS,selected=CHOSEN if ok else 0,windows=[[9,12],[13,16]],
                halves=h.tolist(),frozen_day8=f.tolist(),daily=day_scores.tolist())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('selected',report['selected'],'rolling',h[:,:,0,CHOSEN].round(6).tolist(),'locked',f[:,:,0,CHOSEN].round(6).tolist(),flush=True)
    if ok:np.savez(OUT/'model.npz',**params(hist))


def correction():
    with np.load(OUT/'model.npz') as f:model=dict(f)
    return lambda raw,p:signals(raw,p,model)[:,:,CHOSEN-1]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',nargs='?',choices=('train','screen','verify'),default='train')
    mode=parser.parse_args().mode
    if mode=='train':run()
    else:evaluate(parent,OUT,correction,mode=='screen')
