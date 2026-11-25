"""Fit a joint readout of chronological Y3 forecast revisions without changing the input to any expert."""
import importlib
import json
import os
import sys
import numpy as np
import torch
import y3_clock_and_depth_references as history
import y3_filter_validation as early
import y3_tail_specialist_eval as verifier
from y3_expanding_path_references import read_verified

trial=history.trial
parent=history.incumbent
OUT=trial.clock.ART/'y3_joint_stage_readout_v1'
STAGES=tuple('y3_'+s+'_v1' for s in (
    'interval_rank_heads','state_transition_sources','honest_pair_distribution','risk_rank_partition','pair_crossing_premia',
    'two_sided_path_encoder','path_price_context','expanding_path_complement','scale_conditioned_path',
    'path_magnitude_consensus','expected_price_paths','unfulfilled_price_moves','market_reversal_flow',
    'overshoot_basket_agreement','monotone_horizon_projection','quiet_forecast_revisions','event_forecast_response_shape',
    'same_deadline_event_curve','same_quote_memory','quote_event_tokens','inferred_print_impacts','trade_spell_survival',
    'peer_depth_event_causes','peer_terminal_price_news','joint_order_price_rollout','persistent_order_modes',
    'receiver_price_portfolios','realized_recipient_exposures','crossed_level_book','impact_mass_forecast','clock_and_crossing_depth'))
FAMILIES=(5,12,18,22,28,31)
ARMS=('amplify last two','uniform half','learned common','families','individual experts','day-shrunk experts',
      'common versus stock-relative by family','common versus stock-relative by expert','phase-independent projected ridge')
N=len(STAGES)


def module(name):
    return importlib.import_module('experiment_'+name)


def panels(day,raw):
    ref=early.refs(day,raw)
    z=np.empty((2,len(raw),50,N),dtype=np.float32)
    loaders=('y3_terminal_validation.refs','y3_disjoint_validation.refs','y3_pair_incumbent_validation.refs',
             'y3_risk_rank_validation.references','y3_crossing_validation.references',
             'y3_path_orientation_validation.references','y3_path_context_validation.references')
    for j,name in enumerate(STAGES):
        if j<len(loaders):
            source,func=loaders[j].split('.')
            nxt=getattr(importlib.import_module(source),func)(day,raw)
        else:
            nxt=read_verified(trial.clock.ART/name/'historical_references',day,raw)
        diff=nxt[:,:,:,2]-ref[:,:,:,2]
        z[:,:,:,j]=diff
        ref=nxt
    return ref,z


def moments(day):
    raw=trial.clock.book.rawday(day)
    ref,diff=panels(day,raw)
    y=trial.clock.labels(day).reshape(-1,50,3)[:,:,2]
    z=np.empty((*diff.shape[:3],2*N+3),dtype=np.float32)
    z[:,:,:,0]=1
    z[:,:,:,1]=ref[:,:,:,2]
    mean=diff.mean(2,keepdims=True)
    z[:,:,:,2:2+N]=diff-mean
    z[:,:,:,2+N:-1]=mean
    z[:,:,:,-1]=y
    grams=[]
    for x in z:
        a=x.reshape(-1,x.shape[-1]).astype(float)
        grams.append(a.T@a)
    return np.array(grams)


def covariance(g):
    return g[1:,1:]-np.outer(g[0,1:],g[0,1:])/g[0,0]


def fit(history_grams):
    days=np.asarray(history_grams).mean(1)
    c=covariance(days.sum(0))
    # Remove the direction along the incumbent prediction, to which Pearson correlation is insensitive.
    g=c[1:-1,1:-1]-np.outer(c[1:-1,0],c[0,1:-1])/c[0,0]
    def gradient(gram):
        v=covariance(gram)
        return v[1:-1,-1]*v[0,0]/v[0,-1]-v[1:-1,0]
    grad=gradient(days.sum(0))
    group=np.zeros((N,len(FAMILIES)))
    for i,(a,b) in enumerate(zip((0,*FAMILIES[:-1]),FAMILIES)):
        group[a:b,i]=1
    def split(b):
        return np.block([[b,np.zeros_like(b)],[np.zeros_like(b),b]])
    same=lambda b: np.tile(b,(2,1))
    weights=[]
    last=np.zeros(N)
    last[-2:]=.5
    weights.extend((np.tile(last,2),np.full(2*N,.5)))
    for basis,shrink,reg in ((np.ones((2*N,1)),False,.45),(same(group),False,.65),
                             (same(np.eye(N)),False,.8),(same(np.eye(N)),True,.8),(split(group),True,.8),
                             (np.eye(2*N),True,1.1),(same(np.eye(N)),True,2.)):
        gram=basis.T@g@basis
        rhs=basis.T@grad
        if shrink:
            local=np.stack([gradient(v)@basis for v in days])
            noise=local.std(0)*np.sqrt(len(days))
            rhs*=abs(rhs)/(abs(rhs)+noise+1e-9)
        diagonal=gram.diagonal().clip(.00001)
        beta=np.linalg.solve(gram+reg*np.diag(diagonal),rhs).clip(-1,2)
        weights.append(basis@beta)
    return np.array(weights)


def gains(g,ws):
    c=covariance(g)
    w=np.column_stack((np.ones(len(ws)),ws))
    return (w@c[:-1,-1])/np.sqrt(np.einsum('ij,jk,ik->i',w,c[:-1,:-1],w)*c[-1,-1])-c[0,-1]/np.sqrt(c[0,0]*c[-1,-1])


def train():
    OUT.mkdir(exist_ok=True)
    daily=[moments(d) for d in range(8,16)]
    folds=[]
    single=[]
    for end in (2,4,6):
        w=fit(daily[:end])
        folds.append([gains(g,w) for g in sum(daily[end:end+2])])
        single.extend([[gains(g,w) for g in day] for day in daily[end:end+2]])
    f=np.array(folds)
    mean=f.mean(0)
    eligible=(f>0).sum(0).min(0)>=2
    eligible&=(mean.min(0)>0)&(f[-1].min(0)>0)&((np.asarray(single)>0).sum(0).min(0)>=4)
    ids=[i for i in range(2,len(ARMS)) if eligible[i] and (mean[:,i,None]>mean[:,[0,1]]).all()]
    if not ids and eligible[1] and (mean[:,1]>mean[:,0]).all():
        ids=[1]
    pick=max(ids,key=lambda i:mean[:,i].min()) if ids else None
    report=dict(arms=ARMS,stages=STAGES,selected=pick,folds=f.tolist(),mean=mean.tolist(),evaluation_days=list(range(11,17)))
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('selected',pick,'mean gains',np.round(mean,6),flush=True)
    if pick is not None:
        np.savez(OUT/'model.npz',w=fit(daily)[pick],pick=pick)
    return pick


def revisions(raw,p,stages):
    q=p.copy()
    changes=np.empty((*q.shape[:2],N),'float32')
    for j,update in enumerate(stages):
        last=q[:,:,2].copy()
        q[:,:,2]+=update(raw,q)
        changes[:,:,j]=q[:,:,2]-last
    return q,changes


def adjusted(q,z,w):
    # Use the stage outputs as originally trained for the entire forward pass.
    # Extra weights are applied only after the last expert has made its prediction.
    mean=z.mean(1,keepdims=True)
    delta=np.zeros(z.shape[:2],dtype=float)
    for j in range(N):
        delta+=w[j]*(z[:,:,j]-mean[:,:,j])+w[N+j]*mean[:,:,j]
    out=q.copy()
    out[:,:,2]+=delta.astype('float32')
    return out


def layer():
    w=np.load(OUT/'model.npz')['w']
    start=early.incumbent.layer()
    updates=[module(s).correction() for s in STAGES]
    def apply(raw,p):
        q,z=revisions(raw,start(raw,p),updates)
        return adjusted(q,z,w)
    return apply


def cache_components():
    directory=OUT/'components'
    directory.mkdir(parents=True,exist_ok=True)
    trace=verifier.trace
    older=early.prior.parent.correction()
    begin=[early.prior.incumbent.correction(),early.incumbent.correction()]
    experts=[module(s).correction() for s in STAGES]
    reference=json.loads((parent.OUT/'reload_verification.json').read_text())
    for day in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,day)
        p,h,n=trace.load_day(day,raw)
        def run(r,p):
            p[:,:,2]+=older(r)
            for fn in begin:
                p[:,:,2]+=fn(r,p)
            return revisions(r,p,experts)
        q,z=run(raw,p)
        head,hz=run(raw[:n],h)
        assert np.max(abs(q[:n]-head))<1e-4 and np.max(abs(z[:n]-hz))<1e-4
        y=trace.replay.labels(day)
        base=trace.older.measure([(q.reshape(-1,3),q.reshape(-1,3),y)])
        assert abs(base['score']-reference['daily'][str(day+1)]['score'])<1e-10
        path=directory/f'day_{day}.npz'
        tmp=path.with_suffix(f'.{os.getpid()}.npz')
        np.savez(tmp,raw_hash=trace.replay.digest(raw),full=q,head=head,z=z,hz=hz,
                 hashes=[trace.replay.digest(a) for a in (q,head,z,hz)])
        tmp.replace(path)
        print('verified original expert outputs',day+1,flush=True)


def outer():
    trace=verifier.trace
    ref=json.loads((parent.OUT/'reload_verification.json').read_text())
    previous=None
    for name in ('raw_replay','reload_verification'):
        w=np.load(OUT/'model.npz')['w']
        rows=[]
        days={}
        blocks=[]
        for day in range(16,20):
            raw=trace.replay.original.features(trace.replay.original.DATA,day)
            with np.load(OUT/'components'/f'day_{day}.npz') as f:
                assert str(f['raw_hash'])==trace.replay.digest(raw)
                p,h,z,hz=[f[k] for k in ('full','head','z','hz')]
                assert f['hashes'].tolist()==[trace.replay.digest(a) for a in (p,h,z,hz)]
            q=adjusted(p,z,w)
            head=adjusted(h,hz,w)
            err=float(np.max(abs(head-q[:len(head)])))
            assert err<1e-4 and np.isfinite(q).all()
            np.testing.assert_array_equal(q[:,:,:2],p[:,:,:2])
            y=trace.replay.labels(day)
            row=(p.reshape(-1,3),q.reshape(-1,3),y)
            stat=trace.older.measure([row])
            assert abs(stat['baseline_score']-ref['daily'][str(day+1)]['score'])<1e-10
            days[str(day+1)]=dict(score=stat['score'],gain=stat['gain'],prefix_error=err)
            blocks.append(verifier.audit.blockstats(p,y.reshape(p.shape),[q[:,:,2].astype(float)-p[:,:,2]]))
            rows.append(row)
        pooled=trace.older.measure(rows)
        assert pooled['rows']==936250
        report=dict(parent=parent.OUT.name,pooled=pooled,daily=days,paired_score_percentiles=(.5*np.asarray(verifier.audit.intervals(blocks))[:,0]).tolist())
        if previous is not None:
            assert report==previous
        (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,pooled,flush=True)
        if pooled['gain']<=0:
            break
        previous=report


if __name__=='__main__':
    torch.set_num_threads(2)
    if train() is not None:
        cache_components()
        outer()
