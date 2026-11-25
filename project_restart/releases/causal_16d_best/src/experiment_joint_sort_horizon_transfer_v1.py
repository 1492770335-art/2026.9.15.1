"""Fit the characteristic-portfolio head specifically to Y2 and the five-to-twenty-step interval."""
import argparse
import json

import numpy as np

import experiment_y3_realized_sort_rotation_v1 as rotation

portfolios,clock,parent=rotation.portfolios,rotation.clock,rotation.parent
OUT=parent.ART/'joint_sort_horizon_transfer_v1'
ARMS=('incumbent','reassigned stocks','long-horizon head','Y2-specific head',
      'horizon agreement','Y2 above long','Y2 with agreeing long component')


def medium(p):
    ratio=(.272164/.558942)/parent.base.RATIO
    return np.stack((p[:,:,2],ratio*p[:,:,0],p[:,:,1]),axis=2)


def deltas(raw,p,st,long_state):
    near=portfolios.corrections(raw,medium(p),st)[:,:,2]
    far=portfolios.corrections(raw,p,long_state)[:,:,2]
    agree=np.where(near*far>0,np.sign(near)*np.minimum(abs(near),abs(far)),0)
    return np.stack((np.roll(near,13,axis=1),far,near,agree,near-far,.7*near+.3*agree),axis=2)


def run():
    history=[];longs=[];obs=[];days=[];fixed=[]
    for d in range(4,16):
        raw,refs,y=parent.history.training_day(d)
        refs=np.array(refs,copy=True)
        today=[parent.exposure.observations(p,y) for p in refs]
        if d>=8:
            if d%2==0:
                st=portfolios.fit(history);long=portfolios.fit(longs);w=parent.exposure.fit(obs)[6]
            if d==8:anchor=st,long
            refs[:,:,:,2]+=parent.apply_market(*parent.historical_heads(d),w)
            strict=np.load(parent.ART/'cutoff_reference_panel_v1'/f'panel_{d}.npy')
            group=[];locked=[]
            for i,p in enumerate([*refs,strict]):
                for weights,which in (((st,long),group),(anchor,locked)):
                    z=deltas(raw,p,*weights)
                    if d==8 and i==0:
                        n=2502
                        np.testing.assert_allclose(z[:n],deltas(raw[:n],p[:n],*weights),rtol=0,atol=5e-7)
                    which.append(parent.base.records(medium(p),medium(y),z))
            days.append(group);fixed.append(locked)
        obs.append(tuple(np.mean(v,0) for v in zip(*today)))
        for log,ps,label in ((history,[medium(p) for p in refs],medium(y)),(longs,refs,y)):
            rows=[portfolios.record(raw,p,label,d) for p in ps]
            log.append([tuple(np.mean(v,0) for v in zip(*s)) for s in zip(*rows)])
        print('completed training-period day',d+1,flush=True)
    def score(ds):return np.array([parent.base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(ds,0)])
    h=np.stack((score(days[:4]),score(days[4:])))
    f=np.stack((score(fixed[:4]),score(fixed[4:])))
    daily=np.stack([score([r]) for r in days])
    pick=max(range(3,len(ARMS)),key=lambda i:h[0,:,:,i].min())
    ok=(h[:,:,:,pick].min()>0 and f[:,-1,0,pick].min()>0 and (daily[:,:,0,pick]>0).sum(0).min()>=5
        and (h[:,:,0,pick]>h[:,:,0,1]).all())
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick if ok else 0,first_window_choice=pick,
        windows=[[9,12],[13,16]],halves=h.tolist(),frozen_day8=f.tolist(),daily=daily.tolist()),indent=2))
    if ok:np.savez(OUT/'model.npz',pick=pick,**portfolios.fit(history))
    print('selected',pick if ok else 0,'strict Y2',h[:,-1,0].round(6).tolist(),flush=True)


def correction():
    with np.load(portfolios.OUT/'model.npz') as f:long=dict(f)
    with np.load(OUT/'model.npz') as f:st=dict(f)
    return lambda raw,p:deltas(raw,p,st,long)[:,:,int(st['pick'])-1]


def evaluate(cached):
    before=rotation.correction()
    update=correction()
    verifier,read=(parent.parent if cached else parent).replay_with()
    if cached:upstream=parent.correction_from_heads()
    ref='cached_validation' if cached else 'reload_verification'
    expected=json.loads((rotation.OUT/(ref+'.json')).read_text())
    def prediction(raw,p):
        old=p.copy();old[:,:,2]+=before(raw,p)
        new=old.copy();new[:,:,1]+=update(raw,p)
        assert np.array_equal(new[:,:,[0,2]],old[:,:,[0,2]])
        return old,new
    rows=[];daily={}
    for d in range(16,20):
        raw=verifier.trace.replay.original.features(verifier.trace.replay.original.DATA,d)
        p,h,n=read(d,raw)
        if cached:
            op=np.load(parent.ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy').reshape(p.shape)
            nets=[np.load(parent.ART/parent.experts.experts.SOURCES[j]/f'side_{d}.npy').reshape(p.shape) for j in parent.EXPERT_IDS]
            p[:,:,2]+=upstream(op,nets)
        old,new=prediction(raw,p)
        row=(old.reshape(-1,3),new.reshape(-1,3),verifier.trace.replay.labels(d))
        m=verifier.trace.older.measure([row])
        assert abs(m['baseline_score']-expected['daily'][str(d+1)]['score'])<1e-9
        daily[d+1]=dict(score=m['score'],gain=m['gain'],corr=m['candidate'])
        if not cached:
            _,head=prediction(raw[:n],h)
            err=float(abs(new[:n]-head).max())
            assert err<1e-4
            daily[d+1]['prefix_error']=err
        rows.append(row)
    pooled=verifier.trace.older.measure(rows)
    assert pooled['rows']==936250
    return dict(parent=rotation.OUT.name,pooled=pooled,daily=daily)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),nargs='?',default='train')
    mode=parser.parse_args().mode
    if mode=='train':run()
    else:
        screened=mode=='screen'
        report=evaluate(screened)
        (OUT/('cached_validation.json' if screened else 'raw_replay.json')).write_text(json.dumps(report,indent=2))
        if not screened:
            assert report==evaluate(False)
            (OUT/'reload_verification.json').write_text(json.dumps(report,indent=2))
        print(mode,report['pooled'],flush=True)
