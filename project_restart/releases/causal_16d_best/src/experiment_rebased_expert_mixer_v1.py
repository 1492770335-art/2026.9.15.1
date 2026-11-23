"""Refit the released expert mixer on error examples from successively reconstructed bases."""
import json
import numpy as np
import torch
import experiment_peer_pooled_online_errors_v1 as refit
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'rebased_expert_mixer_v1'
expert=refit.incumbent
roll=expert.rolling


def compare(p,fs,t,old,new,mask):
    a=expert.revised(old,p,fs,t,True)
    b=expert.revised(new,p,fs,t,True)
    prior=np.where(mask,a,p[t])
    return prior,(prior+.5*(b-prior)).astype('float32')


def load():
    old=expert.Mixer().eval()
    old.load_state_dict(torch.load(expert.OUT/'net.pth',map_location='cpu',weights_only=True))
    new=expert.Mixer().eval()
    new.load_state_dict(torch.load(OUT/'net.pth',map_location='cpu',weights_only=True))
    original=np.load(expert.OUT/'mask.npy')
    keep=np.load(OUT/'mask.npy')
    parent=roll.gls.incumbent.load()
    encoders=[fn for fn,_ in roll.panel.models(16)]+[roll.gls.family.loader(m,16) for m in roll.EXTRA]
    def predict(raw,p):
        base=parent(raw,p)
        fs=[fn(raw) for fn in encoders]
        q=base.copy()
        for begin in range(0,len(q),96):
            t=np.arange(begin,min(begin+96,len(q)))
            baseline,candidate=compare(base,fs,t,old,new,original)
            q[t]=np.where(keep,candidate,baseline)
        return q
    return predict


def select(g,days,prior_mask):
    # Until day 11 both training views equal the initial reference. The first
    # fold can test a new target but cannot distinguish two refits of an active head.
    keep=[]
    for k,active in enumerate(prior_mask):
        start=int(active)
        folds=g[start:,:,k]
        daily=np.asarray(days)[2*start:,k]
        keep.append(folds[:,0].min()>1e-6 and folds[:,1].mean()>1e-6
                    and (daily>1e-6).sum()>=int(np.ceil(2*len(daily)/3)))
    return np.array(keep)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    prior_mask=np.load(expert.OUT/'mask.npy')
    mix=[];original=[];recent=[];folds=[];days=[]
    for end in (8,10,12,14):
        coef=roll.fit(mix)
        nets=[expert.fit(rows) for rows in (original,recent)] if original else []
        groups=[[],[]]
        for d in (end,end+1):
            fs=roll.sources(d);t=np.arange(len(fs[0]));y=labels(d).reshape(-1,50,3)
            adaptive=roll.predict(fs,t,coef)
            fixed=np.load(roll.panel.OUT/f'panel_{d}.npy',mmap_mode='r')
            impact=np.load(refit.core.OUT/f'pred_{d}.npy',mmap_mode='r')
            online=np.load(refit.OUT/f'reference_{d}.npy',mmap_mode='r')
            mix.append(roll.moments(fs,y,t[d%5::5]))
            ix=t[d%19::19]
            # Same observations, architecture, optimization steps and seed in both arms.
            for arm,views in ((original,(adaptive,fixed)),(recent,(impact,online))):
                for p in views:
                    arm.append(expert.training_record(p,fs[1:],ix,y))
            if not nets:continue
            ix=t[d%4::4]
            for j,p in enumerate((online,adaptive)):
                old,q=compare(p,fs[1:],ix,*nets,prior_mask)
                if d==end and j==0:
                    st=ix[ix<177]
                    _,head=compare(p[:177],[f[:177] for f in fs[1:]],st,*nets,prior_mask)
                    np.testing.assert_allclose(q[:len(st)],head,rtol=0,atol=2e-6)
                row=tuple(a.reshape(-1,3) for a in (old,q,y[ix]))
                groups[j].append(row)
                if not j:
                    s=measure([row]);days.append(np.subtract(s['candidate'],s['base']))
        if nets:
            folds.append([measure(rows) for rows in groups])
            print(end+1,[np.subtract(s['candidate'],s['base']).tolist() for s in folds[-1]],flush=True)
    rep=expert.fit(original)
    st=torch.load(expert.OUT/'net.pth',map_location='cpu',weights_only=True)
    for k,v in rep.state_dict().items():
        torch.testing.assert_close(v,st[k],rtol=0,atol=2e-6)
    g=np.array([[np.subtract(s['candidate'],s['base']) for s in f] for f in folds])
    keep=select(g,days,prior_mask)
    (OUT/'results.json').write_text(json.dumps(dict(retained=keep.tolist(),folds=folds,
                daily=np.array(days).tolist(),original_parameters_verified=True),indent=2))
    print('retained',keep,flush=True)
    if keep.any():
        np.save(OUT/'mask.npy',keep)
        torch.save(expert.fit(recent).state_dict(),OUT/'net.pth')
        replay=roll.gls.best.walk.replay
        res=replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        old=json.loads((expert.OUT/'reload_verification.json').read_text())['pooled']
        np.testing.assert_allclose(np.array(res['pooled']['candidate'])[~keep],np.array(old['candidate'])[~keep],rtol=0,atol=1e-9)
        print('official',res['pooled']['score'],res['pooled']['score']-old['score'],flush=True)
        if res['pooled']['score']>old['score']:
            check=replay(load())
            np.testing.assert_allclose(res['pooled']['candidate'],check['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':main()
