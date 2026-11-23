"""Package the full20 members; evaluate exact daily windows in batches, cross-check streaming."""
import argparse
import json
import shutil
import numpy as np
import torch
from project.script.evaluate import Evaluator
import spectral_full20_train_v1 as job

OUT=job.OUT
spec=job.spectral
prior=job.prior
source=spec.OUT/'commit_spectral_expert_fusion'
dest=OUT/'commit_spectral_full20'


def calibrators():
    previous=spec.previous
    cal=[]; mix=[]; strength_scores=[]
    for fold,days,prefix in [('A',range(8,12),8),('B',range(12,16),12),('outer',range(16,20),16)]:
        proj=spec.projector(range(prefix))
        for d,p,y,t,l in spec.fold_data(fold,days):
            x=prior.features(p);z=previous.inputs(p);res=y.astype(float)-p
            if d in (10,12,16):
                base,adj=previous.weights(cal)
            if d in (10,12,14,16):
                strength=np.ones(3) if d==10 else spec.current.choose(strength_scores)
            if d>=10:
                delta=.5*(x@base+z@adj)
                old=p+delta*strength
                h=spec.layer(t,l,proj,'four_factors')
                mix.append((h.T@h,h.T@(y[:,2]-old[:,2])))
                strength_scores.append(spec.current.evidence(p,delta,y))
            if d==16:
                orig=np.load(spec.OUT/'layer.npz')
                np.testing.assert_allclose(spec.fit(mix[:-1]),orig['weights'],atol=1e-8)
            cal.append((x.T@x,x.T@res,z.T@z,z.T@res,z.T@x))
    base,adj=previous.weights(cal);strength=spec.current.choose(strength_scores)
    return base*strength,adj*strength,spec.fit(mix),strength


def package():
    coef,adjust,mix,strength=calibrators()
    raw=job.raw_sources()
    graph=torch.from_numpy(np.load(OUT/'graph.npy'))
    proj=torch.tensor(np.load(OUT/'projector.npy'),dtype=torch.float32)
    dest.mkdir(exist_ok=True)
    for sub in ('model','conf'):(dest/sub).mkdir(exist_ok=True)
    for name in ('network.py','base.py'):shutil.copyfile(source/name,dest/name)
    def load(path):return torch.load(path,weights_only=True,map_location='cpu')
    params={}
    def add(prefix,path):
        for name,value in load(path).items():params[prefix+'.'+name]=value
    for i,seed in enumerate(job.SEEDS):
        for j in (0,1):
            add(f'backbones.{j}.raw_members.{i}',raw[seed,j])
            add(f'backbones.{j}.normalized_members.{i}',OUT/f'norm{j}_{seed}.pth')
        add(f'backbones.2.market_y1.{i}',raw[seed,1])
        add(f'backbones.2.normalized_market_y1.{i}',OUT/f'norm1_{seed}.pth')
        for kind,key in [('tcn_local','members128'),('complex_lru','members_lru')]:
            add(f'backbones.2.{key}.{i}',OUT/f'{kind}_{seed}.pth')
    mu,sd=job.dual.norm.global_stats()
    for j in range(3):
        params[f'backbones.{j}.norm_mu']=torch.from_numpy(mu)
        params[f'backbones.{j}.norm_sd']=torch.from_numpy(sd)
    params.update({'backbones.2.peer_graph':graph,'backbones.2.factor_projection':proj,
                   'backbones.2.mix':torch.tensor(mix.reshape(3,2,1),dtype=torch.float32),
                   'backbones.2.scale_reference':torch.tensor(job.dual.scales.reference(),dtype=torch.float64)})
    for j in range(3):
        cfg=f'conf/Model_{j}.json';path=f'model/Model_{j}.pth'
        shutil.copyfile(source/cfg,dest/cfg)
        state=dict(params,coef=torch.tensor(coef[:,j:j+1],dtype=torch.float32),adjust=torch.tensor(adjust[:,j:j+1],dtype=torch.float32))
        assert set(state)==set(load(source/path))
        pending=dest/(path+'.tmp');torch.save(state,pending);pending.replace(dest/path)
    return strength.tolist()


def market_grid(g):
    return torch.cat((g,g.mean(1,keepdim=True).expand_as(g),g.std(1,keepdim=True).expand_as(g)),dim=-1)


@torch.inference_mode()
def predict_day(ev,src,day,check=True):
    raw,y,_=src.load_block([day])
    grid=torch.from_numpy(raw.reshape(-1,50,100)).cuda()
    z=torch.from_numpy(job.norm_day(day).astype(np.float32).reshape(-1,50,100)).cuda()
    np.testing.assert_array_equal(y,prior.labels(day))
    np.testing.assert_array_equal(raw.astype(np.float16).reshape(z.shape),np.load(prior.CACHE/f'day_{day:02d}_x.npy',mmap_mode='r'))
    models=ev.models[0].backbones;third=models[2]
    market=market_grid(grid);norm=market_grid(z)
    peers=(third.peer_graph@z-z.mean(1,keepdim=True)).half().float()
    context=torch.cat((norm,peers),dim=-1)
    # [T,50,16,C]; each day starts with fifteen zero input frames.
    def windows(g):
        padded=torch.cat((g.new_zeros(15,50,g.shape[-1]),g),dim=0)
        return padded.unfold(0,16,1).permute(0,1,3,2)
    wm,wn,wp=map(windows,(market,norm,context))
    scale=torch.from_numpy(job.dual.scales.load_scale([day],job.dual.scales.reference()).reshape(-1,50,1)).cuda()
    pieces=[]
    for start in range(0,len(grid),20):
        sl=slice(start,start+20)
        rawwin=wm[sl].reshape(-1,16,300);normwin=wn[sl].reshape(-1,16,300);peerwin=wp[sl].reshape(-1,16,400)
        p=[]
        for model in models[:2]:
            a=torch.stack([m(rawwin) for m in model.raw_members]).mean(0)
            b=torch.stack([m(normwin) for m in model.normalized_members]).mean(0)
            p.append((.5*(a+b)).reshape(-1,50,1))
        tcn=torch.stack([m(peerwin)[:,1:2] for m in third.members128]).mean(0).reshape(-1,50,1)
        lru=torch.stack([m(peerwin)[:,1:2] for m in third.members_lru]).mean(0).reshape(-1,50,1)
        experts=torch.cat((.5*(tcn+lru),.5*(tcn-lru)),dim=2)*scale[sl]
        market_part=experts.mean(1,keepdim=True);rel=experts-market_part
        factor=third.factor_projection@rel
        q=experts[:,:,:1]+market_part@third.mix[0]+factor@third.mix[1]+(rel-factor)@third.mix[2]
        p.append(.5*(p[1]+q)+.25*(p[1].mean(1,keepdim=True)-q.mean(1,keepdim=True)))
        pieces.append(torch.cat(p,dim=-1).reshape(-1,3).cpu().numpy())
    p=np.concatenate(pieces)
    coef=np.concatenate([m.coef.cpu().numpy() for m in ev.models],axis=1)
    adjust=np.concatenate([m.adjust.cpu().numpy() for m in ev.models],axis=1)
    prediction=(p+.5*(prior.features(p)@coef+spec.previous.inputs(p)@adjust)).astype(np.float32)
    err=None
    if check:
        for m in ev.models:m.reset()
        count=265 if day in (0,16) else 20
        observed=np.concatenate([torch.cat([m.predict(x) for m in ev.models],1).cpu().numpy() for x in grid[:count]])
        err=float(np.abs(observed-prediction[:count*50]).max())
        if not np.isfinite(err) or err>5e-4:raise RuntimeError(('streaming mismatch',day,err))
    return prediction,y,err


def run(smoke):
    src,_=job.dual.outer.prepare()
    strength=None if smoke else package()
    ev=Evaluator(str(job.dual.common.DATA),str(source if smoke else dest),'cuda:0')
    sums=np.zeros((6,3));daily={}
    for d in ([16] if smoke else job.DAYS):
        p,y,err=predict_day(ev,src,d)
        a=p.astype(float);b=y.astype(float)
        sums+=np.stack((np.full(3,len(y)),a.sum(0),b.sum(0),(a*a).sum(0),(b*b).sum(0),(a*b).sum(0)))
        c=prior.corr(a,b)
        daily[d]=dict(rows=len(y),corr=c.tolist(),score=float(c@prior.WEIGHTS),stream_error=err)
        print(d,daily[d],flush=True)
        if smoke:
            expected=job.read(spec.OUT/'results.json')['daily'][str(d)]['four_factors']['candidate']
            np.testing.assert_allclose(c,expected,rtol=0,atol=1e-5)
            return
        job.dual.common.save_json(OUT/'daily_partial.json',daily)
    n,a,b,aa,bb,ab=sums
    c=(ab-a*b/n)/np.sqrt((aa-a*a/n)*(bb-b*b/n))
    assert (n==4681350).all() and np.isfinite(c).all()
    result=dict(days=job.DAYS,rows=int(n[0]),corr=c.tolist(),score=float(c@prior.WEIGHTS),score_4dp=round(float(c@prior.WEIGHTS),4),
                daily=daily,strength=strength,scope='in-sample full20, vectorized identical daily windows; streaming cross-check each day',
                package=str(dest), hashes={str(f.relative_to(dest)):job.dual.common.sha256(f) for f in dest.rglob('*') if f.suffix in ('.pth','.py','.json')})
    job.dual.common.save_json(OUT/'results.json',result)
    print('FULL20',c,result['score'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args().smoke)
