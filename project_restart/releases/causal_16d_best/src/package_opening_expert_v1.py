"""Export open-dependent blend and independently replay all four days from parquet."""
import json
import shutil
import numpy as np
import torch
from project.script.evaluate import Evaluator
import spectral_full20_train_v1 as job
from spectral_full20_evaluate_v1 import market_grid
import experiment_opening_expert_v1 as run


def package():
    report=json.loads((run.OUT/'results.json').read_text())
    if report['selected']!='opening_mix' or report['outer']['gain']<=0:
        raise RuntimeError('No validated opening mixture')
    weights=report['weights']
    source=run.spec.OUT/'commit_spectral_expert_fusion'
    dest=run.OUT/'commit_opening_expert'
    dest.mkdir(exist_ok=True)
    for folder in ('model','conf'):(dest/folder).mkdir(exist_ok=True)
    code=(source/'network.py').read_text()
    code='import math\n'+code
    def replace_once(text,old,new):
        if text.count(old)!=1:raise RuntimeError('Unexpected source count: '+old)
        return text.replace(old,new)
    code=replace_once(code,'        return .5*(p+q)','        return .5*(p+q)+self.raw_offset*(p-q)')
    at=code.index('class DualTemporalScalePeerTarget')
    code=code[:at]+replace_once(code[at:],'p1 = 0.5 * (rawy1 + normy1)','p1 = 0.5 * (rawy1 + normy1) + self.raw_offset * (rawy1 - normy1)')
    code=replace_once(code,"        self.target = config['target']","        self.target = config['target']\n        self.opening_mix = config['opening_mix']")
    at=code.index('class CalibratedHorizon')
    text=replace_once(code[at:],'        self.history = None','        self.history = None\n        self.session_tick = 0')
    line='        p = torch.cat([model.predict(x) for model in self.backbones], dim=1)'
    text=replace_once(text,line,"""        phase = math.exp(-self.session_tick/256.)
        self.session_tick += 1
        for target in (0,1):
            constant, amplitude = self.opening_mix[target]
            self.backbones[target].raw_offset = max(-.35,min(.35,constant + amplitude*phase))
        self.backbones[2].raw_offset = self.backbones[1].raw_offset
"""+line)
    (dest/'network.py').write_text(code[:at]+text)
    shutil.copyfile(source/'base.py',dest/'base.py')
    for i in range(3):
        cp=f'conf/Model_{i}.json'
        cfg=json.loads((source/cp).read_text());cfg['opening_mix']=weights
        (dest/cp).write_text(json.dumps(cfg,indent=2))
        path=f'model/Model_{i}.pth'
        shutil.copyfile(source/path,dest/path)
    return dest,np.array(weights),report


def windows(grid):
    pad=grid.new_zeros(15,50,grid.shape[-1])
    return torch.cat((pad,grid),dim=0).unfold(0,16,1).permute(0,1,3,2)


@torch.inference_mode()
def replay(ev,src,d,weights):
    raw,y,_=src.load_block([d])
    (old, new), err = replay_features(ev, raw, weights, 280 if d == 16 else 20)
    return (old, new, y), err


@torch.inference_mode()
def replay_features(ev, raw, weights, sample_ticks):
    grid=torch.from_numpy(raw.reshape(-1,50,100)).cuda().float()
    anchor=ev.models[0].backbones[0]
    stats=(anchor.norm_mu.cpu().numpy(),anchor.norm_sd.cpu().numpy())
    z=torch.from_numpy(job.dual.outer.transform(raw,*stats).astype(np.float32).reshape(-1,50,100)).cuda()
    models=ev.models[0].backbones
    third=models[2]
    wm=windows(market_grid(grid));wn=windows(market_grid(z))
    peer=(third.peer_graph@z-z.mean(1,keepdim=True)).half().float()
    wp=windows(torch.cat((market_grid(z),peer),dim=2))
    scale=torch.from_numpy(job.dual.scales.causal_scale(raw.reshape(-1,50,100)[:,:,0],third.scale_reference.item()).reshape(-1,50,1)).cuda()
    parts=[];controls=[]
    for start in range(0,len(grid),20):
        stop=min(len(grid),start+20);sl=slice(start,stop)
        rw=wm[sl].reshape(-1,16,300);nw=wn[sl].reshape(-1,16,300);pw=wp[sl].reshape(-1,16,400)
        phase=torch.exp(-torch.arange(start,stop,device='cuda')/256.)[:,None,None]
        base=[];candidate=[]
        for j,model in enumerate(models[:2]):
            a=torch.stack([m(rw) for m in model.raw_members]).mean(0).reshape(-1,50,1)
            b=torch.stack([m(nw) for m in model.normalized_members]).mean(0).reshape(-1,50,1)
            base.append(.5*(a+b))
            off=(weights[j,0]+weights[j,1]*phase).clamp(-.35,.35)
            candidate.append(base[-1]+off*(a-b))
        a=torch.stack([m(pw)[:,1:2] for m in third.members128]).mean(0).reshape(-1,50,1)
        b=torch.stack([m(pw)[:,1:2] for m in third.members_lru]).mean(0).reshape(-1,50,1)
        experts=torch.cat((.5*(a+b),.5*(a-b)),dim=2)*scale[sl]
        market=experts.mean(1,keepdim=True);relative=experts-market
        factor=third.factor_projection@relative
        q=experts[:,:,:1]+market@third.mix[0]+factor@third.mix[1]+(relative-factor)@third.mix[2]
        for signals,storage in ((base,controls),(candidate,parts)):
            p1=signals[1]
            signals.append(.5*(p1+q)+.25*(p1.mean(1,keepdim=True)-q.mean(1,keepdim=True)))
            storage.append(torch.cat(signals,dim=-1).reshape(-1,3).cpu().numpy())
    coef=np.column_stack([m.coef.cpu().numpy().ravel() for m in ev.models])
    adj=np.column_stack([m.adjust.cpu().numpy().ravel() for m in ev.models])
    def calibrate(p):
        return (p+.5*(run.prior.features(p)@coef+run.spec.previous.inputs(p)@adj)).astype(np.float32)
    old=calibrate(np.concatenate(controls));new=calibrate(np.concatenate(parts))
    for m in ev.models:m.reset()
    count=min(len(grid),sample_ticks)
    direct=np.concatenate([torch.cat([m.predict(tick) for m in ev.models],dim=1).cpu().numpy() for tick in grid[:count]])
    err=float(np.abs(direct-new[:count*50]).max())
    if not np.isfinite(err) or err>5e-4:raise RuntimeError(('stream comparison',err))
    return (old,new),err


def main():
    dest,weights,report=package()
    src,_=job.dual.outer.prepare()
    ev=Evaluator(str(job.dual.common.DATA),str(dest),'cuda:0')
    record={'daily':{},'stream_max_error':{},'scope':'batched inference from raw parquet, per-day stream checks'}
    rows=[]
    for d in range(16,20):
        row,err=replay(ev,src,d,weights)
        rows.append(row);record['daily'][d]=run.prior.measure([row])
        record['stream_max_error'][d]=err
        print(d,record['daily'][d]['gain'],err,flush=True)
    record['outer']=run.prior.measure(rows)
    for key in ('base','candidate','score','baseline_score'):
        np.testing.assert_allclose(record['outer'][key],report['outer'][key],rtol=0,atol=1e-5)
    (run.OUT/'replay.json').write_text(json.dumps(record,indent=2))
    print(record['outer'],flush=True)

if __name__=='__main__':
    main()
