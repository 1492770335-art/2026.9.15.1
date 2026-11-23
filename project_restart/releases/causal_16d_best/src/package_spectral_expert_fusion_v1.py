"""Export component-specific expert mixing and compare real windows with offline evaluation."""
import json
import shutil
import numpy as np
import torch
import polars as pl
from project.script.evaluate import Evaluator
import experiment_spectral_expert_fusion_v1 as run


def main():
    report=json.loads((run.OUT/'results.json').read_text())
    if report['selected']!='four_factors' or report['outer']['gain']<=0:
        raise RuntimeError('No four-factor improvement')
    weights=np.load(run.OUT/'layer.npz')
    src=run.current.OUT/'commit_target_strength'
    dest=run.OUT/'commit_spectral_expert_fusion'
    dest.mkdir(exist_ok=True)
    text=(src/'network.py').read_text()
    init='        self.members_lru = nn.ModuleList([PeerLRU() for _ in range(3)])'
    before="""        p128 = 0.5 * (torch.stack([m(history)[:, 1:2] for m in self.members128]).mean(0) + torch.stack([m(history)[:, 1:2] for m in self.members_lru]).mean(0))
        if self.scale_output:
            p128 = p128 * scale"""
    after="""        tcn = torch.stack([m(history)[:, 1:2] for m in self.members128]).mean(0)
        lru = torch.stack([m(history)[:, 1:2] for m in self.members_lru]).mean(0)
        experts = torch.cat((.5*(tcn+lru), .5*(tcn-lru)), dim=1)
        if self.scale_output:
            experts = experts * scale
        market = experts.mean(dim=0, keepdim=True)
        relative = experts-market
        factor = self.factor_projection @ relative
        p128 = experts[:, :1] + market @ self.mix[0] + factor @ self.mix[1] + (relative-factor) @ self.mix[2]"""
    # Preserve the history tensor named market, read by the other branch below.
    after=after.replace('market = experts','market_components = experts').replace('experts-market','experts-market_components').replace('+ market @','+ market_components @')
    if text.count(init)!=1 or text.count(before)!=1:
        raise RuntimeError('Unexpected network layout')
    text=text.replace(init,init+"\n        self.register_buffer('factor_projection',torch.zeros(50,50))\n        self.register_buffer('mix',torch.zeros(3,2,1))").replace(before,after)
    (dest/'network.py').write_text(text)
    shutil.copyfile(src/'base.py',dest/'base.py')
    for folder in ('conf','model'):(dest/folder).mkdir(exist_ok=True)
    for i in range(3):
        cfg=f'conf/Model_{i}.json'
        shutil.copyfile(src/cfg,dest/cfg)
        path=f'model/Model_{i}.pth'
        state=torch.load(src/path,map_location='cpu',weights_only=True)
        state['backbones.2.factor_projection']=torch.tensor(weights['projector'],dtype=torch.float32)
        state['backbones.2.mix']=torch.tensor(weights['weights'].reshape(3,2,1),dtype=torch.float32)
        torch.save(state,dest/path)
    data=run.prior.ROOT/'project/data/public.parquet'
    ev=Evaluator(str(data),str(dest),'cuda:0')
    params=run.settings()[16];errors={}
    with torch.inference_mode():
        for d,p,y,tcn,lru in run.fold_data('outer',range(16,20)):
            ticks={16:280,17:20}.get(d)
            if not ticks:break
            k=ticks*50
            pp=p[:k].copy()
            pp[:,2]+=run.layer(tcn[:k],lru[:k],weights['projector'],'four_factors')@weights['weights']
            expected=run.calibrate(pp,params)
            frame=(pl.scan_parquet(data).filter(pl.col('date')==d).sort(['time','code']).head(k).collect())
            names=sorted((n for n in frame.columns if n.startswith('f')),key=lambda n:int(n[1:]))
            grid=frame.select(names).to_numpy().astype(np.float32).reshape(-1,50,100)
            for model in ev.models:model.reset()
            observed=[]
            for tick in grid:
                x=torch.from_numpy(tick).cuda()
                observed.append(torch.cat([m.predict(x) for m in ev.models],dim=1).cpu().numpy())
            errors[d]=float(np.abs(expected-np.concatenate(observed)).max())
            if not np.isfinite(errors[d]) or errors[d]>5e-4:raise ValueError(errors)
    (run.OUT/'stream_check.json').write_text(json.dumps(errors,indent=2))
    print(errors)

if __name__=='__main__':
    main()
