"""Compose distinct market corrections whose direction replicates in both earlier date pairs."""
import json
import numpy as np
import experiment_term_structure_net_v1 as curve
import experiment_matured_market_innovations_v1 as delayed
import experiment_source_alpha_impact_v1 as source
import experiment_liquidity_clock_market_v1 as clocks
import experiment_relative_residual_tree_v1 as tree
from experiment_endpoint_innovation_v1 import prior
OUT=prior.ART/'curve_market_components_v1'

def delta(day):
    p=np.load(curve.OUT/f'pred_{day}.npy').astype(float)
    b=np.load(tree.OUT/f'pred_{day}.npy').astype(float)
    m=(np.load(delayed.OUT/f'pred_{day}.npy')-b).mean(1)
    raw=source.ex.v1_base(day);old=np.load(source.first.OUT/f'side_{day}.npy')
    w=np.array(json.loads((source.first.OUT/'results.json').read_text())['16']['weights'])
    impact=source.v1.blend(raw,old,w).mean(1)
    s=np.load(source.OUT/f'side_{day}.npy').mean(1)
    sw=np.array(json.loads((source.OUT/'results.json').read_text())['16']['weights'])[0]
    clock=np.load(clocks.OUT/f'market_{day}.npy')
    cw=np.array(json.loads((clocks.OUT/'results.json').read_text())['16']['alpha'])
    return p,np.stack((m,(s-impact)*sw,(clock-impact)*cw),axis=-1)

def main():
    OUT.mkdir(exist_ok=True);report={};early=[];outer=[];daily=[]
    for day in range(12,20):
        p,z=delta(day);y=prior.labels(day).reshape(p.shape)
        if day<16:
            early.append((p,z,y))
            if day in (13,15):
                a,h,b=map(np.concatenate,zip(*early));early=[]
                base=prior.corr(a.reshape(-1,3),b.reshape(-1,3))
                changes=np.stack([prior.corr((a+h[:,None,:,j]).reshape(-1,3),b.reshape(-1,3))-base for j in range(h.shape[-1])])
                report[str(day+1)]=changes.tolist()
            if day==15:
                w=(np.minimum(report['14'],report['16'])>0).astype(float)
                np.save(OUT/'weights.npy',w)
        else:
            q=p+np.einsum('tkj,jk->tk',z,w)[:,None,:]
            np.save(OUT/f'pred_{day}.npy',q.astype(np.float32))
            row=tuple(x.reshape(-1,3) for x in (p,q,y));outer.append(row);daily.append(prior.measure([row]))
    report.update(w=w.tolist(),daily=daily,result=prior.measure(outer))
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print(report['w'],report['result'],flush=True)
if __name__=='__main__':main()
