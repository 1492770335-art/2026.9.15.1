"""Apply the trained stock-relative quantile correction alongside the market corrections."""
import json
import numpy as np
import experiment_curve_market_components_v1 as market
import experiment_rolling_quantile_residual_v1 as quant
from experiment_relative_residual_tree_v1 import OUT as TREE_OUT
prior=market.prior
OUT=prior.ART/'curve_quantile_market_v1'

def combine(common,tree,rank):
    delta=rank.astype(float)-tree
    return common+(delta-delta.mean(1,keepdims=True))

def main():
    OUT.mkdir(exist_ok=True)
    w=np.load(market.OUT/'weights.npy');rows=[];report={'daily':{}}
    for d in range(12,20):
        p,z=market.delta(d);common=p+np.einsum('tkj,jk->tk',z,w)[:,None]
        tree=np.load(TREE_OUT/f'pred_{d}.npy');rank=np.load(quant.OUT/f'pred_{d}.npy')
        q=combine(common,tree,rank)
        np.testing.assert_allclose(q.mean(1),common.mean(1),atol=1e-8)
        np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
        y=prior.labels(d).reshape(p.shape)
        row=tuple(a.reshape(-1,3) for a in (common,q,y))
        report['daily'][d+1]=prior.measure([row]);rows.append(row)
        if d in (15,19):
            report['selection_days' if d==15 else 'outer']=prior.measure(rows);rows=[]
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print(report['outer'],flush=True)
if __name__=='__main__':main()
