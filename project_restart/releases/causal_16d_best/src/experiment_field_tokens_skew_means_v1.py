"""Apply two independently trained neural refinements to their disjoint earlier-selected targets."""
import json
import numpy as np
import torch
import experiment_target_book_family_tokens_v1 as tokens
import experiment_asymmetric_return_distribution_v1 as dist
import experiment_pairwise_book_tree_v1 as base
import experiment_crossfit_predictable_tails_v1 as validation
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'field_tokens_skew_means_v1'


def sources():
    a=np.load(tokens.OUT/'coef.npy');b=np.load(dist.OUT/'coef.npy')
    masks=[np.any(w!=0,axis=(0,1)) for w in (a,b)]
    if (masks[0]&masks[1]).any():raise ValueError('Earlier target selections overlap')
    return masks,[a,b]


def cached():
    OUT.mkdir(exist_ok=True);gates,_=sources();rows=[];daily={}
    for d in range(16,20):
        p=np.load(base.OUT/f'pred_{d}.npy');q=p.copy()
        for gate,root in zip(gates,(tokens.OUT,dist.OUT)):
            part=np.load(root/f'pred_{d}.npy')
            np.testing.assert_allclose(p[:,:,~gate],part[:,:,~gate],atol=2.e-7)
            q[:,:,gate]=part[:,:,gate]
        np.save(OUT/f'pred_{d}.npy',q)
        row=(p.reshape(-1,3),q.reshape(-1,3),labels(d));rows.append(row);daily[d+1]=measure([row])
    val=measure(rows);report={'20':val,'daily':daily,'earlier_horizon_masks':[v.tolist() for v in gates]}
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print(val,flush=True)


def refinement():
    gates,weights=sources();models=[]
    for src,restore in ((tokens,tokens.restore),(dist,lambda st:dist.DistributionNet(st['current.0.weight'].shape[1]))):
        st=torch.load(src.OUT/'net_16.pth',map_location='cpu',weights_only=True)
        net=restore(st).cuda().eval();net.load_state_dict(st);models.append(net)
    def refine(raw,p):
        q=p.copy()
        for net,w,keep in zip(models,weights,gates):
            m,r=validation.readout.basis(p,validation.phase.infer(net,raw))
            proposal=(p+(m@w[0])[:,None]+r@w[1]).astype('float32')
            q[:,:,keep]=proposal[:,:,keep]
        return q
    return OUT,refine


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.replay:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:cached()

