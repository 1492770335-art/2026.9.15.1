"""Combine two pre-specified reversal-head training seeds with equal weight on the selected Y3 update."""
import json
import numpy as np
import torch
import experiment_pressure_reversal_payoff_v1 as reversal
import experiment_reversal_band_replication_v1 as replication
import experiment_pressure_payoff_joint_v1 as joint

OUT=reversal.spell.peer.ART/'pressure_seed_average_v1'


def load():
    parent=reversal.spell.load()
    np.testing.assert_array_equal(np.load(reversal.OUT/'keep.npy'),[[0,0,1],[0,0,0]])
    enc=reversal.heads.block.paths.decoder.net(16)
    pairs=[]
    for path in (reversal.OUT,OUT):
        nets=[]
        for k in (0,1):
            st=torch.load(path/f'head_{k}.pth',map_location='cpu',weights_only=True)
            net=reversal.ReversalHead(st['enc.0.weight'].shape[1]).cuda().eval()
            net.load_state_dict(st)
            nets.append(net)
        pairs.append(nets)
    @torch.inference_mode()
    def predict(raw,p):
        q=parent(raw,p).copy()
        x,v,_,_=reversal.inputs(raw,np.arange(len(raw)),enc)
        for pos in range(0,len(x),64):
            sl=slice(pos,pos+64)
            local=[]
            for control,candidate in pairs:
                d=candidate(x[sl],True)[0]-control(x[sl],False)[0]
                far=.15*reversal.heads.levels(d*v[sl,:,None])[:,:,2].cpu().numpy()
                local.append(far-far.mean(1,keepdims=True))
            q[sl,:,2]+=.5*(local[0]+local[1])
        return q
    return predict


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    r=json.loads((replication.OUT/'results.json').read_text())
    assert min(f['full_y3_gain'] for f in r['folds'])>0
    models=reversal.train(16,reversal.heads.block.paths.decoder.net(16),seed=r['seed'])
    for k,m in enumerate(models):torch.save(m.cpu().state_dict(),OUT/f'head_{k}.pth')
    del models
    best=json.loads((joint.OUT/'raw_replay.json').read_text())['pooled']
    first=None
    for name in ('raw_replay','reload_verification'):
        result=reversal.spell.peer.walk.replay(load())
        np.testing.assert_allclose(best['candidate'][:2],result['pooled']['candidate'][:2],atol=1e-10,rtol=0)
        if first is not None:
            np.testing.assert_allclose(first['pooled']['candidate'],result['pooled']['candidate'],atol=1e-10,rtol=0)
            for d in result['daily']:
                assert abs(result['daily'][d]['score']-first['daily'][d]['score'])<1e-10
        (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(name,result['pooled']['score'],flush=True)
        if result['pooled']['score']<=best['score']:break
        first=result


if __name__=='__main__':main()
