"""Apply the previously fitted side-network innovation to the execution-memory forecast."""
import json
import numpy as np
import torch
import experiment_bid_ask_side_network_v1 as side
import experiment_price_level_execution_memory_v1 as parent
import experiment_causal_price_level_centering_v1 as readout
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'side_network_execution_fusion_v1'


def add(p,n,w):
    m,r=readout.basis(p,n)
    return (p+(m@w[0])[:,None]+r@w[1]).astype('float32')


def evaluate(out,parent_dir,expert,restore):
    out.mkdir(exist_ok=True)
    w=np.load(expert.OUT/'coef.npy')
    rows=[]
    for d in range(16,20):
        p=np.load(parent_dir/f'pred_{d}.npy');n=np.load(expert.OUT/f'side_{d}.npy')
        q=add(p,n,w)
        np.save(out/f'pred_{d}.npy',q)
        rows.append((p.reshape(-1,3),q.reshape(-1,3),labels(d)))
    result=measure(rows)
    (out/'results.json').write_text(json.dumps({'20':result,'coefficients':str(expert.OUT/'coef.npy')},indent=2))
    print(result,flush=True)
    if result['gain']>0:
        ck=torch.load(expert.OUT/'net_16.pth',map_location='cpu',weights_only=True)
        net=restore(ck).eval()
        net.load_state_dict(ck)
        replay.main(parent_dir,out,lambda raw,p:add(p,expert.infer(net,raw),w))


def main():
    evaluate(OUT,parent.OUT,side,lambda st:side.SideNet(st['encoder.0.weight'].shape[1]))


if __name__=='__main__':main()
