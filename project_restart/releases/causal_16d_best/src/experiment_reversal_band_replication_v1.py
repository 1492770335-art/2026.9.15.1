"""Check an independent neural seed and isolate each delayed-head contribution to Y3."""
import json
import numpy as np
import torch
import experiment_pressure_reversal_payoff_v1 as model

OUT=model.spell.peer.ART/'reversal_band_replication_v1'
SEED=46021


@torch.inference_mode()
def terms(raw,t,enc,nets):
    x,v,_,_=model.inputs(raw,t,enc)
    parts=[]
    for lo in range(0,len(t),64):
        sl=slice(lo,lo+64)
        d=nets[1](x[sl],True)[0]-nets[0](x[sl],False)[0]
        bands=(d*v[sl,:,None]).cpu().numpy()
        weighted=.15*bands*(model.heads.UNITS[2]/model.heads.UNITS)
        parts.append(weighted-weighted.mean(1,keepdims=True))
    return np.concatenate(parts)


def ic(a,b):
    a=a.astype(float).ravel();b=b.astype(float).ravel()
    a-=a.mean();b-=b.mean()
    return float(a@b/np.sqrt((a@a)*(b@b)))


def scores(rows):
    p,y,z=(np.concatenate(v) for v in zip(*rows))
    base=ic(p[:,:,2],y[:,:,2])
    separate=[ic(p[:,:,2]+z[:,:,j],y[:,:,2])-base for j in range(3)]
    a=model.spell.bands(p)[:,:,2];b=model.spell.bands(y)[:,:,2]
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    return dict(full_y3_gain=ic(p[:,:,2]+z.sum(2),y[:,:,2])-base,
                single_band_y3_gain=separate,relative_far_gain=ic(a+z[:,:,2],b)-ic(a,b))


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    daily,folds=[],[]
    for end,stop in ((8,12),(12,14),(14,16)):
        enc=model.heads.block.paths.decoder.net(end)
        nets=model.train(end,enc,seed=SEED)
        for first in range(end,stop,2):
            rows=[]
            for d in (first,first+1):
                raw=model.spell.peer.PARENT.before.book.rawday(d)
                t=np.arange(d%4,len(raw),4)
                z=terms(raw,t,enc,nets)
                if d==first:
                    n=193
                    sh=terms(raw[:n],t[t<n],enc,nets)
                    np.testing.assert_allclose(sh,z[:len(sh)],atol=3e-5,rtol=0)
                y=model.spell.peer.labels(d).reshape(-1,50,3)[t]
                rows.append((model.spell.peer.reference(d)[t],y,z))
                daily.append(scores(rows[-1:]))
            folds.append(scores(rows))
            print(first+1,folds[-1],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(seed=SEED,folds=folds,daily=daily,
            bands=[[1,5],[6,20],[21,60]]),indent=2))


if __name__=='__main__':main()
