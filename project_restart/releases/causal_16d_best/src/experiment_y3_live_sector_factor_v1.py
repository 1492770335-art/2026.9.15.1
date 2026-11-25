"""Use live factor membership directly inside the retained 12% far-horizon network branch."""
import json
import numpy as np
import torch
import experiment_y3_adaptive_sector_routing_v1 as live

incumbent=live.sector
base=live.base
clock=live.clock
OUT=clock.ART/'y3_live_sector_factor_v1'
WEIGHT=.12


def run():
    folds=[];days=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        ids=incumbent.grouping(end);typ=live.prior(end,ids);net=incumbent.train(end,ids)
        for begin in range(end,stop,2):
            groups=[[],[]]
            for d in (begin,begin+1):
                raw=clock.book.rawday(d);t=np.arange(d%4,len(raw),4)
                f=live.changes(raw,t,net,ids,typ,levels_only=True)
                if d==begin:
                    st=t[t<427]
                    np.testing.assert_allclose(live.changes(raw[:427],st,net,ids,typ,levels_only=True),f[:len(st)],atol=3e-5,rtol=0)
                y=clock.labels(d).reshape(-1,50,3)[t]
                for j,ref in enumerate(incumbent.incumbent.REFS):
                    p=ref(d).reshape(-1,50,3)[t].copy()
                    old=clock.center(p[:,:,2]-clock.RATIO*p[:,:,1])
                    p[:,:,2]+=WEIGHT*(f[:,:,0]-old)
                    groups[j].append((p,y,WEIGHT*(f[:,:,1]-f[:,:,0])))
                days.append(base.chain.metric.gain(groups[0][-1:]))
            folds.append([base.chain.metric.gain(g) for g in groups]);print(begin+1,folds[-1],flush=True)
    good=all(sum(r[j]['y3']>0 for r in folds)>=3 and np.mean([r[j]['y3'] for r in folds])>0 and
             folds[-1][j]['y3']>0 for j in (0,1)) and sum(d['y3']>0 for d in days)>=5
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,comparison='fixed-sector branch',
        fraction=WEIGHT,selected=bool(good)),indent=2))
    if good:
        pkg=torch.load(incumbent.OUT/'models.pth',map_location='cpu',weights_only=True)
        assert pkg['chosen']==2
        state=pkg['states'][1]
        ids=state['bias'].argmax(1).numpy()
        torch.save(dict(state=state,prior=torch.tensor(live.prior(16,ids))),OUT/'model.pth')
    return good


def layer():
    parent=incumbent.incumbent.layer()
    pkg=torch.load(OUT/'model.pth',weights_only=True,map_location='cpu')
    state=pkg['state'];ids=state['bias'].argmax(1).numpy();typ=pkg['prior'].numpy()
    net=incumbent.Anchored(state['enc.0.weight'].shape[1],ids).cuda().eval();net.load_state_dict(state)
    def apply(raw,p):
        q=parent(raw,p)
        far=live.changes(raw,np.arange(len(q)),net,ids,typ,levels_only=True)[:,:,1]
        old=clock.center(q[:,:,2]-clock.RATIO*q[:,:,1])
        q[:,:,2]+=WEIGHT*(far-old)
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent=ancestor.load();new=layer()
    return lambda raw,p:new(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():base.chain.replay.verify(OUT,layer,incumbent.OUT)
