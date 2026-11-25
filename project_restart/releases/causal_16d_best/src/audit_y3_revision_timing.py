"""Replace the selected price-change revision channel by causal timing controls on first-16-day folds."""
import json
import numpy as np
import torch
import experiment_y3_quiet_forecast_revisions_v1 as model

def revisions(raw,p,events):
    coef=model.trial.curve.settled.historical_geometry()[1][:,1]
    near=model.trial.clock.center(p[:,:,1])*coef[2]/coef[1]
    tail=model.trial.clock.center(p[:,:,2])-near
    signal=np.stack((near,tail),2)
    delta=np.diff(signal,axis=0,prepend=signal[:1])
    prefix=np.concatenate((np.zeros_like(delta[:1],dtype=float),(delta*events[:,:,None]).cumsum(0,dtype=float)))
    end=np.arange(len(raw))[:,None]+1;stock=np.arange(50)[None,:]
    outputs=[]
    for count in (3,12):
        start=model.boundaries(events,count)
        outputs.append(prefix[end,stock]-prefix[start,stock])
    return model.trial.clock.center(np.concatenate(outputs,2)).astype('float32')


def main():
    screen,trial,parent=model.screen,model.trial,model.parent
    pick=json.loads((model.OUT/'results.json').read_text())['selected']
    assert model.ARMS[pick]=='price-change revision'
    names=('actual','controls only','one tick late','eight ticks late','other stock')
    histories=[[] for _ in names];stats=[[],[]]
    props=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        event=raw[:,:,0]!=np.float16(.00136)
        props.append(event.mean(0))
        late=np.zeros_like(event);late[1:]=event[:-1]
        older=np.zeros_like(event);older[8:]=event[:-8]
        refs=parent.references(d,raw) if d>=8 else [screen.history.market.prior.path.early.load(d,raw)]
        if d>=8 and d%2==0:models=[model.fit(hist)[:,pick:pick+1] for hist in histories]
        records=[[] for _ in names]
        for j,p in enumerate(refs):
            z=model.features(raw,p)
            matched=revisions(raw,p,event)
            np.testing.assert_array_equal(matched,z[:,:,8:12])
            signals=[];refitted=[]
            for k,v in enumerate((matched,np.zeros_like(matched),revisions(raw,p,late),revisions(raw,p,older),revisions(raw,p,np.roll(event,17,axis=1)))):
                local=z.copy();local[:,:,8:12]=v
                records[k].append(screen.record(local,p,y,d))
                if d>=8:
                    signals.append(screen.changes(local,p,models[0])[0])
                    refitted.append(screen.changes(local,p,models[k])[0])
            if d>=8:
                stats[j].append(trial.scoring.audit.blockstats(p,y,signals+refitted).sum(0))
        for hist,rec in zip(histories,records):hist.append([np.mean(v,0) for v in zip(*rec)])
    rates=np.mean(props,0)
    result=dict(days=list(range(9,17)),price_change_rate=float(rates.mean()),across_stocks=np.quantile(rates,[0,.5,1]).tolist(),
                variants=names,columns=['frozen','refitted'],y3_gains=[
                    trial.scoring.audit.pooled(np.sum(s,0)).reshape(2,-1).tolist() for s in stats])
    (model.OUT/'training_timing_audit.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);main()
