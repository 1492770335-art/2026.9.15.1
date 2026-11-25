"""Independently reconstruct the selected activity correction from its nine size and direction cells."""
import json
import numpy as np
import torch
import experiment_y3_impact_mass_forecast_v1 as model


@torch.inference_mode()
def run():
    st=dict(np.load(model.OUT/'model.npz'))
    pick=int(st['pick'])
    assert pick==9
    w=st['w'][:,pick]
    assert np.count_nonzero(np.delete(w,[0,1,2,12]))==0
    raw=model.trial.clock.book.rawday(15)
    p=model.history.references(15,raw)[0]
    fn=model.correction()
    full=fn(raw,p)
    t=np.unique(np.r_[0,1,7,8,63,64,2400,2401,2402,np.arange(9,len(raw),67)])
    x,_,_=model.context(raw,t,st,model.event.restore(st))
    net=model.restore(st)[1]
    _,gross,sign=net(torch.from_numpy(x))
    # Sum conditional magnitude minus its training mean times predicted sign for each window and class.
    effect=np.zeros((len(t),50),dtype=float)
    for h in range(3):
        for j in range(3):
            effect+=(gross[:,:,h,j].numpy()-st['typical'][h,j])*sign[:,:,h,j].numpy()
    effect=model.trial.clock.center(effect*st['scale'])
    direct=model.trial.clock.center(p[t])@w[:3]+effect*w[12]
    diff=float(abs(direct-full[t]).max())
    stops=(1,8,64,311,2401,2402)
    prefix=max(float(abs(fn(raw[:n],p[:n])-full[:n]).max()) for n in stops)
    assert max(diff,prefix)<4e-6
    record=dict(day=16,stock_ticks=len(t)*50,direct_error=diff,prefixes=len(stops),prefix_error=prefix)
    (model.OUT/'inference_verification.json').write_text(json.dumps(record,indent=2))
    print(record,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
