"""Test on training dates whether receiver-specific predicted flow magnitude adds to the imbalance term."""
import json

import numpy as np

import y3_panel_message_origin_references as ref

model,base=ref.incumbent,ref.base
ARMS=('unchanged','current imbalance','common flow magnitude','reassigned stock magnitude',
      'own estimated magnitude','own minus reassigned','signed magnitude without imbalance','raw signed flow')


def signals(raw,st):
    lower,upper=st['lim']
    imbalance=(2*(raw[:,:,21].astype(float)-lower)/(upper-lower)-1).clip(-1,1)
    flow=np.zeros(raw.shape[:2],np.float32)
    w=st['w'][0,:,7]-st['w'][1,:,7]
    for lo,hi in base.segments(len(raw)):
        a,_=model.inputs(base.raw_returns(raw[lo:hi]),imbalance[lo:hi])
        for order in st['orders']:
            for i in range(0,len(a),144):
                end=min(i+144,len(a))
                x=model.features(a[i:end,order])
                flow[lo+i:lo+end,order]+=(x@w)/len(st['orders'])
    def gate(x):
        return x*np.tanh(2*x*imbalance)
    value=gate(flow)
    np.testing.assert_array_equal(value,gate(-flow))
    market=gate(np.sqrt((flow**2).mean(1,keepdims=True)))
    wrong=gate(np.roll(flow,13,axis=1))
    z=np.stack((imbalance,market,wrong,value,value-wrong,np.sign(flow)*abs(value),flow),2)
    return (z-z.mean(1,keepdims=True)).astype('float32')


def main():
    with np.load(model.OUT/'model.npz') as f: st=dict(f)
    rows,folds=[],[]
    for d in range(4,16):
        raw=ref.trial.clock.book.rawday(d)
        if d in (8,12):
            w=base.calibrate(rows);fold=[]
        z=signals(raw,st)
        if d==4:
            direct=model.design(raw[:197],st['w'],st['orders'],st['lim'])[:,:,int(st['pick'])]
            np.testing.assert_allclose(z[:197,:,3],direct,rtol=0,atol=6e-7)
        y=ref.trial.clock.labels(d).reshape(-1,50,3)
        ps=model.warm.read(d,raw) if d<8 else ref.previous.references(d,raw)
        block=[base.records(p,y,z) for p in ps];rows.append(block)
        if d>=8:
            fold.append(block)
            if d in (11,15):
                folds.append([base.score(m,w) for m in np.sum(fold,0)])
    report=dict(arms=ARMS,folds=np.asarray(folds).tolist(),magnitude_invariance_exact=True,
                validation_days=[[9,12],[13,16]],paired_control_stock_shift=13)
    (model.OUT/'flow_magnitude_audit.json').write_text(json.dumps(report,indent=2))
    print(np.asarray(folds)[:,:,0].round(7),flush=True)


if __name__=='__main__':
    main()
