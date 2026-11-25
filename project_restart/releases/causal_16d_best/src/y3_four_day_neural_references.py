"""Chronological early reference forecasts from a return-and-book network and an event GRU, fitted on days 1–4."""
import hashlib
import json

import numpy as np
import torch

import experiment_quote_event_gru_v1 as gru
import experiment_phase_future_book_v1 as phase
import y3_cross_market_renewal_references as recent

OUT=recent.incumbent.OUT.parent/'y3_four_day_neural_references_v1'
digest=recent.digest


def book_checkpoint(end):
    if end!=4:
        return phase.OUT/f'net_{end}.pth'
    existing=OUT.parent/'y3_booknet_quote_rollforward_v1'/'net_4.pth'
    return existing if existing.is_file() else OUT/'net_4.pth'


def book_model(end):
    checkpoint=book_checkpoint(end)
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        phase.train(end,out=checkpoint.parent)
    state=torch.load(checkpoint,weights_only=True,map_location='cpu')
    model=phase.BookNet(state['current.0.weight'].shape[1])
    model.load_state_dict(state)
    return model.cuda().eval()


def read(d,raw):
    if d>=8:
        return recent.references(d,raw)
    assert 4<=d<8
    with np.load(OUT/f'day_{d}.npz') as f:
        p=f['pred']
        assert int(f['cutoff'])==4 and str(f['raw'])==digest(raw) and str(f['prediction'])==digest(p)
    return p


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    a=book_model(4)
    checkpoint=gru.OUT/'net_4.pth'
    if not checkpoint.exists():
        gru.OUT.mkdir(exist_ok=True)
        gru.train(4)
    b=gru.EventNet(109).cuda().eval()
    b.load_state_dict(torch.load(checkpoint,weights_only=True))
    def infer(raw):
        return np.stack((phase.infer(a,raw),gru.infer(b,raw)))
    report={}
    for d in range(4,8):
        raw=recent.trial.clock.book.rawday(d)
        p=infer(raw)
        error=float(abs(p[:,:173]-infer(raw[:173])).max())
        assert error<1e-4 and np.isfinite(p).all()
        np.savez(OUT/f'day_{d}.npz',cutoff=4,pred=p,raw=digest(raw),prediction=digest(p))
        np.testing.assert_array_equal(p,read(d,raw))
        truth=recent.trial.clock.labels(d).reshape(-1,3)
        report[d+1]=dict(prefix_error=error,Y3_ic=[float(np.corrcoef(x[:,:,2].ravel(),truth[:,2])[0,1]) for x in p])
    (OUT/'verified.json').write_text(json.dumps(dict(training_days=[1,2,3,4],
        checksums=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [book_checkpoint(4),checkpoint]],
        days=report),indent=2))
    print(report)


if __name__=='__main__':
    main()
