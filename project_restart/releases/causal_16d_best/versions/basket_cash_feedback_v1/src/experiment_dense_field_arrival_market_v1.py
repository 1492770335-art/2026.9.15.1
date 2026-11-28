"""Resolve market book updates into fields that appear earlier and later across the 50 names."""
import numpy as np
import experiment_dense_market_surprise_transfer_v1 as transfer

rel, ref, fine = transfer.rel, transfer.ref, transfer.fine
OUT = ref.base.ROOT / "artifacts/dense_field_arrival_market_v1"
ARMS = ("early_reporters", "delayed_consensus", "timing_contrast")
COLUMNS = (7,10,11,14,21,23,25,32,39,40,47,50,60,64,67,72,74,75,76)
LAG = 8


def book(raw):
    return np.tanh(.65*np.arcsinh(np.take(raw, COLUMNS, axis=2).astype(float)))


def collect():
    OUT.mkdir(exist_ok=True)
    for day in range(16):
        def load(k):
            return np.load(ref.base.CACHE/f"day_{day:02d}_{k}.npy",mmap_mode="r")
        grid = load("time").astype(int)
        y = load("y").mean(1,dtype=float)
        direction = power = payout = 0.
        n = 0
        for a,b in rel.fields.parent.sessions(grid):
            x = book(load("x")[a:b])
            x -= rel.fields.parent.smooth(x,96)
            peers = (x.sum(1,keepdims=True)-x)/49
            # A field leads when its current innovation better predicts the other names' next change than the reverse.
            earlier, later = x[:-LAG:4], x[LAG::4]
            peer_early, peer_late = peers[:-LAG:4], peers[LAG::4]
            direction += (earlier*peer_late-later*peer_early).sum(0)
            power += np.stack(((earlier**2+later**2).sum(0)/2,
                               (peer_early**2+peer_late**2).sum(0)/2))
            t = np.arange(LAG,b-a-60,4)
            cash = y[a+t,2]-ref.base.UNITS[1]*y[a+t,1]
            cash -= cash.mean()
            payout += np.stack((np.einsum("t,tsf->sf",cash,x[t]),np.einsum("t,tsf->sf",cash,x[t-LAG])))
            n += len(t)
        np.savez(OUT/f"day_{day:02d}.npz", timing=direction/np.sqrt(np.prod(power,axis=0)).clip(1e-9), payout=payout/n)


def train(history):
    records = []
    for day in range(len(history)):
        with np.load(OUT/f"day_{day:02d}.npz") as f:
            records.append((f["timing"],f["payout"]))
    timing, payoff = map(np.stack,zip(*records))
    banks=[]
    for i in (0,1):
        sample = timing[i::2]
        # Compare senders within the same field to remove market-wide persistence.
        t = sample - sample.mean(1,keepdims=True)
        avg, noise = t.mean(0),t.std(0)/np.sqrt(len(t))
        weights = np.tanh(avg/(noise+.012))
        weights -= weights.mean(0,keepdims=True)
        def bank(w, delayed=False):
            labels=payoff[1-i::2]
            slope=((labels[:,0] if not delayed else labels[:,0]-labels[:,1])*w).mean(1)
            mu=slope.mean(0)
            mu=mu*abs(mu)/(abs(mu)+slope.std(0)/np.sqrt(len(slope))+.0001)
            factor=w*mu
            return factor/np.sqrt((factor**2).sum()).clip(1e-9)
        early=np.maximum(weights,0);late=np.maximum(-weights,0)
        early-=early.mean(0);late-=late.mean(0)
        banks.append(np.stack((bank(early),bank(late,True),bank(weights,True))))
    return dict(bank=np.mean(banks,0))


def prepare_inputs(raw,grid,m,params):
    result = np.zeros((len(raw),3))
    for a,b in rel.fields.parent.sessions(grid):
        x=book(raw[a:b])
        x-=rel.fields.parent.smooth(x,96)
        now=np.einsum("tif,kif->tk",x,params["bank"],optimize=False)
        result[a:b,0]=now[:,0]
        if b-a>LAG:
            result[a+LAG:b,1:]=now[LAG:,1:]-now[:-LAG,1:]
    return result


def views(state,grid,params):
    return state


if __name__=="__main__":
    collect()
    fine.memory.runner.run(OUT,ARMS,train,views,targets=(2,),prepare_inputs=prepare_inputs,
                           prior_head=(fine.memory,fine.news,fine.path,fine),keep_oof=True,check_leave_one_out=True)
