"""Use other stocks' learned quote signals to predict the cross-sectional residual returns."""
import numpy as np

import experiment_dense_matched_book_readout_v1 as book
from prepare_dense_matched_book_reference_v1 import OUT as REFERENCE

onset,rel = book.onset,book.rel
OUT = REFERENCE.parent / "dense_directed_book_transfer_v1"
ARMS = ("all_other_stocks","date_consistent_edges","strongest_peer_edges")
N = 50*6


def signals(raw,ticks):
    x,_ = book.hidden(raw,ticks)
    w = book.encoder().idio.weight.detach().cpu().numpy()
    z = x @ w.T
    z /= np.sqrt((z*z).mean(1,keepdims=True)+.04)
    return z.reshape(len(ticks),N)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y = [onset.settled.load(d,k) for k in ("x","time","y")]
        p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6
             else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        current = p-p.mean(1,keepdims=True)
        label = y-y.mean(1,keepdims=True)
        scale = ((current*label).sum((0,1))/(current**2).sum((0,1))).clip(.3,3)
        err = label/scale-current
        g = np.zeros((N,N+150))
        count = 0
        for a,b in rel.fields.parent.sessions(grid):
            t = np.arange(a+d%6,b,6)
            x = signals(raw,t).astype(float)
            right = np.column_stack((x,err[t].reshape(-1,150)))
            # Remove persistent stock/session offsets when estimating transferable edges.
            right -= right.mean(0)
            g += right[:,:N].T @ right
            count += len(t)
        np.save(OUT/f"gram_{d:02d}.npy",g/count)


def train(end):
    grams = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    def solve(g):
        cov = g[:,:N]
        weights = np.linalg.solve(cov+1.5*np.diag(cov.diagonal().clip(.1)),g[:,N:])
        w = weights.reshape(50,6,50,3)
        # Remove the source stock's own readout from every candidate.
        w[np.arange(50),:,np.arange(50),:] = 0
        return w
    joint = solve(grams.mean(0))
    splits = [solve(group.mean(0)) for group in (grams[::2],grams[1::2])]
    agreement = (splits[0]*splits[1]).sum(1,keepdims=True)
    consistent = joint*(agreement>0)
    top = np.sort(agreement,axis=0)[-8]
    sparse = consistent*(agreement>=top)
    return np.stack([w.reshape(N,50,3) for w in (joint,consistent,sparse)],axis=2)


def design(raw,grid,code,p,bank):
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    for a,b in rel.fields.parent.sessions(grid):
        x = signals(raw,np.arange(a,b))
        pred = (x@bank.reshape(N,-1)).reshape(b-a,50,len(ARMS),3)
        out[a:b] = pred-pred.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
