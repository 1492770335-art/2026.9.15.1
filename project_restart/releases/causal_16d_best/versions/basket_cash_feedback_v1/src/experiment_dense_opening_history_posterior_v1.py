"""Update the inferred hidden quote history as each new opening observation arrives."""
import numpy as np

import experiment_dense_reconstructed_opening_history_v1 as init

book,onset,rel,REFERENCE = init.book,init.onset,init.rel,init.REFERENCE
OUT = REFERENCE.parent / "dense_opening_history_posterior_v1"
ARMS = ("conditional_history","replicated_history","revision_since_first_frame")
WINDOWS = (5,24,120)
F = len(init.F)
N = 1+4*F


def design_row(first,current,seen,elapsed):
    return np.concatenate((np.ones((*first.shape[:2],1)),first,elapsed*first,
                           current-first,seen-first),axis=2)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        grams = np.zeros((3,N+F,N+F))
        counts = np.zeros(3)
        for a,b in rel.fields.parent.sessions(grid):
            field = np.arcsinh(raw[a:b][:,:,init.F].astype(np.float32))
            total = np.concatenate((np.zeros_like(field[:1]),field.cumsum(0,dtype=float)))
            for j,h in enumerate(WINDOWS):
                cut = np.arange(h+3+d%19,b-a-h,29)
                for elapsed in np.unique((np.array([0,.13,.33,.57,.85])*(h-1)).astype(int)):
                    used = elapsed+1
                    seen = (total[cut+used]-total[cut])/used
                    invisible = (total[cut]-total[cut-(h-used)])/(h-used)
                    x = design_row(field[cut],field[cut+elapsed],seen,used/h)
                    row = np.concatenate((x,invisible-seen),axis=2).reshape(-1,N+F)
                    grams[j] += row.T@row
                    counts[j] += len(row)
        np.save(OUT/f"gram_{d:02d}.npy",grams/counts[:,None,None])


def solve(g,indices):
    cov = g[np.ix_(indices,indices)]
    pred = np.zeros((N,F))
    pred[indices] = np.linalg.solve(cov+.09*np.diag(cov.diagonal().clip(.003)),g[indices,N:])
    return pred


def train(end):
    all_g = np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(end)])
    result = np.zeros((3,3,N,F))
    indices = np.arange(N)
    for j in range(3):
        rows = all_g[:,j]
        g = rows.mean(0)
        w = solve(g,indices)
        fractions = []
        for parity in (0,1):
            trained = solve(rows[parity::2].mean(0),indices)
            for test in rows[1-parity::2]:
                cross = (trained*test[:N,N:]).sum(0)
                var = (trained*(test[:N,:N]@trained)).sum(0)
                fractions.append((2*cross-var)/test.diagonal()[N:].clip(1e-5))
        # Weight fields by recovery of unseen history on independent dates.
        reliable = np.quantile(fractions,.25,axis=0).clip(0,1)
        prior = solve(g,np.arange(1+2*F))
        result[j] = np.stack((w,w*reliable,prior))
    return result


def design(raw,grid,code,p,bank):
    out = np.zeros((*p.shape[:2],3,3))
    data = raw[:min(len(raw),120)]
    t = np.arange(len(data))
    quotes = np.arcsinh(data[:,:,init.F].astype(np.float32))
    total = quotes.cumsum(0,dtype=float)
    x,vol = book.phase.spline.features(data,t)
    copies = [x.copy() for _ in range(3)]
    for j,h in enumerate(WINDOWS):
        live = t[t+1<h]
        used = (live+1)[:,None,None]
        seen = total[live]/used
        first = np.broadcast_to(quotes[:1],seen.shape)
        z = design_row(first,quotes[live],seen,used/h)
        sl = slice(100+j*F,100+(j+1)*F)
        for arm,c in enumerate(copies):
            shift = (z@bank[j,arm])*((h-used)/h)
            c[live,:,sl] = (x[live,:,sl]-shift).clip(-10,10)
    baseline = init.response(x,vol,t)
    conditional,replicable,prior = [init.response(c,vol,t) for c in copies]
    for j,delta in enumerate((conditional-baseline,replicable-baseline,conditional-prior)):
        out[t,:,j] = delta-delta.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
