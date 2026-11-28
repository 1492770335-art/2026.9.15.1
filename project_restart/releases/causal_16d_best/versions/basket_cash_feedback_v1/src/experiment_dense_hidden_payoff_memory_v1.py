"""Find delayed return information in a historical book that is absent from the forecast of the future book."""
import numpy as np
import experiment_dense_conditional_lag_memory_v1 as memory

ref = memory.ref
parent = ref.market.parent
OUT = ref.base.ROOT / "artifacts/dense_hidden_payoff_memory_v1"
ARMS = ("direct_delay", "unexplained_payoff", "replicated_gap")


def collect():
    OUT.mkdir(exist_ok=True)
    code = parent.codec()
    for day in range(16):
        load = lambda key: np.load(ref.base.CACHE/f"day_{day:02d}_{key}.npy", mmap_mode="r")
        raw, grid, y = load("x"), load("time").astype(int), load("y")[...,1]
        x,av = parent.state(raw,grid,code,with_market=True)
        m = ref.market.market_state(av,grid)
        moments = np.zeros((2,2,2,m.shape[1]))
        count = 0
        for a,b in parent.sessions(grid):
            t = np.arange(a+40,b-61,6)
            if not len(t):continue
            for j,lag in enumerate((20,40)):
                label = y[t+lag].astype(float)
                common = label.mean(1)
                rel = (label-common[:,None]).ravel()
                for i,idx in enumerate((t,t-lag)):
                    moments[0,j,i] += x[idx].reshape(-1,m.shape[1]).T@rel / 50
                    moments[1,j,i] += m[idx].T@common
            count += len(t)
        np.save(OUT/f"payoffs_{day:02d}.npy",moments/count)


def train(history):
    st,mk,sy,my,sv,mv = [np.stack(x) for x in zip(*history)]
    response = np.stack([np.load(OUT/f"payoffs_{d:02d}.npy") for d in range(len(history))])
    sd = np.sqrt(mk[:,0].mean(0).diagonal()).clip(.006)
    ss = np.sqrt(st[:,0].mean(0).diagonal()).clip(.03)
    scale = np.sqrt(mv[:,2].mean()/sv[:,2].mean())
    cov = .5*(st/np.outer(ss,ss)+mk/np.outer(sd,sd))
    pay = .5*(scale*sy[:,:,2]/ss+my[:,:,2]/sd)
    # Standardize the observed label in the same physical units as the frozen payoff decoder.
    direct = .5*ref.base.UNITS[1]*(scale*response[:,0]/ss+response[:,1]/sd)
    results = []
    for fold in (0,1):
        stats = cov[fold::2].mean(0)
        c,eye = stats[0],np.eye(len(sd))
        opposite = slice(1-fold,None,2)
        payoff = np.linalg.solve(cov[opposite,0].mean(0)+.22*eye,pay[opposite].mean(0))
        coeff = []
        for j,i in enumerate((3,4)):
            k,twice = stats[i],stats[i+1]
            q = c+.06*eye
            back = np.linalg.solve(q,k.T)
            fwd = np.linalg.solve(q,k)
            innovation = c-k@back-back.T@k.T+back.T@c@back
            conditional = twice-k@fwd-back.T@k+back.T@c@fwd
            book = conditional@payoff
            responses = direct[opposite,j]
            rhs = []
            for current,past in responses:
                target_now = np.linalg.solve(q,current)
                rhs.append(past-k@target_now-back.T@current+back.T@c@target_now)
            rhs = np.stack(rhs)
            gap = rhs-book
            avg = gap.mean(0)
            noise = gap.std(0)/np.sqrt(len(gap))
            repeated = avg*np.maximum(0, 1-(noise/(np.abs(avg)+1e-9))**2)
            gain = np.linalg.solve(innovation+.25*eye,np.stack((rhs.mean(0),avg,repeated),1))
            coeff.append(np.stack((-back@gain/sd[:,None],gain/sd[:,None])))
        results.append(coeff)
    return dict(bank=np.mean(results,0))


def views(state,grid,model):
    out = np.zeros((len(grid),len(ARMS)))
    for a,b in parent.sessions(grid):
        for j,lag in enumerate((20,40)):
            t = np.arange(a+lag,b)
            now,old = model["bank"][j]
            out[t] += state[t]@now+state[t-lag]@old
    return out


if __name__ == "__main__":
    collect()
    memory.runner.run(OUT,ARMS,train,views,prior_head=memory,keep_oof=True)
