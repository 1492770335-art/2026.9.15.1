"""Transfer the part of the remaining continuous fields that the existing market state cannot reconstruct."""
import numpy as np
import experiment_dense_conditional_lag_memory_v1 as memory

ref = memory.ref
parent = ref.market.parent
OUT = ref.base.ROOT / "artifacts/dense_complementary_book_fields_v1"
ARMS = ("missing_to_core", "core_to_missing", "missing_to_missing", "combined")
CORE = (7,10,11,14,21,23,25,32,39,40,47,50,60,64,67,72,74,75,76)
COLUMNS = np.setdiff1d(np.arange(77),CORE)


def book_inputs(raw,grid,m,weights):
    chunks = []
    for a,b in parent.sessions(grid):
        source = raw[a:b][:,:,COLUMNS].astype(float)
        observed = np.tanh(.65*np.arcsinh(source)).mean(1)
        chunks.append(observed-parent.smooth(observed,320))
    return np.column_stack((m,np.concatenate(chunks)))


def collect():
    OUT.mkdir(exist_ok=True)
    code = parent.codec()
    for d in range(16):
        load = lambda k: np.load(ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y = load("x"),load("time").astype(int),load("y")
        x,av = parent.state(raw,grid,code,with_market=True)
        m = book_inputs(raw,grid,ref.market.market_state(av,grid),{})
        n = m.shape[1]
        cov = np.zeros((2,3,n,n))
        pay = np.zeros((2,n))
        variance = np.zeros(2)
        count = 0
        for a,b in parent.sessions(grid):
            for t in np.array_split(np.arange(a,b-60,8),4):
                if not len(t):continue
                def slice_stock(time):
                    new = np.tanh(.65*np.arcsinh(raw[time][:,:,COLUMNS].astype(float)))
                    new -= new.mean(1,keepdims=True)
                    return np.concatenate((x[time],new),axis=2).reshape(-1,n)
                current = slice_stock(t)
                avg = y[t,:,1].mean(1,dtype=float)
                rel = (y[t,:,1]-avg[:,None]).ravel()
                pay[0] += current.T@rel/50
                pay[1] += m[t].T@avg
                variance += [rel@rel/50,avg@avg]
                for j,lag in enumerate((0,20,40)):
                    later = slice_stock(t+lag) if lag else current
                    cov[0,j] += current.T@later/50
                    cov[1,j] += m[t].T@m[t+lag]
                count += len(t)
        np.savez(OUT/f"moments_{d:02d}.npz",cov=cov/count,pay=pay/count,variance=variance/count)


def whiten(covariance):
    eigen, vec = np.linalg.eigh(.5*(covariance+covariance.T))
    return vec/np.sqrt(eigen.clip(0)+.22)


def train(history):
    moments = []
    for d in range(len(history)):
        with np.load(OUT/f"moments_{d:02d}.npz") as f:
            moments.append((f["cov"],f["pay"],f["variance"]))
    grams,payoff,variance = (np.stack(x) for x in zip(*moments))
    st,mk = grams[:,0],grams[:,1]
    ss = np.sqrt(st[:,0].mean(0).diagonal()).clip(.03)
    sd = np.sqrt(mk[:,0].mean(0).diagonal()).clip(.006)
    core = len(sd)-len(COLUMNS)
    cov = .5*(st/np.outer(ss,ss)+mk/np.outer(sd,sd))
    pay = .5*(payoff[:,0]/ss*np.sqrt(variance[:,1].mean()/variance[:,0].mean())+payoff[:,1]/sd)
    banks = []
    for fold in (0,1):
        c,k20,k40 = cov[fold::2].mean(0)
        # Estimate on training dates how much of a new field is predictable from the original state.
        projection = np.eye(len(c))
        projection[:core,core:] = -np.linalg.solve(c[:core,:core]+.10*np.eye(core), c[:core,core:])
        quality = projection.T@c@projection
        projection[:,:core] = projection[:,:core]@whiten(quality[:core,:core])
        projection[:,core:] = projection[:,core:]@whiten(quality[core:,core:])
        trans = projection.T@(k20+k40)@projection
        payoff = projection.T@pay[1-fold::2].mean(0)
        # Separate learning new signals today from forecasting omitted fields later in the interval.
        left = projection[:,core:]@trans[core:,:core]@payoff[:core]
        right = projection[:,:core]@trans[:core,core:]@payoff[core:]
        middle = projection[:,core:]@(.5*(trans[core:,core:]+trans[core:,core:].T))@payoff[core:]
        banks.append(np.column_stack((left,right,middle,left+right+middle))/sd[:,None])
    return dict(bank=np.mean(banks,0))


def views(state,grid,parameters):
    return state@parameters["bank"]


if __name__ == "__main__":
    collect()
    memory.runner.run(OUT,ARMS,train,views,prior_head=memory,prepare_inputs=book_inputs,keep_oof=True)
