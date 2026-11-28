"""Use already observed parts of a future field predictor and forecast only its unknown inputs."""
import numpy as np
import experiment_dense_unexpected_fields_v1 as news

fields,memory,ref = news.fields,news.memory,news.ref
OUT = ref.base.ROOT/"artifacts/dense_field_news_history_v1"
ARMS = ("known_context","conditional_forecast","joint_update")


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        load = lambda k: np.load(ref.base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid = load("x"),load("time").astype(int)
        x,av = fields.parent.state(raw,grid,fields.parent.codec(),with_market=True)
        m = fields.book_inputs(raw,grid,ref.market.market_state(av,grid),{})
        core = x.shape[-1]
        block = np.zeros((2,core,m.shape[-1]))
        count = 0
        for a,b in fields.parent.sessions(grid):
            t = np.arange(a,b-61,8)
            xt = x[t].reshape(-1,core).astype(float)
            z = np.tanh(.65*np.arcsinh(raw[t+60][:,:,fields.COLUMNS].astype(float)))
            z -= z.mean(1,keepdims=True)
            later = np.concatenate((x[t+60],z),axis=-1).reshape(-1,m.shape[-1])
            block[0] += xt.T@later/50
            block[1] += m[t,:core].T@m[t+60]
            count += len(t)
        np.save(OUT/f"lag60_{d:02d}.npy",block/count)


def train(history):
    data = []
    for d in range(len(history)):
        with np.load(fields.OUT/f"moments_{d:02d}.npz") as f:
            data.append((f["cov"],f["pay"],f["variance"],np.load(news.delayed.OUT/f"payoffs_{d:02d}.npy"),np.load(OUT/f"lag60_{d:02d}.npy")))
    grams,payoffs,var,delays,extra = (np.stack(x) for x in zip(*data))
    sd = np.sqrt(grams[:,1,0].mean(0).diagonal()).clip(.006)
    ss = np.sqrt(grams[:,0,0].mean(0).diagonal()).clip(.03)
    n = len(sd)-len(fields.COLUMNS)
    scale = np.sqrt(var[:,1].mean()/var[:,0].mean())
    cov = .5*(grams[:,0]/np.outer(ss,ss)+grams[:,1]/np.outer(sd,sd))
    pay = .5*(payoffs[:,0]*scale/ss+payoffs[:,1]/sd)
    past_pay = .5*(delays[:,0,:,0]*scale/ss[:n]+delays[:,1,:,0]/sd[:n])
    cross_late = .5*(extra[:,0]/np.outer(ss[:n],ss)+extra[:,1]/np.outer(sd[:n],sd))
    outputs = []
    for fold in (0,1):
        first = slice(fold,None,2);second = slice(1-fold,None,2)
        k = cov[first].mean(0);late = cross_late[first].mean(0)
        def cross(t,u):
            delta = u-t
            if delta==3:return late
            return (k[delta] if delta>=0 else k[-delta].T)[:n]
        origins = (0,-1,-2)
        gram = np.block([[cross(t,u)[:,:n] for u in origins] for t in origins])
        field = np.concatenate([cross(t,0)[:,n:] for t in origins])
        pred = np.linalg.solve(gram+.17*np.eye(len(gram)),field)
        reward = pay[second,n:].mean(0)-pred.T@np.r_[pay[second,:n].mean(0),past_pay[second].mean(0).ravel()]
        residual = k[0,n:,n:]-pred.T@field-field.T@pred+pred.T@gram@pred
        decoder = np.linalg.solve(residual+.29*np.eye(len(residual)),reward)
        coef = (pred@decoder).reshape(3,n)
        single = gram[:n,:n]+.22*np.eye(n)
        two = gram[:2*n,:2*n]+.22*np.eye(2*n)
        total = np.zeros((2*n,3))
        for h in (1,2):
            present_to_future = cross(0,h)[:,n:]@decoder
            delayed_to_future = cross(-1,h)[:,n:]@decoder
            unknown_cov = np.r_[present_to_future,delayed_to_future]
            known_change = np.zeros(2*n)
            for lag,readout in enumerate(coef):
                target = h-lag
                assumed = cross(0,target)[:,:n]@readout
                if target<=0:
                    # Exact observed book at t or t-20 supersedes its conditional estimate.
                    known_change[:n] += np.linalg.solve(single,assumed)
                    start = -target*n
                    known_change[start:start+n] -= readout
                else:
                    unknown_cov -= np.r_[assumed,cross(-1,target)[:,:n]@readout]
            older_information = np.linalg.solve(two,unknown_cov)
            older_information[:n] -= np.linalg.solve(single,unknown_cov[:n])
            total += np.column_stack((known_change,older_information,known_change+older_information))
        outputs.append(total.reshape(2,n,3)/sd[None,:n,None])
    return dict(bank=np.mean(outputs,0))


def views(m,grid,weights):
    out = np.zeros((len(grid),len(ARMS)))
    for a,b in fields.parent.sessions(grid):
        # Start at the same point as the deployed source head.
        t = np.arange(a+40,b)
        out[t] = m[t]@weights["bank"][0]+m[t-20]@weights["bank"][1]
    return out


if __name__ == "__main__":
    collect()
    memory.runner.run(OUT,ARMS,train,views,prior_head=(memory,news),keep_oof=True)
