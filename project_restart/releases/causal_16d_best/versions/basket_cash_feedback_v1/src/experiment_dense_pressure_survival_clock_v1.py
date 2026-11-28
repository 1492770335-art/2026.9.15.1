"""Measure a pressure spell in accumulated conditional exit risk, using forecasts of next-tick sign changes."""
import json
import numpy as np

import experiment_dense_pressure_onset_order_v1 as onset

rel, REFERENCE = onset.rel, onset.REFERENCE
OUT = REFERENCE.parent / "dense_pressure_survival_clock_v1"
ARMS = ("surprise_given_elapsed_time", "path_above_current_hazard", "survival_and_price_response")
DIMS = (50,5,2,4,4,3)
N = int(np.prod(DIMS))


def keys(v,s,start):
    age = np.arange(len(v))[:,None,None]-start
    change = np.concatenate((np.zeros_like(v[:1]),np.diff(v,axis=0)))
    ix = np.ravel_multi_index(np.broadcast_arrays(np.arange(50)[None,:,None],np.arange(5)[None,None,:],
                             (s>0).astype(int),np.digitize(age,[3,12,36]),np.digitize(s*v,[0,.2,.55]),
                             np.digitize(s*change,[-.035,.035])),DIMS)
    return age,ix


def rates(end):
    rows = sum(np.load(OUT/f"risk_{d:02d}.npy") for d in range(end)).reshape(*DIMS,2)
    hit,total = rows[...,0],rows[...,1]
    def ratio(h,n):
        return (h+.5)/(n+1)
    marginal = ratio(hit.sum((4,5),keepdims=True),total.sum((4,5),keepdims=True))
    pooled = ratio(hit.sum(0,keepdims=True),total.sum(0,keepdims=True))
    estimate = (hit+32*pooled+8*marginal)/(total+40)
    return np.stack(np.broadcast_arrays(estimate,marginal),axis=-1).reshape(N,2).clip(.0001,.99)


def risk_row(v,s,start,visible):
    _,key = keys(v,s,start)
    valid = (s[:-1]!=0)&(start[:-1]>0)&visible[:-1,:,None]&visible[1:,:,None]
    event = (s[1:]!=s[:-1])[valid]
    ix = key[:-1][valid]
    return np.column_stack((np.bincount(ix,event,minlength=N),np.bincount(ix,minlength=N)))


def state(v,s,start,level,visible,rates_table):
    age,idx = keys(v,s,start)
    risk = rates_table[idx]
    hazard = -np.log1p(-risk)*visible[:,:,None,None]
    integrated = np.concatenate((np.zeros_like(hazard[:1]),hazard.cumsum(0)[:-1]))
    clock = integrated-integrated[start,np.arange(50)[None,:,None],np.arange(5)[None,None,:]]
    movement = (level[:,:,None]-level[start,np.arange(50)[None,:,None]])/np.sqrt(1+age)
    # Compare risk accumulated on the realized path with age-only and frozen-current-rate clocks.
    late = np.log1p(clock[...,0])-np.log1p(clock[...,1])
    trajectory = np.log1p(clock[...,0])-np.log1p(hazard[...,0]*age)
    gaps = np.vstack((np.zeros((1,50),int),(~visible).cumsum(0)))
    use = (start>0)&(gaps[1:,:,None]==gaps[start,np.arange(50)[None,:,None]])
    return age,np.tanh(late)*use,np.tanh(trajectory)*use,np.tanh(movement)


def features(t,v,s,q,age,late,trajectory,move):
    v,s,q,age,late,trajectory,move = [z[t] for z in (v,s,q,age,late,trajectory,move)]
    weight = np.tanh(2*abs(v))
    base = np.concatenate((q,v,s,s*weight*age/(10+age),move),axis=2)
    def marks(z):
        return np.concatenate((s*z,s*z*weight,s*z*np.tanh(s*q[:,:,2,None])),axis=-1)
    interaction = np.concatenate((move*late,move*trajectory,s*abs(move)*late),axis=-1)
    x = np.concatenate((base,marks(late),marks(trajectory),interaction),axis=2)
    return x-x.mean(1,keepdims=True)


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    audit = []
    with np.load(onset.settled.GEO) as geom:
        missing = geom["row"]
    for d in range(16):
        raw,grid = [onset.settled.load(d,k) for k in ("x","time")]
        if d>=4:
            p = (np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6
                 else onset.settled.head.ranks.band.pairs.net.future.forecast(d))
            q = p-p.mean(1,keepdims=True,dtype=float)
            y = onset.settled.load(d,"y")
            target = y-y.mean(1,keepdims=True,dtype=float)
            slope = ((q*target).sum((0,1))/(q*q).sum((0,1))).clip(.2,4)
            error = target/slope-q
            table = rates(d)
            gram = cross = n = 0
        transitions = 0
        for a,b,v,s,start,level in onset.sessions(raw,grid,code):
            visible = ~np.all(raw[a:b]==missing,axis=-1)
            transitions += risk_row(v,s,start,visible)
            if d<4:
                continue
            state_values = state(v,s,start,level,visible,table)
            for t in np.array_split(np.arange(20+d%5,b-a-60,5),6):
                z = features(t,v,s,q[a:b],*state_values)
                x = z.reshape(-1,z.shape[-1])
                gram += x.T@x
                cross += x.T@error[a+t].reshape(-1,3)
                n += len(x)
        np.save(OUT/f"risk_{d:02d}.npy",transitions)
        if d>=4:
            np.savez(OUT/f"day_{d:02d}.npz",g=gram/n,r=cross/n)
            yes,total = transitions.T
            loss = -(yes[:,None]*np.log(table)+(total-yes)[:,None]*np.log1p(-table)).sum(0)/total.sum()
            audit.append(dict(day=d+1,conditional_exit_log_loss=loss[0],age_only_log_loss=loss[1]))
    (OUT/"exit_forecasts.json").write_text(json.dumps(audit,indent=2))


def train(end):
    w = onset.fit(end,OUT,((23,38),(38,53),(53,68)))
    np.savez(OUT/f"fit_{end}.npz",w=w,table=rates(end))
    return np.array([end])


def design(raw,grid,code,p,bank):
    with np.load(onset.settled.GEO) as f:
        missing = f["row"]
    with np.load(OUT/f"fit_{int(bank[0])}.npz") as f:
        w,table = f["w"],f["table"]
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    q = p-p.mean(1,keepdims=True,dtype=float)
    for a,b,v,s,start,level in onset.sessions(raw,grid,code):
        visible = ~np.all(raw[a:b]==missing,axis=-1)
        state_values = state(v,s,start,level,visible,table)
        for ix in np.array_split(np.arange(b-a),max(1,(b-a+127)//128)):
            x = features(ix,v,s,q[a:b],*state_values)
            out[a+ix] = (x@w.reshape(len(w),-1)).reshape(len(ix),50,len(ARMS),3)
    return out


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[:,:,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
