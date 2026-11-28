"""Select market inputs by their across-date association with future-only returns and a clock-shift null."""
import sys
import json
import numpy as np
import experiment_dense_replicated_market_cash_shapes_v1 as parent

strip,rel,onset,REFERENCE,read_p=parent.strip,parent.rel,parent.onset,parent.REFERENCE,parent.read_p
cash=parent.cash
OUT=REFERENCE.parent/'dense_market_field_screen_v1'
ARMS=('best_eight','different_fields','clock_shift_threshold')
C=parent.C
W=parent.W
F=W-C
B=255


def field_name(i):
    group,within=divmod(int(i),cash.study.W)
    kind,field=divmod(within,len(cash.study.clock.memory.FIELDS))
    return (('current','last_16','previous_48','previous_128')[group],
            ('mean','spread','skew')[kind],int(cash.study.clock.memory.FIELDS[field]))


def project(g):
    v=np.eye(W)[:,C:]
    v[:C]=-np.linalg.solve(g[:C,:C]+.0001*np.diag(g.diagonal()[:C].clip(1e-7)),g[:C,C:W])
    return v


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        x,legs,_,t=cash.records(d)
        p=read_p(d)
        raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
        f=parent.fields(x,p[t],parent.past_returns(raw,grid)[t])
        y=np.column_stack((legs[:,1:4].sum(1),legs[:,4:].sum(1)))
        f-=f.mean(0);y-=y.mean(0)
        controls=f[:,:C]
        # Day-specific nuisance fits are used only for scoring the association within each historical date.
        xx=f[:,C:]-controls@np.linalg.lstsq(controls,f[:,C:],rcond=None)[0]
        yy=y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
        xx/=np.sqrt(np.sum(xx**2,axis=0)).clip(1e-6)
        yy/=np.sqrt(np.sum(yy**2,axis=0)).clip(1e-6)
        random=np.random.default_rng(40886+d)
        moved=np.zeros((len(t),2,B))
        for a,b in rel.fields.parent.sessions(grid):
            loc=np.where((t>=a)&(t<b))[0]
            shifts=random.integers(12,len(loc)-11,size=B)
            perm=loc[(np.arange(len(loc))[:,None]+shifts)%len(loc)]
            moved[loc]=yy[perm].transpose(0,2,1)
        null=(xx.T@moved.reshape(len(t),-1)).reshape(F,2,B)
        np.savez(OUT/f'day_{d:02d}.npz',corr=xx.T@yy,null=null.astype(np.float32))


def screens(end):
    stats=[];null=[]
    for d in range(4,end):
        with np.load(OUT/f'day_{d:02d}.npz') as f:
            stats.append(f['corr']);null.append(f['null'])
    raw=np.stack(stats);noise=np.stack(null)
    def repeat(data):
        m=data.mean(0)
        # Penalize directions whose estimated effect flips across dates.
        return m**3/(m*m+data.var(0)/len(data)+1e-7)
    score=repeat(raw);control=repeat(noise)
    cutoff=np.quantile(np.max(np.abs(control),axis=0),.90,axis=1)
    selected={}
    for h in range(2):
        ids=np.argsort(-abs(score[:,h]))
        groups=[];seen=set()
        for i in ids:
            k=field_name(i)[2]
            if k not in seen:
                groups.append(i);seen.add(k)
            if len(groups)==8:break
        selected[h]=(ids[:8],np.array(groups),ids[abs(score[ids,h])>cutoff[h]][:16])
    return score,cutoff,selected


def train(end):
    rows=[];means=[];betas=[]
    for d in range(4,end):
        with np.load(parent.OUT/f'day_{d:02d}.npz') as f:
            rows.append(f['gram']);means.append(f['mean']);betas.append(f['beta'])
    score,cutoff,selected=screens(end)
    rows=np.stack(rows);g=rows.mean(0)
    v=project(g)
    _,coef,_=strip.physical()
    unit=np.mean(betas,0)*coef[2]/coef
    w=np.zeros((W,3,2))
    for h,(l,r) in enumerate(((1,4),(4,12))):
        for j,ids in enumerate(selected[h]):
            if not len(ids):continue
            ax=v[:,ids]
            cov=ax.T@g[:W,:W]@ax
            rhs=(ax.T@rows[:,:W,W+l:W+r].sum(-1)[...,None])[...,0]/unit[h+1]
            m=rhs.mean(0)
            m=m**3/(m*m+rhs.var(0)/len(rhs)+1e-9)
            w[:,j,h]=ax@np.linalg.solve(cov+1.5*np.diag(cov.diagonal().clip(1e-7)),m)
    return np.r_[np.mean(means,axis=0),w.ravel()]


signals=parent.signals


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


def audit():
    result={}
    for end in (8,12):
        scores,cutoff,selected=screens(end)
        future=[]
        for day in range(end,end+4):
            with np.load(OUT/f'day_{day:02d}.npz') as f:
                future.append(f['corr'])
        out=np.mean(future,axis=0)
        ids=selected[1][1]
        result[end+1]=dict(null_cutoff=float(cutoff[1]),number_passing=int(np.sum(abs(scores[:,1])>cutoff[1])),
            selected_fields=[dict(field=field_name(i),training=scores[i,1],later_signed=out[i,1]*np.sign(scores[i,1])) for i in ids],
            matched=np.mean((out[ids,1]*scores[ids,1])>0),mean_signed=np.mean(out[ids,1]*np.sign(scores[ids,1])))
    (OUT/'field_audit.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=='__main__':
    collect()
    audit()
    strip.evaluation.rolling(sys.modules[__name__])
