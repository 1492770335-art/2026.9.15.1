"""Estimate how an intraday break changes the forecast for intervals crossing or following it."""
import json
import numpy as np
import experiment_delayed_tail_band_skill_v1 as early
import experiment_eight_date_expert_contrasts_v1 as readout
import experiment_pairwise_book_tree_v1 as book
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART, corr, labels, measure

PARENT=ART/'checkpoint_disagreement_v1'
OUT=ART/'session_restart_v1'


def restart():
    curves=[]
    for d in range(8):
        r=book.rawday(d)[:,:,0].astype(float)-.00136
        curves.append((r*r).mean(1)[1900:2800])
    return int(1900+np.argmax(np.median(curves,axis=0)))


def designs(raw,p,boundary):
    t=np.arange(len(p))
    keep=np.flatnonzero((t>=boundary-160)&(t<=boundary+350))
    if not len(keep):
        return keep, []
    dist=keep-boundary
    raw=raw[:keep[-1]+1]
    r=raw[:,:,0].astype(float)-.00136
    cs=np.vstack((np.zeros(50),r.cumsum(0)))
    price=cs[keep+1]
    # At each tick, the reference price comes from the earlier of that tick and the last pre-break tick.
    prev=cs[np.minimum(keep+1,boundary)]
    jump=r[np.minimum(keep,boundary)]*(dist>=0)[:,None]
    post=(dist>=0)[:,None]
    book_now=np.arcsinh(raw[keep][:,:,[10,20,21,22,23,26,38]].astype(float))
    state=np.concatenate((p[keep],book_now, np.stack((
        np.arcsinh(price-prev),np.arcsinh(jump),np.arcsinh((price-cs[(keep-19).clip(0)])/4.5),
        np.arcsinh((price-cs[(keep-99).clip(0)])/10)),axis=2)),axis=2)
    result=[]
    for h in (5,20,60):
        crosses=((dist<0)&(dist+h>=0))[:,None,None]
        before=np.exp(-abs(dist+h/2)/70)[:,None,None]*(~post)[:,:,None]
        since=post[:,:,None]*np.exp(-dist.clip(0)/40)[:,None,None]
        late=post[:,:,None]*np.exp(-dist.clip(0)/150)[:,None,None]
        f=np.concatenate((crosses*state,before*state,since*state,late*state),axis=2)
        m=f.mean(1)
        result.append((m,(f-m[:,None]).astype('float32')))
    return keep,result


def stats(p,y,idx,x,j):
    a=p[:,:,j].ravel().astype(float)
    b=y[:,:,j].ravel().astype(float)
    a-=a.mean();b-=b.mean()
    slope=np.clip((a@b)/(a@a),.4,2.5)
    e=y[idx,:,j]/slope-p[idx,:,j]
    m,r=x;common=e.mean(1);local=(e-common[:,None]).ravel()
    r=r.reshape(len(local),-1).astype(float)
    return ((m.T@m/len(m),m.T@common/len(m)),(r.T@r/len(r),r.T@local/len(r)))


def fit(history):
    weights=[]
    for horizon in zip(*history):
        branch=[]
        for strength,days in zip((4.,1.1),zip(*horizon)):
            g,b=(np.mean(a,0) for a in zip(*days))
            reg=strength*np.diag(g.diagonal().clip(.0002,None))
            sols=np.stack([np.linalg.solve(gg+reg,bb) for gg,bb in days])
            w=.5*np.linalg.solve(g+reg,b)+.5*np.median(sols,0)
            branch.append(.75*w*abs(np.sign(sols).mean(0)))
        weights.append(branch)
    return np.array(weights)


def advance(raw,p,boundary,w):
    idx,options=designs(raw,p,boundary)
    terms=np.zeros((2,*p.shape),np.float32)
    for j,(m,r) in enumerate(options):
        terms[0,idx,:,j]=(m@w[j,0])[:,None]
        terms[1,idx,:,j]=r@w[j,1]
    return p+terms.sum(0),terms


def main():
    OUT.mkdir(exist_ok=True)
    boundary=restart()
    hist=[];gains=[];report={'restart_tick':boundary}
    for end,stop in ((8,10),(10,12),(12,14),(14,16),(16,20)):
        if end>=10:
            w=fit(hist)
        if end==16:
            g=np.stack(gains)
            mask=((g>0).sum(0)>=4)&(g[-2:].mean(0)>0)&(g.mean(0)>0)
            report['eligible']=mask.tolist()
            if not mask.any():
                (OUT/'results.json').write_text(json.dumps(report,indent=2))
                return
            w*=mask.T[:,:,None]
            np.savez(OUT/'params.npz',weights=w,boundary=boundary)
        rows=[]
        for d in range(end,stop):
            p=np.load(PARENT/f'pred_{d}.npy') if end==16 else early.base(d)
            raw=book.rawday(d)
            if end>=10:
                q,terms=advance(raw,p,boundary,w)
                if d==end:
                    for k in (boundary-11,boundary+12):
                        np.testing.assert_allclose(q[:k],advance(raw[:k],p[:k],boundary,w)[0],atol=1e-5)
                if end==16:
                    np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:
                idx,z=designs(raw,p,boundary)
                hist.append([stats(p,y,idx,feat,j) for j,feat in enumerate(z)])
            if end>=10:
                row=tuple(v.reshape(-1,3) for v in (p,q,y))
                rows.append(row)
                if end<16:
                    base=corr(row[0],row[2])
                    gains.append([corr((p+c).reshape(-1,3),row[2])-base for c in terms])
        if end>=10:
            report[str(stop)]=measure(rows)
            print(stop,report[str(stop)],flush=True)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if report['20']['gain']>0:
        replay.main(PARENT,OUT,lambda raw,p:advance(raw,p,boundary,w)[0])


if __name__=='__main__':
    main()
