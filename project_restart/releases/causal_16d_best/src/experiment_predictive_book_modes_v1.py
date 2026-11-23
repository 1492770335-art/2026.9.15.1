"""Learn current-book directions predictive of the delayed book, then forecast ticks 21–60."""
import json
import numpy as np

import experiment_pairwise_book_tree_v1 as book
import experiment_tail_portfolio_filter_v1 as panel
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure, corr

OUT = ART / 'predictive_book_modes_v1'
SIGNED = [0, 2, 3, 5, 7, 8, 10, 12, 21, 23, 28, 29, 30]


def parts(state, risk, t):
    a = state[t]
    avg = a.mean(1, keepdims=True)
    signed = a[:, :, SIGNED]
    common = np.column_stack((avg[:, 0], np.tanh(1.5*signed).mean(1),
                              np.exp(-t/380), np.log(risk[t]).mean(1)))
    relative = np.concatenate((a-avg, np.tanh(1.5*signed), np.log(risk[t,:,None])), axis=2)
    return common, relative


def fit_axes():
    daily = [[], []]
    for d in range(8):
        state, risk = book.state(book.rawday(d))
        t = np.arange(d%9,len(state)-61,9)
        now = parts(state,risk,t)
        later = [parts(state,risk,t+k) for k in (24,32,40,48,56)]
        for i in range(2):
            x = now[i].reshape(-1,now[i].shape[-1]).astype(float)
            f = [v[i][...,SIGNED].reshape(len(x),-1) for v in later]
            future = np.column_stack((np.mean(f,axis=0),(f[-2]+f[-1]-f[0]-f[1])*.5)).astype(float)
            xm = x.mean(0);x -= xm;future -= future.mean(0)
            daily[i].append((xm,x.T@x/len(x),x.T@future/len(x),future.T@future/len(x)))
    result = []
    for samples in daily:
        mu,g,c,f = [np.mean(v,axis=0) for v in zip(*samples)]
        def whiten(gram):
            reg = .18*np.diag(gram.diagonal().clip(.002,None))
            return np.linalg.inv(np.linalg.cholesky(gram+reg))
        wx = whiten(g); wf = whiten(f)
        u,_,_ = np.linalg.svd(wx@c@wf.T,full_matrices=False)
        result.append((mu.astype('float32'),g.diagonal().clip(.002,None).astype('float32')**.5,
                       (wx.T@u[:,:14]).astype('float32')))
    return result


def designs(values,axis):
    mean,scale,rotation = axis
    x = values.reshape(-1,values.shape[-1])-mean
    c = (x@rotation).clip(-5,5)
    x = (x/scale).clip(-5,5)
    state = np.tanh(x[:,SIGNED])
    pair_i,pair_j = np.triu_indices(8,1)
    direct = np.column_stack((np.ones_like(c[:,0]),x,state))
    delayed = np.column_stack((np.ones_like(c[:,0]),c,np.tanh(c),c[:,pair_i]*c[:,pair_j],
                               (state[:,:,None]*np.tanh(c[:,:5])[:,None,:]).reshape(len(x),-1)))
    return direct,delayed


def daily_moments(raw,y,d,axes):
    state,v = book.state(raw)
    t = np.arange(d%6,len(state)-61,6)
    sides = parts(state,v,t)
    tail = y[t,:,2]-panel.RATIO*y[t,:,1]
    common = tail.mean(1,keepdims=True)
    measurements = []
    for i in range(2):
        vol = .45+.55*v[t]
        truth = common[:,0]/vol.mean(1) if i==0 else (tail-common)/vol
        b = truth.ravel().astype(float).clip(-5,5)
        for x in designs(sides[i],axes[i]):
            a = x.astype(float)
            measurements.append((a.T@a/len(a),a.T@b/len(a)))
    return measurements


def fit(history):
    result = []
    for part,days in enumerate(zip(*history)):
        g,b = [np.stack(v) for v in zip(*days)]
        strength = .65 if part<2 else .22
        reg = strength*np.diag(g.mean(0).diagonal().clip(.002,None))
        reg[0,0] = .025
        result.append(np.linalg.solve(g.mean(0)+reg,b.mean(0)).astype('float32'))
    return result


def predict(raw,axes,w):
    state,risk = book.state(raw)
    out = np.empty((len(raw),50,2),np.float32)
    for begin in range(0,len(raw),128):
        t = np.arange(begin,min(begin+128,len(raw)))
        v = .45+.55*risk[t]
        market,relative = parts(state,risk,t)
        for j,(m,x) in enumerate(zip(designs(market,axes[0]),designs(relative,axes[1]))):
            local = (x@w[2+j]).reshape(len(t),50)*v
            out[t,:,j] = ((m@w[j])*v.mean(1))[:,None]+local-local.mean(1,keepdims=True)
    return out


def corrections(p,side):
    tail = p[:,:,2]-panel.RATIO*p[:,:,1]
    delta = side-tail
    m = delta.mean(1,keepdims=True)
    return np.stack((np.broadcast_to(m,delta.shape),delta-m),axis=-1)


def slopes(p,y,z):
    e = panel.target_residual(p,y)
    return ((z*e[:,:,None]).mean((0,1))/(z**2).mean((0,1)).clip(.00001,None)).clip(0,.5)


def validation(advance):
    import polars as pl
    import replay_session_experts_16d as replay
    import experiment_peer_implied_tail_v1 as peer
    from experiment_peer_quote_reliability_v1 import partial
    with np.load(peer.OUT/'model.npz') as model:
        graph,coef = model['peers'],model['readout']
    ref = json.loads((peer.OUT/'raw_replay.json').read_text())
    all_rows=[]; daily={};actual=[];inc=[];new=[]
    for day in range(16,20):
        raw=replay.features(replay.DATA,day)
        p,head,n=panel.replay_parent(day,raw)
        p=partial(raw,p,graph,coef);head=partial(raw[:n],head,graph,coef)
        q=advance(raw,p);short=advance(raw[:n],head)
        err=float(abs(q[:n]-short).max())
        if not np.isfinite(q).all() or err>=1e-4:
            raise ValueError('causal prefix mismatch')
        y=pl.scan_parquet(replay.DATA).filter(pl.col('date')==day).sort('time','code').select('y0','y1','y2').collect().to_numpy().reshape(p.shape)
        row=tuple(v.reshape(-1,3) for v in (p,q,y));all_rows.append(row)
        score=measure([row])
        if abs(score['baseline_score']-ref['daily'][str(day+1)]['score'])>2e-6:
            raise ValueError('reference mismatch')
        daily[day+1]=dict(score=score['score'],gain=score['gain'],prefix_error=err)
        for buf,v in zip((inc,new,actual),(p,q,y)):buf.append((v[:,:,2]-panel.RATIO*v[:,:,1]).reshape(-1,1))
    s=measure(all_rows)
    if s['rows']!=936250:raise ValueError('incorrect row count')
    return dict(pooled=s,daily=daily,tail_ic=[float(corr(np.concatenate(buf),np.concatenate(actual))[0]) for buf in (inc,new)])


def main():
    OUT.mkdir(exist_ok=True)
    axes = fit_axes();history=[];strength=[[],[]];report={'folds':[]}
    changes=[[],[]]
    for end in (8,10,12,14):
        for d in range(len(history),end):
            raw=book.rawday(d);y=labels(d).reshape(-1,50,3)
            history.append(daily_moments(raw,y,d,axes))
        coeff=fit(history)
        w = [.65*np.median(h,axis=0) if h else None for h in strength]
        fold=[[],[]]
        for d in range(end,end+2):
            raw=book.rawday(d);p=panel.early(d);y=labels(d).reshape(p.shape)
            pred=predict(raw,axes,coeff)
            if d==end:np.testing.assert_allclose(pred[:233],predict(raw[:233],axes,coeff),rtol=0,atol=3e-5)
            tail=y[:,:,2]-panel.RATIO*y[:,:,1]
            for j in (0,1):
                z=corrections(p,pred[:,:,j])
                q=p.copy()
                if w[j] is not None:
                    q[:,:,2]+=z@w[j]
                    changes[j].append(panel.scores(q,y,0)-panel.scores(p,y,0))
                fold[j].append((p,q,y,pred[:,:,j],tail))
                strength[j].append(slopes(p,y,z))
        info={'days':[end+1,end+2]}
        for j,name in enumerate(('direct','delayed_modes')):
            p,q,y,f,tail=(np.concatenate(v) for v in zip(*fold[j]))
            info[name]=dict(standalone_tail_ic=float(corr(f.reshape(-1,1),tail.reshape(-1,1))[0]),
                            baseline=panel.scores(p,y,0).tolist(),combined=panel.scores(q,y,0).tolist())
        report['folds'].append(info);print(info,flush=True)
    gains=np.array(changes[1])
    blocks=np.array([np.subtract(f['delayed_modes']['combined'],f['delayed_modes']['baseline']) for f in report['folds'][1:]])
    gate=bool((blocks>0).all() and ((gains>0).sum(0)>=4).all())
    report['retained']=gate;path=OUT/'results.json'
    path.write_text(json.dumps(report,indent=2))
    if gate:
        for d in range(14,16):
            history.append(daily_moments(book.rawday(d),labels(d).reshape(-1,50,3),d,axes))
        coeff=fit(history);w=.65*np.median(strength[1],axis=0)
        params={f'axis_{i}_{j}':v for i,a in enumerate(axes) for j,v in enumerate(a)}
        np.savez(OUT/'model.npz',**params,**{f'coef_{i}':v for i,v in enumerate(coeff)},blend=w)
        def advance(raw,p):
            q=p.copy();q[:,:,2]+=corrections(p,predict(raw,axes,coeff)[:,:,1])@w
            return q
        report['development']=validation(advance)
        path.write_text(json.dumps(report,indent=2));print(report['development'],flush=True)


if __name__=='__main__':
    main()
