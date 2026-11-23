"""Estimate information in upper and lower tails of the fitted conditional return distribution."""
import json
import numpy as np
import torch
from scipy.special import gammaln, stdtr
import experiment_asymmetric_return_distribution_v1 as dist
import experiment_eight_date_expert_contrasts_v1 as early
import experiment_pairwise_book_tree_v1 as pair
import experiment_phase_future_book_v1 as phase
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

OUT=ART/'conditional_tail_payouts_v1'


def load_model(end,device):
    st=torch.load(dist.OUT/f'net_{end}.pth',weights_only=True,map_location='cpu')
    net=dist.DistributionNet(st['current.0.weight'].shape[1]);net.load_state_dict(st)
    return net.to(device).eval()


def payouts(mode,left,right,df,mean):
    unit=np.ones((1,51,1),np.float32);unit[:,50]=.45
    md=mode/unit;l=left/unit;r=right/unit;mu=mean/unit
    def tail(level):
        k=(level-md)/np.where(level<md,l,r)
        cdf=stdtr(df,k)
        density=np.exp(gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi)-.5*(df+1)*np.log1p(k*k/df))
        factor=(df+k*k)/(df-1)*density
        call=2*r/(l+r)*((md-level)*(1-cdf)+r*factor)
        put=2*l/(l+r)*((level-md)*cdf+l*factor)
        return np.where(level>=md,call,mu-level+put)
    def put(level):return tail(level)-mu+level
    k=-md/np.where(md>0,l,r)
    cdf=stdtr(df,k)
    prob=np.where(md>0,2*l/(l+r)*cdf,(l-r+2*r*cdf)/(l+r))
    parts=[md,mu-md,(1-2*prob)*.75,mu*np.tanh(np.log((l+r)/2+.001))]
    for boundary in (.45,1.25,2.75):
        up=tail(boundary);down=put(-boundary)
        net=up-down
        parts.extend((net,mu*np.tanh(up+down)))
    return np.concatenate([a*unit for a in parts],axis=2).astype('float32')


@torch.inference_mode()
def design(net,raw,p):
    device=next(net.parameters()).device
    market=None
    for lag in range(4):
        ts=np.arange(lag,len(raw),4);x,v=phase.spline.features(raw,ts)
        for first in range(0,len(ts),48):
            t=ts[first:first+48];xx=torch.from_numpy(x[first:first+48]).to(device).float()
            vv=(.45+.55*v[first:first+48]).clip(.55,3)
            g=torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')).to(device)
            pred,_,mo,l,r,nu,mean=net(xx,torch.from_numpy(vv).to(device),g)
            z=payouts(*[a.float().cpu().numpy() for a in (mo,l,r,nu,mean)])
            if market is None:
                market=np.empty((len(raw),z.shape[-1]),np.float32)
                relative=np.empty((len(raw),50,z.shape[-1]),np.float32)
            market[t]=z[:,50]*vv.mean(1,keepdims=True)
            rel=z[:,:50]*vv[:,:,None];rel-=rel.mean(1,keepdims=True)
            # Remove ranks spanned by the reference and by the parent model's mean.
            control=np.concatenate((p[t],pred.cpu().numpy()),axis=-1).astype(float)
            control-=control.mean(1,keepdims=True)
            tr=control.transpose(0,2,1);gram=tr@control/50
            reg=.0005+.035*gram.diagonal(axis1=1,axis2=2)
            relative[t]=rel-control@np.linalg.solve(gram+np.eye(6)*reg[:,None,:],tr@rel/50)
    return market,relative


def main():
    OUT.mkdir(exist_ok=True);hist=[];report={};good=np.ones((2,3),bool)
    for end,stop in ((12,14),(14,16),(16,20)):
        net=load_model(end,'cpu');rows=[];daily=[]
        if end>=14:
            w=[a*b for a,b in zip(early.fit(hist),good)]
            if end==16:np.savez(OUT/'model.npz',market=w[0],relative=w[1])
        for d in range(end,stop):
            base=ART/'categorical_suffix_responses_v1' if d>=16 else (pair.OUT if d>=14 else phase.ref.OUT)
            p=np.load(base/f'pred_{d}.npy');raw=pair.rawday(d)
            z=design(net,raw,p)
            if d==end:
                small=design(net,raw[:137],p[:137])
                for a,b in zip(z,small):np.testing.assert_allclose(a[:137],b,atol=.00008)
            if end>=14:
                a=(z[0]@w[0])[:,None];b=z[1]@w[1];q=(p+a+b).astype('float32')
                if d>=16:np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:hist.append(early.moments(p,y,z))
            if end>=14:
                row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
                if end==14:
                    old=corr(row[0],row[2]);daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in (a,b)])
        if end>=14:
            report[str(stop)]=measure(rows)
            if end==14:good=np.min(daily,axis=0)>0;report['eligible']=good.tolist()
            (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report,flush=True)
            if not good.any():break


def refinement():
    import experiment_categorical_suffix_responses_v1 as previous
    _,first=previous.refinement()
    net=load_model(16,'cpu')
    with np.load(OUT/'model.npz') as st:
        wm=st['market'].copy();wr=st['relative'].copy()
    def refine(raw,p):
        base=first(raw,p);m,r=design(net,raw,base)
        return (base+(m@wm)[:,None]+r@wr).astype('float32')
    return OUT,refine


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_paired_book_trees_v1 as replay
        replay.main(refinement())
    else:main()
