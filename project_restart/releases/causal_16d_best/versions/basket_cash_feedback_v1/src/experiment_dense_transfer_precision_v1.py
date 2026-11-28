"""Borrow stock-level transitions and payoffs in proportion to their date-level evidence in market units."""
import numpy as np
import prepare_dense_market_operator_moments_v1 as reference

OUT=reference.base.ROOT/"artifacts/dense_transfer_precision_v1"
ARMS=("payoff_precision","transition_precision","joint_precision")


def adapt(target,source):
    # Paired dates share a market regime; a persistent difference limits the stock prior.
    n=len(target);diff=source-target
    error=np.square(diff.mean(0))
    sampling=(np.var(diff,axis=0,ddof=1)/n)
    direct=np.maximum(0,np.var(target,axis=0,ddof=1)-np.mean((target-target.mean(0))*(source-source.mean(0)),0)*n/(n-1))/n
    base=np.mean(sampling)+1e-10
    bias=np.maximum(0,error-sampling)
    w=((direct+.2*base)/(sampling+bias+.4*base)).clip(.06,.94)
    return target.mean(0)+w*diff.mean(0)


def operators(stock,mkt):
    g,s=np.mean(mkt,0),np.mean(stock,0);n=(len(g)-1)//3
    sd=np.sqrt(g.diagonal()[:n]).clip(.006);ss=np.sqrt(s.diagonal()[:n]).clip(.03)
    ratio=np.r_[np.tile(sd/ss,3),np.sqrt(g[-1,-1]/s[-1,-1])]
    mkt=np.stack(mkt);st=np.stack(stock)*np.outer(ratio,ratio)
    cov=.5*(g[:n,:n]+st[:,:n,:n].mean(0))/np.outer(sd,sd)
    eig,u=np.linalg.eigh(cov+.25*np.eye(n))
    root=(u/np.sqrt(eig))/sd[:,None]
    def parts(c):
        A=root.T@(c[:,:n,n:2*n]+c[:,:n,2*n:3*n])@root
        return .5*(A+A.transpose(0,2,1)),c[:,:n,-1]@root
    ma,mp=parts(mkt);sa,sp=parts(st)
    mixed=.5*(ma+sa);reward=.5*(mp+sp)
    bank=[]
    for j in (0,1):
        a=slice(j,None,2);b=slice(1-j,None,2)
        fixed=mixed[a].mean(0);pay=reward[b].mean(0)
        empirical=adapt(ma[a],sa[a]);learned=adapt(mp[b],sp[b])
        bank.append(root@np.column_stack((fixed@learned,empirical@pay,empirical@learned)))
    return np.mean(bank,0)


if __name__=="__main__":reference.run(OUT,ARMS,operators)
