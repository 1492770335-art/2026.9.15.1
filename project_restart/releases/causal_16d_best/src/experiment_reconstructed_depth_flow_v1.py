"""Infer the two sides of displayed depth from total depth and imbalance, then track depth net of executions."""
import numpy as np
from scipy.optimize import least_squares
import experiment_persistent_execution_state_v1 as parent
import experiment_execution_location_markouts_v1 as loc
import experiment_execution_volume_clock_v1 as clock
from experiment_causal_cross_horizon_ridge_v1 import ART

OUT=ART/'reconstructed_depth_flow_v1'


def observations(raw,g):
    z=raw[:,:,37].astype(float);old=np.vstack((z[:1],z[:-1]))
    amount=raw[:,:,52].astype(float)-g['floor52']
    vol=raw[:,:,8].astype(float)-g['floor'][0]
    mask=(raw[:,:,54]>g['floor'][1])&(raw[:,:,55]==g['floor'][2])&(vol>.025)&(amount>.002)
    mask[0]=False
    return z,old,amount,vol,mask


def depths(raw,g):
    slope,offset=g['depth'];center,radius,_,_=g['inverse']
    total=np.exp((slope*raw[:,:,37].astype(float)+offset).clip(-13,8))
    imbalance=((raw[:,:,22].astype(float)-center)/radius).clip(-.9999,.9999)
    return total*(1+imbalance)*.5,total*(1-imbalance)*.5,imbalance,total


def traded(raw,g):
    a,b,v,*_=clock.decode(raw,g)
    return a,b,v


def geometry():
    with np.load(loc.OUT/'model.npz') as st:g={k:st[k].copy() for k in st.files if k not in ('market','relative')}
    g['carry_locked_quantum']=np.array(True)
    g['floor52']=np.array(clock.pair.rawday(0)[:,:,52].min(),dtype=float)
    logdata=[]
    for d in range(4):
        z,o,a,v,ok=observations(clock.pair.rawday(d),g)
        logdata.append(np.column_stack((z[ok],o[ok],np.log(v[ok]/a[ok])))[::9])
    z,o,y=np.concatenate(logdata).T
    def cost(w):return w[1]+np.logaddexp(w[0]*z,w[0]*o)-np.log(2)-y
    g['depth']=least_squares(cost,[1.,-.8],loss='soft_l1',f_scale=.002).x
    states=[]
    for d in range(4):
        raw=clock.pair.rawday(d)[d%9::9];v=raw[:,:,22].astype(float)
        depth=np.exp(g['depth'][0]*raw[:,:,37].astype(float)+g['depth'][1])
        obs=raw[:,:,38].astype(float);ok=(depth>.025)&(depth<20)
        states.append(np.column_stack((v[ok],depth[ok],obs[ok])))
    obs,depth,target=np.concatenate(states).T
    def inverse(w):
        frac=((obs-w[0])/w[1]).clip(-.998,.998)
        return w[3]+w[2]*frac/(depth*(1-frac*frac))-target
    g['inverse']=least_squares(inverse,[-.09,2.39,.37,-.04],bounds=([-.4,2.25,.05,-.5],[.2,2.75,1.,.5]),loss='soft_l1',f_scale=.008).x
    # Estimate the common scale between the anonymous depth and quantity units from isolated depletions.
    ratios=[]
    for d in range(4):
        raw=clock.pair.rawday(d);bid,ask,_,_=depths(raw,g);buy,sell,_=traded(raw,g)
        for size,side,opposite,age in ((bid,sell,buy,77),(ask,buy,sell,78)):
            dif=np.diff(size,axis=0,prepend=size[:1]);live=raw[:,:,age]>raw[:,:,age].min()
            ok=live&(opposite==0)&(side>.04)&(dif<-.002)
            z=-dif[ok]/side[ok];ratios.append(z[(z>.01)&(z<30)])
    z=np.log(np.concatenate(ratios));hist,edge=np.histogram(z,bins=160,range=(np.log(.01),np.log(30)))
    i=np.argmax(hist);g['amount_scale']=np.array(np.exp((edge[i]+edge[i+1])*.5))
    return g


def audit(g):
    result={}
    for d in range(4,8):
        raw=clock.pair.rawday(d);z,o,a,v,ok=observations(raw,g)
        buy,sell,f,total=depths(raw,g);old=np.vstack((total[:1],total[:-1]))
        p=.5*(total+old);_,_,gain,bias=g['inverse'];obs=raw[:,:,38].astype(float)
        est=bias+gain*f/(total*(1-f*f)).clip(1e-9,None);keep=(total>.025)&(total<20)
        result[d+1]=dict(denominator_median_relative_error=float(np.median(abs(p[ok]*a[ok]/v[ok]-1))),
            inverse_depth_corr=float(np.corrcoef(est[keep],obs[keep])[0,1]),inverse_depth_median_error=float(np.median(abs(est[keep]-obs[keep]))),
            diagnostic_coverage=float(np.mean(keep)))
    return result


class Day:
    def __init__(self,raw,g):
        bid,ask,imb,total=depths(raw,g);buy,sell,v,b,a,sp,q,_=clock.decode(raw,g)
        tb=np.vstack((bid[:1],bid[:-1]));ta=np.vstack((ask[:1],ask[:-1]))
        norm=np.maximum(.015,(total+tb+ta)*.5);fac=g['amount_scale']
        # Anonymous depth can span several levels, so a shift in quote has its own feature.
        change_b=(bid-tb)/norm;change_a=(ask-ta)/norm
        hit_b=fac*sell/norm;hit_a=fac*buy/norm
        inflow_b=change_b+hit_b;inflow_a=change_a+hit_a
        keep_b=(b==0);keep_a=(a==0);rest=inflow_b*keep_b-inflow_a*keep_a
        clear=change_b*(~keep_b)-change_a*(~keep_a)
        micro=imb*sp*q
        thin=(1/(bid+.03)-1/(ask+.03))*norm
        signs=np.stack((imb,micro,np.arcsinh(thin),change_b-change_a,hit_a-hit_b,inflow_b-inflow_a,
                     rest,clear,hit_a*(a==0)-hit_b*(b==0),inflow_b*(inflow_b>0)-inflow_a*(inflow_a>0),
                     inflow_b*(inflow_b<0)-inflow_a*(inflow_a<0),
                     fac*buy/(ta+.025)-fac*sell/(tb+.025),np.tanh(a/2)+np.tanh(b/2),
                     (inflow_b-inflow_a)*np.tanh(change_b+change_a)),axis=2)
        z=np.arcsinh(signs).clip(-5,5).astype('float32')
        t=np.arange(len(raw))+1;cs=np.concatenate((np.zeros_like(z[:1],dtype=float),z.cumsum(0,dtype=float)))
        pieces=[z]
        for lo,hi in ((0,6),(6,23),(23,81),(81,233)):
            end=(t-lo).clip(0);start=(t-hi).clip(0)
            hist=np.arcsinh((cs[end]-cs[start])/np.sqrt((end-start).clip(1))[:,None,None])
            pieces.extend((hist,(hist[:,:,:8]*np.tanh(np.log(total+.01))[:,:,None]),
                           hist[:,:,:8]*np.tanh(imb*micro)[:,:,None]))
        self.x=np.concatenate(pieces,axis=2).astype('float32')

    def design(self,t):
        v=self.x[t];m=v.mean(1)
        return np.column_stack((m,m[:,:22]*np.exp(-t[:,None]/340))).astype('float32'),v-m[:,None]


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as replay
        replay.main(parent.OUT,OUT,clock.frozen(OUT,Day))
    else:
        g=geometry();clock.evaluate(OUT,parent.OUT,Day,g,{'depth_identity_checks':audit(g)})
