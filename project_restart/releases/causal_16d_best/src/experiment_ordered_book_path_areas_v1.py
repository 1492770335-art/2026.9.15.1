"""Use signed areas of observed price, liquidity and pressure paths to retain event order."""
import json
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure
import experiment_repricing_adjusted_book_flow_v1 as flow
import experiment_eight_date_expert_contrasts_v1 as earlier
import experiment_delayed_tail_band_skill_v1 as ref

OUT=ART/'ordered_book_path_areas_v1'
HORIZONS=(8,32,120,420)
PAIRS=((0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),
       (1,2),(1,3),(1,6),(1,7),(1,8),(2,4),(2,6),(2,7),(2,9),
       (3,4),(3,6),(3,7),(3,9),(4,6),(4,8),(5,6),(5,9),(5,10))
I,J=np.array(PAIRS).T
BOTH=np.where((I<6)&(J<6))[0]


class Path:
    def __init__(self,raw):
        t=np.arange(len(raw))
        r=raw[:,:,0].astype(float)-.00136
        vol=np.sqrt((8+flow.trailing(r*r,96))/(8+np.minimum(t+1,96))[:,None]).clip(.5,4)
        def field(j):return np.arcsinh(raw[:,:,j].astype('float32'))
        pressure=np.tanh(.7*(field(41)-field(42)))
        imbalance=np.tanh(.7*(field(54)-field(55)))
        signed_trade=np.tanh(field(11))
        market=r.mean(1)
        def cumulative(x):return np.cumsum(x,axis=0,dtype=np.float64)
        def common(x):return np.broadcast_to(x[:,None],r.shape)
        # Five signed channels accompany price; the rest are liquidity states or clocks.
        z=np.stack([cumulative(r/vol),common(cumulative(market)),cumulative(pressure),
                    cumulative(imbalance),signed_trade,field(23),field(39),field(40),
                    np.log(vol),cumulative((abs(r)>.018).astype(float)),common(t)],axis=2)
        delta=np.diff(z,axis=0,prepend=z[:1])
        self.z=z
        self.q=cumulative(delta*delta)
        self.cross=np.empty((len(raw),50,len(I)),dtype=np.float64)
        for a,(i,j) in enumerate(PAIRS):
            # z[k] wedge delta[k] equals z[k-1] wedge delta[k] for linear interpolation.
            self.cross[:,:,a]=.5*cumulative(z[:,:,i]*delta[:,:,j]-z[:,:,j]*delta[:,:,i])
        self.sign=np.tanh(.55*(pressure+imbalance)+.6*signed_trade)

    def design(self,t):
        parts=[]
        for h in HORIZONS:
            l=(t-h).clip(0);base=self.z[l];end=self.z[t]
            y=self.cross[t]-self.cross[l]
            shift=.5*(base[:,:,I]*(end-base)[:,:,J]-base[:,:,J]*(end-base)[:,:,I])
            area=y-shift
            q=self.q[t]-self.q[l]
            volume=np.sqrt((.5+q[:,:,I])*(.5+q[:,:,J]))
            # A bounded normalization keeps sparse price spells and the activity clock comparable.
            a=np.arcsinh(area/(1+volume)).clip(-4,4)
            parts.extend([a,a[:,:,BOTH]*self.sign[t,:,None]])
        x=np.concatenate(parts,axis=2).astype('float32');m=x.mean(1)
        return np.column_stack((m,m[:,:len(PAIRS)]*np.exp(-t[:,None]/350))).astype('float32'),x-m[:,None]


def predict(data,p,w):
    a=np.empty((len(p),1,3),np.float32);b=np.empty_like(p)
    for start in range(0,len(p),119):
        t=np.arange(start,min(start+119,len(p)))
        m,r=data.design(t);a[t,0]=m@w[0];b[t]=r@w[1]
    return a,b


def main():
    OUT.mkdir(exist_ok=True);hist=[];report={};daily=[]
    for d in range(8,20):
        if d in (10,12,14,16):
            w=flow.fit(hist);rows=[]
            if d==16:
                a=np.stack(daily)
                gate=((a>0).sum(0)>=4)&(a.mean(0)>0)&(a[-2:].mean(0)>0)
                report['eligible']=gate.tolist()
                if not gate.any():break
                w=[v*g for v,g in zip(w,gate)]
                np.savez(OUT/'model.npz',market=w[0],relative=w[1])
        raw=flow.rawday(d);p=ref.base(d);data=Path(raw)
        if d>=10:
            a,b=predict(data,p,w);q=(p+a+b).astype('float32')
            if d in (10,16):
                a0,b0=predict(Path(raw[:318]),p[:318],w)
                np.testing.assert_allclose(q[:318],p[:318]+a0+b0,atol=3e-6)
            if d>=16:np.save(OUT/f'pred_{d}.npy',q)
        y=labels(d).reshape(p.shape)
        if d<16:
            t=np.arange(d%6,len(p),6)
            hist.append(earlier.moments(p[t],y[t],data.design(t)))
        if d>=10:
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            if d<16:
                old=corr(row[0],row[2])
                daily.append([corr((p+c).reshape(-1,3),row[2])-old for c in (a,b)])
        if d in (11,13,15,19):
            report[str(d+1)]=measure(rows);print(d+1,report[str(d+1)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

