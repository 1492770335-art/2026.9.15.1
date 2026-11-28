"""Estimate stock portfolios whose forward price response to the book replicates on separate dates."""
import sys
import numpy as np
import experiment_dense_matched_distal_panels_v1 as panel
from prepare_dense_distal_online_angles_reference_v1 import OUT as REFERENCE

onset, rel = panel.onset, panel.rel
OUT = REFERENCE.parent / "dense_replicated_stock_portfolios_v1"
ARMS = ("unrestricted", "replicated_distal_portfolios", "shared_distal_and_full_portfolios")
M = 450


def inputs(raw, grid):
    for a,b in rel.fields.parent.sessions(grid):
        v = raw[a:b]
        def fields(ids):
            return np.tanh(.65*np.arcsinh(v[:,:,ids].astype(float)))
        depth = fields([41,50,54]) - fields([42,51,55])
        state = np.concatenate((depth,fields([0,11,26,38,64,74])),axis=2)
        fast = panel.smooth(state,8)
        slow = panel.smooth(state,96)
        agg = np.stack((fast[:,:,0],fast[:,:,1:3].mean(2),fast[:,:,3],fast[:,:,4],
                        slow[:,:,0],slow[:,:,1:3].mean(2),slow[:,:,5]-slow[:,:,6],
                        slow[:,:,7],slow[:,:,8]),axis=2)
        # Subtract the common direction and give every stock its own inputs.
        yield a,b,panel.centered(agg).reshape(b-a,M)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(16):
        raw,grid,y=[onset.settled.load(d,key) for key in ("x","time","y")]
        full=panel.centered(y[:,:,2].astype(float))
        far=panel.centered(y[:,:,2].astype(float)-panel.ratio()*y[:,:,1])
        g=0.; n=0
        for a,b,x in inputs(raw,grid):
            t=np.arange(8+d%5,b-a-60,5)
            obs=np.column_stack((x[t],far[a+t],full[a+t]))
            obs-=obs.mean(0)
            g+=obs.T@obs
            n+=len(obs)
        np.save(OUT/f"gram_{d:02d}.npy",g/n)


def whitening(g, penalty):
    sd=np.sqrt(g.diagonal().clip(1e-8))
    eig,u=np.linalg.eigh(g/np.outer(sd,sd))
    return (u/np.sqrt(eig.clip(0)+penalty))@u.T/sd[:,None]


def fit(end):
    grams=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(end)])
    g=grams.mean(0)
    far=slice(M,M+50)
    full=slice(M+50,None)
    wx=whitening(g[:M,:M],1.8)
    wy=whitening(g[far,far],.55)
    # wx.T X and wy.T Y have well-conditioned covariance across stock names.
    a=wx.T@grams[:,:M,far]@wy
    b=wx.T@grams[:,:M,full]@wy
    w=np.zeros((M,3,50))
    diag=[]
    for held in range(3):
        first,second,read=[np.arange(j,end,3) for j in ((held+1)%3,(held+2)%3,held)]
        left,right=a[first].mean(0),a[second].mean(0)
        response=wx.T@grams[read,:M,far].mean(0)
        w[:,0]+=wx@response/3
        for arm,partner in enumerate((right,.5*(right+b[second].mean(0))),1):
            evidence=left.T@partner
            eig,u=np.linalg.eigh(.5*(evidence+evidence.T))
            # Independent fitting dates choose the basket; the third partition learns its price response.
            u=u[:,-6:][:,eig[-6:]>0]
            q=np.linalg.qr(wy@u,mode="reduced")[0]
            w[:,arm]+=wx@response@q@q.T/3
            diag.append((arm,float(eig[-1]),len(q.T)))
    return w,diag


def train(end):
    return fit(end)[0]


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],len(ARMS)))
    for a,b,x in inputs(raw,grid):
        out[a:b]=(x@bank.reshape(M,-1)).reshape(b-a,len(ARMS),50).swapaxes(1,2)
    return panel.centered(out)


def predict(raw,grid,p,params):
    q=p.copy()
    z=signals(raw,grid,p,params["bank"])
    q[...,2]+=(rel.relative.features(p,z)@params["w"]).astype(q.dtype)
    return q


if __name__=="__main__":
    collect()
    panel.rolling(sys.modules[__name__])
