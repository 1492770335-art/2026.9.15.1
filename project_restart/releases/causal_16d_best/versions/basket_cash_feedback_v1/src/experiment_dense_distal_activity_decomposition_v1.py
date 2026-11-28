"""Separate the distal effect of activity on future book states from its effect on their payoff."""
import numpy as np
import experiment_dense_activity_transition_curve_v1 as activity
from prepare_dense_activity_reference_v1 import OUT as REFERENCE

rel=activity.rel
OUT=REFERENCE.parent/"dense_distal_activity_decomposition_v1"
ARMS=("distal_transition","distal_payoff","first_twenty")


def train(end):
    data=[]
    for d in range(end):
        with np.load(activity.OUT/f"day_{d:02d}.npz") as f:
            data.append(tuple(f[k] for k in ("g","lag5","lag20","paid")))
    gs,l5,l20,prices=map(np.stack,zip(*data))
    n=len(prices[0]);small=n//3
    bank=[]
    for side in (0,1):
        rules=[]
        for width in (small,n):
            g=gs[:,:width,:width].mean(0)
            inverse=np.linalg.inv(g+.18*np.diag(np.sqrt(g.diagonal()).clip(.008)**2))
            f=inverse@l5[side::2,:width,:width].mean(0)
            coarse=inverse@l20[side::2,:width,:width].mean(0)
            pay=inverse@prices[1-side::2,:width].mean(0)
            near=np.eye(width)+f+f@f+f@f@f
            far=(coarse+coarse@coarse)@near
            rules.append((pay,near,far))
        (s,near,far),(big,full_near,full_far)=rules
        old=np.pad(s,(0,n-small))
        trans=full_far@old-np.pad(far@s,(0,n-small))
        price=full_far@(big-old)
        first=full_near@big-np.pad(near@s,(0,n-small))
        bank.append(np.column_stack((trans,price,first)))
    bank=np.mean(bank,0)
    anchor=activity.train(end)[:,2,:]
    np.testing.assert_allclose(bank.sum(1),anchor[:,2],rtol=1e-7,atol=1e-9)
    risk=np.einsum("fj,fg,gj->j",bank,gs.mean(0),bank)
    return bank/np.sqrt(risk.clip(1e-12))


def views(raw,grid,code,bank):
    x=activity.state(raw,grid,code)
    z=x@bank
    return z-z.mean(1,keepdims=True)


if __name__=="__main__":
    rel.main(OUT,ARMS,train,views,rolling_reference=REFERENCE)
