"""Partially remove nuisance exposure from revision crossings according to its replication across dates."""
import sys
import numpy as np
import experiment_dense_vintage_crossing_graph_v1 as graph

strip,rel,onset,REFERENCE,read_p=graph.strip,graph.rel,graph.onset,graph.REFERENCE,graph.read_p
OUT=REFERENCE.parent/'dense_vintage_crossing_shrinkage_v1'
ARMS=('half_conditioned_crossings','replicated_nuisance','agreement_across_conditioning')


def train(end):
    units,days=graph.collect(end)
    g=days.mean(0)
    n=len(g)-2
    c=graph.C
    def nuisance(row):
        return np.linalg.solve(row[:c,:c]+.008*np.diag(row.diagonal()[:c].clip(1e-7)),row[:c,c:n])
    estimates=np.stack([nuisance(v) for v in days])
    mean=estimates.mean(0)
    reliability=mean**2/(mean**2+estimates.var(0)+1e-8)
    def solve(strength):
        v=np.eye(n)[:,c:]
        v[:c]=-strength*nuisance(g)
        cov=v.T@g[:n,:n]@v
        pay=v.T@days[:,:n,n:]
        avg=pay.mean(0)
        rhs=avg**3/(avg*avg+pay.var(0)/len(days)+1e-10)
        return v@np.linalg.solve(cov+.45*np.diag(cov.diagonal().clip(1e-7))+1e-8*np.eye(len(cov)),rhs)
    return np.r_[units,np.stack([solve(v) for v in (.5,reliability,0.,1.)]).ravel()]


def signals(raw,grid,p,bank):
    weights=bank[3:].reshape(4,-1,2)
    out=np.zeros((*p.shape[:2],3,2))
    for t,peers,wt,x in graph.edge_blocks(raw,grid,p,bank[:3],1,0):
        half,rep,untreated,treated=[x@w for w in weights]
        agree=np.where(untreated*treated>0,np.sign(treated)*np.minimum(abs(untreated),abs(treated)),0.)
        for j,edge in enumerate((half,rep,agree)):
            out[t,:,j]=graph.recover(peers,wt,edge)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
