"""Estimate the competition's marginal pooled Pearson payoff of a tail correction from matured contracts."""
import json
import sys
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_forward_tail_probability_v1 as risk

base,ref,REFERENCE,read_p,TARGETS = risk.base,risk.ref,risk.REFERENCE,risk.read_p,risk.TARGETS
OUT = REFERENCE.parent/'dense_settled_pearson_tail_v1'
ARMS = ('cumulative_positive','cumulative_signed','recent_signed')
BLOCK = 100


def train(end):
    return np.r_[risk.audit.curve.codec(),risk.train(end)]


def realized(delta,j,code):
    return code[1+j]+code[4+j]*np.expm1(delta*code[0]/code[6])/code[0]


def marginal(gram):
    # The intercept and benchmark slope are estimated across all observed names AND times.
    # One normalized coefficient multiplies z in the original forecast units.
    c = gram[1:,1:]-np.outer(gram[0,1:],gram[0,1:])/max(gram[0,0],1)
    yy,pp,zz = c.diagonal()
    yp,yz,pz = c[0,1],c[0,2],c[1,2]
    partial = yz-yp*pz/max(pp,1e-9)
    score = partial/np.sqrt(max(yy*zz,1e-16))
    denom = max(yp/max(pp,1e-9),.15)*max(zz-pz**2/max(pp,1e-9),.12*zz,1e-9)
    return score,float(np.clip(partial/denom,-2,2))


def coefficient(blocks):
    if len(blocks)<3:
        return 0.
    total = np.sum(blocks,axis=0)
    score,alpha = marginal(total)
    leave = np.array([marginal(total-g)[0] for g in blocks])
    se2 = (len(leave)-1)*leave.var()
    reliability = score*score/(score*score+2*se2+.00005)
    return alpha*reliability


def signals(raw,grid,p,bank):
    z = risk.signals(raw,grid,p,bank[7:])[:,:,1,:]
    out = np.zeros((*p.shape[:2],3,2))
    for j,h in enumerate((20,60)):
        blocks = []
        for a,b,m,ok in event.sessions(raw,grid):
            path = m.cumsum(0)
            gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
            for lo in range(0,b-a,BLOCK):
                hi = min(lo+BLOCK,b-a)
                since_open = coefficient(blocks)
                recent = coefficient(blocks[-7:])
                out[a+lo:a+hi,:,:,j] = z[a+lo:a+hi,:,j,None]*np.array((max(0,since_open),since_open,recent))
                # Use observed settlements after the forecasts in this block. A partial
                # final block cannot alter the predictions already issued within it.
                stops = np.arange(max(h,lo),hi,5)
                if len(stops)<5:
                    continue
                issue = stops-h
                valid = gap[stops+1]==gap[issue]
                truth = realized(path[stops]-path[issue],j+1,bank[:7])
                x = np.stack((np.ones_like(truth),truth,p[a+issue,:,j+1],z[a+issue,:,j]),axis=2)[valid]
                blocks.append(x.T@x)
    return out


def audit():
    rows = {}
    for d in (8,12):
        vals = []
        for day in range(d,d+4):
            raw,grid,y = [onset.settled.load(day,k) for k in ('x','time','y')]
            obs = []
            for a,b,m,ok in event.sessions(raw,grid):
                path = m.cumsum(0)
                gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
                stops = np.arange(60,b-a,5)
                issue = stops-60
                valid = gap[stops+1]==gap[issue]
                r = realized(path[stops]-path[issue],2,risk.audit.curve.codec())
                obs.append(np.stack((r,y[a+issue,:,2]),axis=2)[valid])
            q,v = np.concatenate(obs).T
            vals.append((np.corrcoef(q,v)[0,1],float(np.quantile(abs(q-v),.99))))
        rows[d+1] = vals
    (OUT/'settlement_alignment.json').write_text(json.dumps(rows,indent=2))
    print(rows,flush=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
