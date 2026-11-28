"""Price the part of omitted book fields not predictable from present and earlier core observations."""
import numpy as np
import experiment_dense_hidden_payoff_memory_v1 as delayed
import experiment_dense_complementary_book_fields_v1 as fields

memory = fields.memory
ref = fields.ref
OUT = ref.base.ROOT/"artifacts/dense_unexpected_fields_v1"
ARMS = ("unanticipated_fields","future_field_news","reconstructed_fields")
prepare_inputs = fields.book_inputs


def train(history, *, payoff_transform=None, stock_fraction=.5, stock_output=False, decompose=False, teacher_decoder=False, teacher_probes=None):
    stats = []
    delays = []
    for d in range(len(history)):
        with np.load(fields.OUT/f"moments_{d:02d}.npz") as f:
            stats.append((f["cov"],f["pay"],f["variance"]))
        delays.append(np.load(delayed.OUT/f"payoffs_{d:02d}.npy"))
    grams,payoffs,variance = (np.stack(a) for a in zip(*stats))
    delays = np.stack(delays)
    sd = np.sqrt(grams[:,1,0].mean(0).diagonal()).clip(.006)
    ss = np.sqrt(grams[:,0,0].mean(0).diagonal()).clip(.03)
    core = len(sd)-len(fields.COLUMNS)
    scale = np.sqrt(variance[:,1].mean()/variance[:,0].mean())
    def blend(stock,market):
        if stock_fraction==.5:return .5*(stock+market)
        return stock_fraction*stock+(1-stock_fraction)*market
    cov = blend(grams[:,0]/np.outer(ss,ss),grams[:,1]/np.outer(sd,sd))
    pay = blend(payoffs[:,0]*scale/ss,payoffs[:,1]/sd)
    pastpay = blend(delays[:,0,:,0]*scale/ss[:core],delays[:,1,:,0]/sd[:core])
    coefficients = []
    for fold in (0,1):
        eve = slice(fold,None,2)
        odd = slice(1-fold,None,2)
        k = cov[eve].mean(0)
        def covariance(left,right):
            return k[right-left] if right>=left else k[left-right].T
        history_times = (0,-1,-2)
        gram = np.block([[covariance(a,b)[:core,:core] for b in history_times] for a in history_times])
        to_field = np.concatenate([covariance(a,0)[:core,core:] for a in history_times])
        predict = np.linalg.solve(gram+.17*np.eye(len(gram)),to_field)
        pred_pay = np.r_[pay[odd,:core].mean(0),pastpay[odd].mean(0).ravel()]
        innovation = k[0,core:,core:]-predict.T@to_field-to_field.T@predict+predict.T@gram@predict
        gradient = pay[odd,core:].mean(0)-predict.T@pred_pay
        reward = np.linalg.solve(innovation+.29*np.eye(len(innovation)),gradient)
        if teacher_decoder:
            if teacher_probes is None:
                coefficients.append(np.r_[reward,-predict@reward])
            else:
                eig,vec=np.linalg.eigh((innovation+innovation.T)*.5)
                probes=vec@((vec.T@teacher_probes)/np.sqrt(eig.clip(0)+.29)[:,None])
                decoder=np.column_stack((reward,probes))
                coefficients.append(np.vstack((decoder,-predict@decoder)))
            continue
        if decompose:
            parts=np.zeros((core,4))
            reconstruction=(predict@reward).reshape(3,core)
            for future in (1,2):
                parts[:,0]+=covariance(0,future)[:core,core:]@reward
                for j,t in enumerate(history_times):
                    parts[:,1+j]-=covariance(0,future+t)[:core,:core]@reconstruction[j]
            coefficients.append(np.linalg.solve(k[0,:core,:core]+.22*np.eye(core),parts)/sd[:core,None])
            continue
        if payoff_transform is not None:
            alternative=payoff_transform(innovation,gradient,reward)
            changed=np.zeros((core,alternative.shape[1]))
            for future in (1,2):
                expect=np.concatenate([covariance(0,future+t)[:core,:core] for t in history_times],axis=1)
                future_cov=covariance(0,future)[:core,core:]-expect@predict
                changed+=np.linalg.solve(k[0,:core,:core]+.22*np.eye(core),future_cov@alternative)
            output_scale=ss[:core,None]*scale if stock_output else sd[:core,None]
            coefficients.append(changed/output_scale)
            continue
        weights = np.zeros((3,len(sd),len(ARMS)))
        # Actual innovation and its predictable counterpart can have independent forecasting skill.
        reconstructed = (predict@reward).reshape(3,core)/sd[:core]
        weights[:,:core,0] = -reconstructed
        weights[0,core:,0] = reward/sd[core:]
        weights[:,:core,2] = reconstructed
        for future in (1,2):
            expect = np.concatenate([covariance(0,future+t)[:core,:core] for t in history_times],axis=1)
            future_cov = covariance(0,future)[:core,core:]-expect@predict
            gain = np.linalg.solve(k[0,:core,:core]+.22*np.eye(core), future_cov@reward)
            weights[0,:core,1] += gain/sd[:core]
        coefficients.append(weights)
    return dict(bank=np.mean(coefficients,0))


def views(state,grid,model):
    result = np.zeros((len(grid),len(ARMS)))
    for a,b in fields.parent.sessions(grid):
        t = np.arange(a+40,b)
        for j in range(3):
            result[t] += state[t-j*20]@model["bank"][j]
    return result


if __name__ == "__main__":
    memory.runner.run(OUT,ARMS,train,views,prior_head=memory,prepare_inputs=fields.book_inputs,keep_oof=True)
