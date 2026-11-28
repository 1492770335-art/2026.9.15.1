"""Fit the cumulative readouts and regenerate their chronological calibration tapes."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from replay import configure
from stages import STAGES, SELECTED


def prepare_inputs(index, m):
    # Each collector uses only days before 17. It runs before any fit that consumes its output.
    collectors = {
        1: ('proj.future','proj'), 2: ('pairs.net',), 4: ('ranks.band',),
        5: ('settled',), 8: ('curves.basket',), 9: ('graph',),
        18: ('source',), 20: ('base.panel','base.hedge','base.gls'),
        21: ('rollout',), 23: ('live.frozen.student','live.frozen'),
    }
    if index == 6:
        m.prior.OUT.mkdir(exist_ok=True)
    if index == 8:
        m.curves.OUT.mkdir(exist_ok=True)
        m.curves.basket.front.direction.stage_one()
    if index == 10:
        m.base.OUT.mkdir(exist_ok=True)
        m.base.prepare()
    for path in collectors.get(index,()):
        module = m
        for attr in path.split('.'):
            module = getattr(module,attr)
        print('Preparing',module.__name__,flush=True)
        module.collect()


def fit(index):
    import torch
    import folds
    from version_paths import ROOT
    name, _, reference = STAGES[index-1]
    model = importlib.import_module(name)
    torch.set_num_threads(1 if index in (2,4,5,6,8,9,10,11) else 2)
    model.OUT.mkdir(parents=True,exist_ok=True)
    prepare_inputs(index,model)
    if index in (1,2,4,6):
        if index==1:
            model.check_template()
        import experiment_dense_relative_path_inputs_v1 as relative
        relative.main(model.OUT,model.ARMS,model.train,model.views,with_prediction=True,
                      rolling_reference=getattr(model,'REFERENCE',None))
    elif index==3:
        model.collect()
        model.run()
    elif index==5:
        model.rolling(model.OUT,model.ARMS,model.train,model.design,model.REFERENCE)
    elif index<20:
        if hasattr(model,'collect'):
            model.collect()
        model.onset.rolling(model.OUT,model.ARMS,model.train,model.design,model.REFERENCE)
    elif index==20:
        model.rolling()
    else:
        if index==21:
            model.collect()
        model.strip.evaluation.rolling(model)
    chosen = json.loads((model.OUT/'rolling.json').read_text())['selected']
    if chosen != SELECTED[name]:
        raise ValueError(f'Rolling selection changed at {name}: {chosen}, expected {SELECTED[name]}')
    with np.load(model.OUT/'weights.npz') as saved:
        folds.store(index,model,ROOT/'artifacts'/reference,dict(saved))
    print('Finished',index,name,json.loads((model.OUT/'rolling.json').read_text())['selected'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-src',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--rolling',type=Path)
    p.add_argument('--start',type=int,default=1,choices=range(1,len(STAGES)+1))
    p.add_argument('--stop',type=int,default=len(STAGES),choices=range(1,len(STAGES)+1))
    p.add_argument('--stage',type=int,choices=range(1,len(STAGES)+1),help=argparse.SUPPRESS)
    args = p.parse_args()
    configure(args.run_dir,args.prior_src,args.rolling)
    if args.stage:
        fit(args.stage)
        return
    if args.start>args.stop:
        p.error('--start must be less than or equal to --stop')
    for i in range(args.start,args.stop+1):
        print(f'Training {i}/{len(STAGES)} {STAGES[i-1][0]}',flush=True)
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--stage',str(i)],check=True)


if __name__=='__main__':
    main()
