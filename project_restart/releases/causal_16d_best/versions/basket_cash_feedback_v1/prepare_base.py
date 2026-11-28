"""Regenerate the five-tick release's full and truncated upstream predictions from public features."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def run(args, scratch):
    prior = args.prior_src.resolve()
    previous = Path(__file__).resolve().parent.parent/'five_tick_field_curve_v1'
    sys.path[:0] = [str(previous),str(prior)]
    os.environ['TULIN_RUN_DIR'] = str(args.run_dir.resolve())
    import torch
    import polars as pl
    import replay_session_experts_16d as session
    from released_model import SessionRefinement
    from refinement import FieldRefinement
    from replay import score

    torch.set_num_threads(1)
    obs = scratch/'observations.parquet'
    if not args.base_replay:
        (pl.scan_parquet(args.data).filter(pl.col('date').is_in(list(session.PREFIX)))
         .select('date','time','code',*session.FIELDS).sink_parquet(obs))
        cmd = ('import sys,torch; from pathlib import Path; torch.set_num_threads(2); '
               'import replay_session_experts_16d as m; m.original(Path(sys.argv[1]),Path(sys.argv[2]))')
        env = dict(os.environ,PYTHONPATH=os.pathsep.join((str(prior),os.environ.get('PYTHONPATH',''))))
        subprocess.run([sys.executable,'-c',cmd,str(scratch),str(previous/'base')],cwd=scratch,env=env,check=True)
    pre, tail = SessionRefinement(), FieldRefinement()
    out = args.run_dir/'artifacts/dense_five_tick_verified_reference_v1'
    out.mkdir(parents=True,exist_ok=True)
    rows, hashes = [], {}
    for day, n in session.PREFIX.items():
        raw = session.features(obs,day)
        with np.load(scratch/f'base_{day}.npz') as batch:
            old = pre.predict(raw,batch['full'],batch['opening'])
            prefix = pre.predict(raw[:n],batch['prefix'],batch['opening_prefix'])
        full, partial = tail.predict(raw,old), tail.predict(raw[:n],prefix)
        np.testing.assert_allclose(full[:n],partial,rtol=0,atol=1e-4)
        np.savez(out/f'previous_{day}.npz',full=old,prefix=prefix)
        for key,pred in (('full',full),('prefix',partial)):
            path = out/f'{key}_{day}.npy'
            np.save(path,pred)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        y = (pl.scan_parquet(args.data).filter(pl.col('date')==day).sort('time','code')
             .select('y0','y1','y2').collect().to_numpy())
        rows.append((full.reshape(-1,3),y))
    stats = score(rows)
    assert stats['rows']==936250 and abs(stats['score']-.37212517118484645)<2e-8, stats
    (out/'verified.json').write_text(json.dumps(dict(commit='25a42406c7a72bff1bc099f5d85f417a4c66ec13',
                                                      pooled=stats,hashes=hashes),indent=2))
    print(stats,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prior-src','run-dir','data'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--base-replay',type=Path,help='Optional feature-only base replay from the previous release')
    args = parser.parse_args()
    if args.base_replay:
        run(args,args.base_replay.resolve())
    else:
        with tempfile.TemporaryDirectory() as scratch:
            run(args,Path(scratch))
