"""Replay both releases on feature-only inputs, then evaluate the sealed predictions."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import polars as pl


def score(rows):
    p, y = map(np.concatenate, zip(*rows))
    corr = [float(np.corrcoef(p[:, h], y[:, h])[0, 1]) for h in range(3)]
    return dict(
        correlations=corr, score=float(np.dot(corr, [0.2, 0.3, 0.5])), rows=len(p)
    )


def run(args, scratch):
    os.environ["TULIN_RUN_DIR"] = str(args.run_dir.resolve())
    prior = str(args.prior_src.resolve())
    sys.path.insert(1, prior)
    import torch
    import replay_session_experts_16d as session
    from released_model import SessionRefinement
    from refinement import FieldRefinement

    torch.set_num_threads(1)
    observations = scratch / "observations.parquet"
    if not args.base_replay:
        features = pl.scan_parquet(args.data).filter(
            pl.col("date").is_in(list(session.PREFIX))
        )
        features.select("date", "time", "code", *session.FIELDS).sink_parquet(
            observations
        )
        command = (
            "import torch,sys; from pathlib import Path; torch.set_num_threads(2); "
            "import replay_session_experts_16d as r; r.original(Path(sys.argv[1]),Path(sys.argv[2]))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((prior, env.get("PYTHONPATH", "")))
        subprocess.run(
            [sys.executable, "-c", command, str(scratch), str(args.base_src.resolve())],
            env=env,
            cwd=scratch,
            check=True,
        )
    previous = SessionRefinement()
    model = FieldRefinement()
    rows = []
    base_rows = []
    daily = []
    for day, prefix in session.PREFIX.items():
        print(
            f"Replaying validation day {day + 1} and its {prefix}-tick prefix",
            flush=True,
        )
        raw = session.features(observations, day)
        with np.load(scratch / f"base_{day}.npz") as b:
            old = previous.predict(raw, b["full"], b["opening"])
            pred = model.predict(raw, old)
            before = previous.predict(raw[:prefix], b["prefix"], b["opening_prefix"])
            prefix_pred = model.predict(raw[:prefix], before)
        err = float(np.abs(pred[:prefix] - prefix_pred).max())
        if err >= 1e-4:
            raise ValueError(f"prefix mismatch on day {day + 1}: {err}")
        truth = (
            pl.scan_parquet(args.data)
            .filter(pl.col("date") == day)
            .sort("time", "code")
            .select("y0", "y1", "y2")
            .collect()
            .to_numpy()
        )
        rows.append((pred.reshape(-1, 3), truth))
        base_rows.append((old.reshape(-1, 3), truth))
        daily.append(dict(day=day + 1, prefix_error=err, **score(rows[-1:])))
        print(daily[-1], flush=True)
    final = score(rows)
    base = score(base_rows)
    if final["rows"] != 936250:
        raise ValueError(f"Unexpected validation count {final['rows']}")
    result = dict(
        previous=base, pooled=final, gain=final["score"] - base["score"], daily=daily
    )
    (args.run_dir / "raw_replay.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prior-src", type=Path, required=True)
    p.add_argument(
        "--base-src",
        type=Path,
        default=Path(__file__).resolve().parent / "base",
        help="Base predictor source; defaults to the adapter for the previous release training outputs",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--base-replay",
        type=Path,
        help="Reuse raw base replay files produced by the same base release on this dataset",
    )
    args = p.parse_args()
    if args.base_replay:
        run(args, args.base_replay.resolve())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(args, Path(tmp))
