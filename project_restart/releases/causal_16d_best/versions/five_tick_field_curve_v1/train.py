"""Train the additional readouts exclusively on dates 1–16."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prior-src", type=Path, required=True, help="Trained causal_16d_best/src"
    )
    p.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Previous release's daily tensor cache",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--rolling",
        type=Path,
        help="Reuse a previously generated chronological forecast directory",
    )
    args = p.parse_args()
    os.environ["TULIN_RUN_DIR"] = str(args.run_dir.resolve())
    os.environ["TULIN_CACHE"] = str(args.cache.resolve())
    os.environ["TULIN_ROLLING"] = str(
        (args.rolling or args.run_dir / "rolling").resolve()
    )
    if not args.rolling:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(args.prior_src.resolve()), env.get("PYTHONPATH", ""))
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("rolling.py"))],
            env=env,
            check=True,
        )

    import settings, short, stock, market, tape, moments, precision, nonlinear, clock_data, clock
    import runner, memory, fields, delayed, news, history_head, paired, fine

    (settings.ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    for day in range(4, 16):
        if not (settings.TAPE / f"forecast_{day:02d}.npy").is_file():
            raise FileNotFoundError(
                f"Missing chronological forecast for training day {day + 1}"
            )
    steps = [
        short.main,
        stock.main,
        market.main,
        tape.main,
        moments.main,
        lambda: moments.run(precision.OUT, precision.ARMS, precision.operators),
        nonlinear.run,
        clock_data.main,
        clock.main,
        lambda: runner.run(
            memory.OUT, memory.ARMS, memory.parameters, memory.views, keep_oof=True
        ),
        fields.collect,
        delayed.collect,
        lambda: runner.run(
            news.OUT,
            news.ARMS,
            news.train,
            news.views,
            prior_head=memory,
            prepare_inputs=fields.book_inputs,
            keep_oof=True,
        ),
        history_head.collect,
        lambda: runner.run(
            history_head.OUT,
            history_head.ARMS,
            history_head.train,
            history_head.views,
            prior_head=(memory, news),
            keep_oof=True,
        ),
        paired.collect,
        fine.collect,
        lambda: runner.run(
            fine.OUT,
            fine.ARMS,
            fine.train,
            fine.views,
            prior_head=(memory, news, history_head),
            targets=(2,),
            keep_oof=True,
            check_leave_one_out=True,
        ),
    ]
    for i, step in enumerate(steps, 1):
        print(f"Training stage {i}/{len(steps)}", flush=True)
        step()


if __name__ == "__main__":
    main()
