"""CLI entry point for the global level (Section 2.3): runs a GridWorld
for a number of days, logging one CSV row per day with the max and mean
per-cell average behavioral score across the grid -- the same two curves
the paper's own Figure 2/3/4 plot -- plus a cheap communication-activity
proxy (total active speech-channel-bits per individual per day), both
averaged across the grid and specifically for whichever cell holds that
day's top score, so a run can be checked afterward for whether the
best-scoring cell actually communicates more than the population at
large.

Periodically checkpoints the whole GridWorld to disk (--checkpoint-every)
and always checkpoints once more at the end, whether that's a normal
finish or a SIGINT/SIGTERM (a plain `kill <pid>`) -- the currently
in-progress day is always allowed to finish first, so a checkpoint is
never taken mid-day. Resume a run with --resume: picks up the day after
the checkpoint's, appends to the existing CSV rather than overwriting it.
"""

import argparse
import csv
import os
import signal
import sys
from pathlib import Path

import numpy as np

from altruism.grid import GRID_SIZE_DEFAULT, GridWorld

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE_DEFAULT)
    parser.add_argument("--days", type=int, default=1000)
    parser.add_argument("--wind-period", type=int, default=None, help="Windy every Nth day; omit to disable wind.")
    parser.add_argument(
        "--festival-period", type=int, default=None, help="Festival every Nth day; omit to disable festivals."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1, help="Days between logged CSV rows.")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes for per-cell scoring (0 or 1 = single-threaded). Default: all detected CPUs.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=500, help="Days between checkpoint saves (also saved at the end)."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from this run's checkpoint instead of starting fresh."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grid_seed{args.seed}_size{args.grid_size}.csv"
    checkpoint_path = out_dir / f"checkpoint_seed{args.seed}_size{args.grid_size}.pkl"
    n_workers = args.workers if args.workers > 1 else None

    if args.resume:
        if not checkpoint_path.exists():
            sys.exit(f"--resume given but no checkpoint found at {checkpoint_path}")
        world = GridWorld.load_checkpoint(str(checkpoint_path), n_workers=n_workers)
        print(f"Resumed from {checkpoint_path} at day {world.day}")
        csv_mode = "a"
    else:
        world = GridWorld(
            seed=args.seed,
            grid_size=args.grid_size,
            wind_period=args.wind_period,
            festival_period=args.festival_period,
            n_workers=n_workers,
        )
        csv_mode = "w"

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    with world, open(out_path, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow([
                "day", "max_cell_avg_score", "mean_cell_avg_score",
                "best_cell_speech_activity", "mean_cell_speech_activity",
            ])

        stopped_early = False
        for day in range(world.day + 1, args.days + 1):
            scores = world.run_day()
            if day % args.log_every == 0 or day == args.days:
                cell_avg = scores.mean(axis=2)  # (grid_size, grid_size)
                speech_avg = world.last_day_speech.mean(axis=2)  # (grid_size, grid_size)
                best_cell = np.unravel_index(np.argmax(cell_avg), cell_avg.shape)
                writer.writerow([
                    day, float(cell_avg.max()), float(cell_avg.mean()),
                    float(speech_avg[best_cell]), float(speech_avg.mean()),
                ])
                f.flush()
                print(
                    f"day {day}/{args.days}: max={cell_avg.max():.2f} mean={cell_avg.mean():.2f} "
                    f"best_cell_speech={speech_avg[best_cell]:.1f} mean_speech={speech_avg.mean():.1f}"
                )

            if day % args.checkpoint_every == 0:
                world.save_checkpoint(str(checkpoint_path))

            if _stop_requested:
                stopped_early = True
                break

        world.save_checkpoint(str(checkpoint_path))

    if stopped_early:
        print(f"Stopped early at day {world.day} (checkpoint saved) -- resume with --resume")
    else:
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
