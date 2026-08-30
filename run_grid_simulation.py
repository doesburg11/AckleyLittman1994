"""CLI entry point for the global level (Section 2.3): runs a GridWorld
for a number of days, logging one CSV row per day with:
  - max/mean per-cell average behavioral score across the grid -- the
    same two curves the paper's own Figure 2/3/4 plot.
  - 8 fixed randomly-sampled subpopulations' scores (the paper's own
    "sample" scatter series, showing spread across the array).
  - a cheap communication-activity proxy (total active speech-channel-
    bits per individual per day), both the grid-wide mean and
    specifically for whichever cell holds that day's top score, broken
    down by which of the 9 stimulus pairs was active (index 4 is the
    paper's own "Left Pred/Right Pred" case) -- so a run can be checked
    afterward for whether high scorers actually communicate more than
    average, and whether they do it constantly or only selectively.

Periodically writes a full spatial snapshot (--snapshot-every) to
{out-dir}/snapshots/day{day}.npz -- per-cell score, per-cell per-
stimulus speech, and per-cell genetic purity -- the data behind the
paper's "Plates" (which clone occupies which cell); see
analyze_snapshot.py for a border/mixing-zone read on one of these.

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
from altruism.world import N_STIM_PAIRS

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
    parser.add_argument(
        "--snapshot-every", type=int, default=0, help="Days between full spatial snapshots; 0 disables them."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grid_seed{args.seed}_size{args.grid_size}.csv"
    checkpoint_path = out_dir / f"checkpoint_seed{args.seed}_size{args.grid_size}.pkl"
    snapshot_dir = out_dir / "snapshots"
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
        print(f"Sample cells (fixed for this run): {world.sample_cells}")

    if args.snapshot_every:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    sample_cols = [f"sample_{i + 1}" for i in range(len(world.sample_cells))]
    speech_stim_cols = [f"speech_stim_{i:02d}" for i in range(N_STIM_PAIRS)]

    with world, open(out_path, csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(
                ["day", "max_cell_avg_score", "mean_cell_avg_score"]
                + sample_cols
                + ["best_cell_speech_activity", "mean_cell_speech_activity"]
                + speech_stim_cols
            )

        stopped_early = False
        for day in range(world.day + 1, args.days + 1):
            scores = world.run_day()
            if day % args.log_every == 0 or day == args.days:
                cell_avg = scores.mean(axis=2)  # (grid_size, grid_size)
                speech_avg = world.last_day_speech.mean(axis=2)  # (grid_size, grid_size)
                # grid-wide mean speech per stimulus pair: (N_STIM_PAIRS,)
                speech_stim_avg = world.last_day_speech_by_stimulus.mean(axis=(0, 1, 2))
                best_cell = np.unravel_index(np.argmax(cell_avg), cell_avg.shape)
                sample_scores = [float(cell_avg[r, c]) for r, c in world.sample_cells]

                writer.writerow(
                    [day, float(cell_avg.max()), float(cell_avg.mean())]
                    + sample_scores
                    + [float(speech_avg[best_cell]), float(speech_avg.mean())]
                    + [float(v) for v in speech_stim_avg]
                )
                f.flush()
                print(
                    f"day {day}/{args.days}: max={cell_avg.max():.2f} mean={cell_avg.mean():.2f} "
                    f"best_cell_speech={speech_avg[best_cell]:.1f} mean_speech={speech_avg.mean():.1f}"
                )

            if args.snapshot_every and day % args.snapshot_every == 0:
                snap = world.snapshot(scores)
                np.savez_compressed(
                    snapshot_dir / f"day{day:06d}.npz",
                    day=snap["day"], score=snap["score"],
                    speech_by_stimulus=snap["speech_by_stimulus"], purity=snap["purity"],
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
