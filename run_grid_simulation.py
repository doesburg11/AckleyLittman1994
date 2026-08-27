"""CLI entry point for the global level (Section 2.3): runs a GridWorld
for a number of days, logging one CSV row per day with the max and mean
per-cell average behavioral score across the grid -- the same two curves
the paper's own Figure 2/3/4 plot.
"""

import argparse
import csv
from pathlib import Path

from altruism.grid import GRID_SIZE_DEFAULT, GridWorld


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
    args = parser.parse_args()

    world = GridWorld(
        seed=args.seed,
        grid_size=args.grid_size,
        wind_period=args.wind_period,
        festival_period=args.festival_period,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grid_seed{args.seed}_size{args.grid_size}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "max_cell_avg_score", "mean_cell_avg_score"])

        for day in range(1, args.days + 1):
            scores = world.run_day()
            if day % args.log_every == 0 or day == args.days:
                cell_avg = scores.mean(axis=2)  # (grid_size, grid_size)
                writer.writerow([day, float(cell_avg.max()), float(cell_avg.mean())])
                f.flush()
                print(f"day {day}/{args.days}: max={cell_avg.max():.2f} mean={cell_avg.mean():.2f}")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
