"""Mixing-zone/border analysis for one spatial snapshot (see
run_grid_simulation.py's --snapshot-every). The paper notes that
"borders between regions usually have lower behavioral scores resulting
from crosses between incompatible organisms" -- this checks that
directly: a cell is a border cell if any of its 8 Moore neighbors'
score differs from its own by more than --threshold, then reports
mean score and mean genetic purity for border vs. interior cells.

Usage: python analyze_snapshot.py results/case2/snapshots/day005000.npz
"""

import argparse

import numpy as np

from altruism.grid import COMPASS_OFFSETS


def border_mask(score: np.ndarray, threshold: float) -> np.ndarray:
    """True at cell (r, c) if any of its 8 Moore neighbors' score differs
    from its own by more than `threshold` (torus wraparound, same 8
    compass directions the wind mechanism itself uses)."""
    grid_size = score.shape[0]
    is_border = np.zeros_like(score, dtype=bool)
    for row_offset, col_offset in COMPASS_OFFSETS:
        neighbor = np.roll(np.roll(score, -row_offset, axis=0), -col_offset, axis=1)
        is_border |= np.abs(neighbor - score) > threshold
    return is_border


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_path", type=str)
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="Score gap to a neighbor that counts as a different clone."
    )
    args = parser.parse_args()

    data = np.load(args.snapshot_path)
    score, purity = data["score"], data["purity"]
    day = int(data["day"])

    is_border = border_mask(score, args.threshold)
    n_border, n_interior = int(is_border.sum()), int((~is_border).sum())

    print(f"Day {day} -- grid {score.shape[0]}x{score.shape[1]}, threshold={args.threshold}")
    print(f"Border cells:   {n_border:>7} ({100 * n_border / score.size:.1f}%)")
    print(f"Interior cells: {n_interior:>7} ({100 * n_interior / score.size:.1f}%)")
    print()
    print(f"{'':14}{'mean score':>14}{'mean purity':>14}")
    print(f"{'Border':14}{score[is_border].mean():>14.2f}{purity[is_border].mean():>14.3f}")
    print(f"{'Interior':14}{score[~is_border].mean():>14.2f}{purity[~is_border].mean():>14.3f}")


if __name__ == "__main__":
    main()
