"""Analysis of one spatial snapshot (see run_grid_simulation.py's
--snapshot-every / --snapshot-days), covering three of the paper's own
claims that raw score/purity numbers alone can't verify:

- Mixing zones: "borders between regions usually have lower behavioral
  scores resulting from crosses between incompatible organisms." Checks
  this directly -- a cell is a border cell if any of its 8 Moore
  neighbors' score differs by more than --threshold -- and then goes
  further than score alone: is a border actually a meeting of
  genetically/ancestrally different types (higher genome Hamming
  distance, more lineage_id mismatches), or just a score gap?
- Signal honesty ("-12's discovered ways to mislead"): is speech
  concentrated in stimulus pairs where a predator is actually present
  (honest/informative), or roughly uniform including the no-danger case
  (indiscriminate/deceptive)?
- The "cautious communicator" claim (26's mostly play safe, but rely on
  hearing specifically in the Left Pred/Right Pred case): does hearing
  something actually change the odds of moving, and is that effect
  concentrated at stimulus pair index 4 (Pred, Pred) rather than spread
  evenly across all 9?

Usage: python analyze_snapshot.py results/case2/snapshots/day005000.npz
"""

import argparse

import numpy as np

from altruism.grid import COMPASS_OFFSETS
from altruism.world import STIM_PAIRS

_NO_PRED_STIM_INDICES = [i for i, (left, right) in enumerate(STIM_PAIRS) if left != "Pred" and right != "Pred"]
_PRED_PRESENT_STIM_INDICES = [i for i, (left, right) in enumerate(STIM_PAIRS) if left == "Pred" or right == "Pred"]


def _mean_or_nan(a: np.ndarray) -> float:
    return float(a.mean()) if a.size else float("nan")


def border_mask(score: np.ndarray, threshold: float) -> np.ndarray:
    """True at cell (r, c) if any of its 8 Moore neighbors' score differs
    from its own by more than `threshold` (torus wraparound, same 8
    compass directions the wind mechanism itself uses)."""
    is_border = np.zeros_like(score, dtype=bool)
    for row_offset, col_offset in COMPASS_OFFSETS:
        neighbor = np.roll(np.roll(score, -row_offset, axis=0), -col_offset, axis=1)
        is_border |= np.abs(neighbor - score) > threshold
    return is_border


def border_genetic_distance(
    dominant_genome: np.ndarray, dominant_lineage_id: np.ndarray, score: np.ndarray, threshold: float
) -> dict:
    """For every (cell, neighbor) pair across all 8 Moore directions,
    split into "border crossings" (score differs by more than
    threshold) vs. "non-border" pairs, and compare genome Hamming
    distance and lineage_id mismatch rate between the two groups --
    direct evidence for whether a border is a meeting of genetically/
    ancestrally different types, or merely a score gap between two
    cells that could otherwise be closely related."""
    n_bits = dominant_genome.shape[-1]
    border_hamming, interior_hamming = [], []
    border_lineage_mismatch, interior_lineage_mismatch = [], []
    for row_offset, col_offset in COMPASS_OFFSETS:
        neighbor_score = np.roll(np.roll(score, -row_offset, axis=0), -col_offset, axis=1)
        neighbor_genome = np.roll(np.roll(dominant_genome, -row_offset, axis=0), -col_offset, axis=1)
        neighbor_lineage = np.roll(np.roll(dominant_lineage_id, -row_offset, axis=0), -col_offset, axis=1)

        is_crossing = np.abs(neighbor_score - score) > threshold
        hamming = (dominant_genome != neighbor_genome).sum(axis=-1) / n_bits
        lineage_mismatch = dominant_lineage_id != neighbor_lineage

        border_hamming.append(hamming[is_crossing])
        interior_hamming.append(hamming[~is_crossing])
        border_lineage_mismatch.append(lineage_mismatch[is_crossing])
        interior_lineage_mismatch.append(lineage_mismatch[~is_crossing])

    border_hamming = np.concatenate(border_hamming)
    interior_hamming = np.concatenate(interior_hamming)
    border_lineage_mismatch = np.concatenate(border_lineage_mismatch)
    interior_lineage_mismatch = np.concatenate(interior_lineage_mismatch)

    return {
        "border_mean_hamming": _mean_or_nan(border_hamming),
        "interior_mean_hamming": _mean_or_nan(interior_hamming),
        "border_lineage_mismatch_rate": _mean_or_nan(border_lineage_mismatch),
        "interior_lineage_mismatch_rate": _mean_or_nan(interior_lineage_mismatch),
    }


def signal_informativeness(speech_by_stimulus: np.ndarray) -> np.ndarray:
    """Per cell: mean speech during Pred-present stimulus pairs, divided
    by mean speech during no-Pred pairs (+ epsilon). Much greater than 1
    = honest/informative signaling, concentrated when a predator is
    actually present; close to 1 = indiscriminate calling regardless of
    danger -- the signal-honesty half of the paper's deception claim
    (speech *volume* alone, already logged elsewhere, can't tell these
    apart)."""
    pred_present = speech_by_stimulus[..., _PRED_PRESENT_STIM_INDICES].mean(axis=-1)
    no_pred = speech_by_stimulus[..., _NO_PRED_STIM_INDICES].mean(axis=-1)
    return pred_present / (no_pred + 1e-9)


def hearing_response_summary(hearing_response: np.ndarray) -> dict:
    """Grid-wide, per stimulus pair: P(moved | heard something) -
    P(moved | heard nothing) -- how much hearing something changes the
    odds of moving, in that specific stimulus context. The paper's
    "cautious communicator" claim (relies on hearing only for its Left
    Pred/Right Pred case, stimulus index 4) predicts this should be
    much larger at index 4 than at the other 8 indices.

    Returns nan (not 0) for a stimulus pair where "heard" or "unheard"
    was never observed at all -- that conditional probability is
    genuinely undefined, not zero; also returns the raw heard/unheard
    totals so a caller can see how much data backed each number."""
    n_stim_pairs = hearing_response.shape[-2]
    totals = hearing_response.reshape(-1, n_stim_pairs, 4).sum(axis=0)
    heard_moved, heard_stayed, unheard_moved, unheard_stayed = (totals[:, i] for i in range(4))
    heard_total = heard_moved + heard_stayed
    unheard_total = unheard_moved + unheard_stayed

    p_moved_given_heard = np.full(n_stim_pairs, np.nan)
    np.divide(heard_moved, heard_total, out=p_moved_given_heard, where=heard_total > 0)
    p_moved_given_unheard = np.full(n_stim_pairs, np.nan)
    np.divide(unheard_moved, unheard_total, out=p_moved_given_unheard, where=unheard_total > 0)

    return {
        "diff": p_moved_given_heard - p_moved_given_unheard,
        "p_moved_given_heard": p_moved_given_heard,
        "p_moved_given_unheard": p_moved_given_unheard,
        "heard_total": heard_total,
        "unheard_total": unheard_total,
    }


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
    if n_border == 0 or n_interior == 0:
        print("(all cells fall in one group -- border/interior comparisons below will show nan for the other)")
    print()
    print(f"{'':14}{'mean score':>14}{'mean purity':>14}")
    print(f"{'Border':14}{_mean_or_nan(score[is_border]):>14.2f}{_mean_or_nan(purity[is_border]):>14.3f}")
    print(f"{'Interior':14}{_mean_or_nan(score[~is_border]):>14.2f}{_mean_or_nan(purity[~is_border]):>14.3f}")

    if "dominant_genome" in data and "dominant_lineage_id" in data:
        genetics = border_genetic_distance(data["dominant_genome"], data["dominant_lineage_id"], score, args.threshold)
        print()
        print("Mixing zones -- is a border genetically/ancestrally different, or just a score gap?")
        print(f"{'':14}{'mean Hamming':>14}{'lineage mismatch':>18}")
        print(f"{'Border':14}{genetics['border_mean_hamming']:>14.3f}{genetics['border_lineage_mismatch_rate']:>18.3f}")
        print(f"{'Interior':14}{genetics['interior_mean_hamming']:>14.3f}{genetics['interior_lineage_mismatch_rate']:>18.3f}")

    if "speech_by_stimulus" in data:
        informativeness = signal_informativeness(data["speech_by_stimulus"])
        print()
        print("Signal honesty (Pred-present speech / no-Pred speech; >>1 = honest, ~1 = indiscriminate/deceptive):")
        print(f"  grid mean: {_mean_or_nan(informativeness):.2f}   border: {_mean_or_nan(informativeness[is_border]):.2f}"
              f"   interior: {_mean_or_nan(informativeness[~is_border]):.2f}")

    if "hearing_response" in data:
        response = hearing_response_summary(data["hearing_response"])
        print()
        print("Cautious-communicator check: P(moved | heard) - P(moved | didn't hear), per stimulus pair")
        print("(paper's claim predicts this concentrated at (Pred, Pred), index 4; 'n/a' = never observed)")
        for i, (left, right) in enumerate(STIM_PAIRS):
            marker = "  <-- Left Pred/Right Pred" if i == 4 else ""
            diff = response["diff"][i]
            diff_str = f"{diff:+.3f}" if not np.isnan(diff) else "  n/a"
            heard_n, unheard_n = int(response["heard_total"][i]), int(response["unheard_total"][i])
            print(f"  [{i}] ({left}, {right}): {diff_str}  (n_heard={heard_n}, n_unheard={unheard_n}){marker}")


if __name__ == "__main__":
    main()
