import numpy as np

from analyze_snapshot import (
    _NO_PRED_STIM_INDICES,
    _PRED_PRESENT_STIM_INDICES,
    border_genetic_distance,
    border_mask,
    hearing_response_summary,
    signal_informativeness,
)
from altruism.world import N_STIM_PAIRS


def test_border_mask_and_genetic_distance_split_correctly():
    """A 6-row grid with two 3-row blocks (rows 0-2 score 100, rows 3-5
    score 0), block A given all-zero genomes + lineage_id 1, block B
    given all-one genomes + lineage_id 2. Rows 1 and 4 (block interiors)
    should be flagged interior; rows 0, 2, 3, 5 (each touching the other
    block, including via torus wraparound between row 5 and row 0)
    should be flagged border. Cross-block genome Hamming distance is
    exactly 1.0 (all-zero vs all-one differs at every bit) and lineage
    mismatch is exactly 1.0; same-block pairs are exactly 0.0 for both --
    a fully deterministic expected outcome, not a statistical one."""
    grid_size = 6
    score = np.zeros((grid_size, grid_size))
    score[0:3, :] = 100.0
    score[3:6, :] = 0.0

    dominant_genome = np.zeros((grid_size, grid_size, 448), dtype=np.uint8)
    dominant_genome[3:6, :, :] = 1

    dominant_lineage_id = np.zeros((grid_size, grid_size), dtype=np.int64)
    dominant_lineage_id[0:3, :] = 1
    dominant_lineage_id[3:6, :] = 2

    is_border = border_mask(score, threshold=5.0)
    assert np.all(is_border[[0, 2, 3, 5], :])
    assert np.all(~is_border[[1, 4], :])

    genetics = border_genetic_distance(dominant_genome, dominant_lineage_id, score, threshold=5.0)
    assert genetics["border_mean_hamming"] == 1.0
    assert genetics["interior_mean_hamming"] == 0.0
    assert genetics["border_lineage_mismatch_rate"] == 1.0
    assert genetics["interior_lineage_mismatch_rate"] == 0.0


def test_signal_informativeness_exact_ratio():
    """A single cell's speech_by_stimulus with every Pred-present index
    (per the module's own _PRED_PRESENT_STIM_INDICES) set to 10 and
    every no-Pred index (_NO_PRED_STIM_INDICES) set to 2 must yield
    exactly 10/2 = 5.0 -- both index sets partition all 9 stimulus pairs
    (Pred-present = contains Pred on either side; no-Pred = neither side
    is Pred), so this exercises every index exactly once."""
    assert sorted(_PRED_PRESENT_STIM_INDICES + _NO_PRED_STIM_INDICES) == list(range(N_STIM_PAIRS))
    speech_by_stimulus = np.zeros((1, 1, N_STIM_PAIRS))
    speech_by_stimulus[0, 0, _PRED_PRESENT_STIM_INDICES] = 10.0
    speech_by_stimulus[0, 0, _NO_PRED_STIM_INDICES] = 2.0

    result = signal_informativeness(speech_by_stimulus)
    assert result.shape == (1, 1)
    assert np.isclose(result[0, 0], 5.0)


def test_hearing_response_summary_nan_when_a_condition_is_never_observed():
    """A stimulus pair with zero 'heard' observations must report
    P(moved | heard) as nan, not a fabricated 0 or 1 from an epsilon
    hack; likewise zero 'unheard' observations must report P(moved |
    unheard) as nan. A pair with both conditions observed must compute
    the real difference exactly."""
    hearing_response = np.zeros((1, 1, 1, 3, 4))
    # stim 0: all unheard, all moved -- P(moved|heard) undefined
    hearing_response[..., 0, 2] = 4
    # stim 1: all heard, all moved -- P(moved|unheard) undefined
    hearing_response[..., 1, 0] = 4
    # stim 2: 3 heard-and-moved + 1 heard-and-stayed; 2 unheard-and-moved + 2 unheard-and-stayed
    hearing_response[..., 2, 0] = 3
    hearing_response[..., 2, 1] = 1
    hearing_response[..., 2, 2] = 2
    hearing_response[..., 2, 3] = 2

    result = hearing_response_summary(hearing_response)

    assert np.isnan(result["p_moved_given_heard"][0])
    assert not np.isnan(result["p_moved_given_unheard"][0])
    assert result["p_moved_given_unheard"][0] == 1.0
    assert np.isnan(result["diff"][0])

    assert not np.isnan(result["p_moved_given_heard"][1])
    assert result["p_moved_given_heard"][1] == 1.0
    assert np.isnan(result["p_moved_given_unheard"][1])
    assert np.isnan(result["diff"][1])

    assert result["p_moved_given_heard"][2] == 3 / 4
    assert result["p_moved_given_unheard"][2] == 2 / 4
    assert result["diff"][2] == 3 / 4 - 2 / 4
    assert result["heard_total"][2] == 4
    assert result["unheard_total"][2] == 4
