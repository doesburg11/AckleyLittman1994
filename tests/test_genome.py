import numpy as np
import pytest

from altruism.genome import (
    GENOME_BITS,
    N_CONNECTION_GROUPS,
    N_INITIAL_STATE_BITS,
    N_PSEUDO_BITS,
    N_SYNAPTIC_BITS,
    Genome,
    crossover,
    develop,
    founder_genome,
    mutate,
)
from altruism.units import MOVE, PRED


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_genome_bit_budget_matches_paper():
    assert GENOME_BITS == 448
    assert N_INITIAL_STATE_BITS == 19
    assert N_SYNAPTIC_BITS == 405
    assert N_PSEUDO_BITS == 24


def test_founder_genome_shape(rng):
    g = founder_genome(rng)
    assert g.bits.shape == (448,)
    assert set(np.unique(g.bits)).issubset({0, 1})


def test_initial_states_is_19_bits(rng):
    g = founder_genome(rng)
    assert g.initial_states.shape == (19,)


def test_initial_source_range(rng):
    for _ in range(50):
        g = founder_genome(rng)
        assert 0 <= g.initial_source <= 31


def test_connection_groups_count(rng):
    g = founder_genome(rng)
    groups = list(g.connection_groups())
    assert len(groups) == N_CONNECTION_GROUPS
    for unit_index, weight_specifier in groups:
        assert 0 <= unit_index <= 31
        assert 0 <= weight_specifier <= 7


def _bits_for(value: int, n: int) -> list[int]:
    return [int(c) for c in format(value, f"0{n}b")]


def test_develop_creates_a_source_mode_connection_with_correct_weight():
    """Craft a genome by hand: initial source = unit 1 (PRED), first
    connection group targets unit 19 (MOVE) with weight_specifier=6
    (weight +11). Every other group is all-zero (weight_specifier=0,
    a harmless repeated source-mode-shift to TRUE). Should yield exactly
    one connection: PRED -> MOVE, weight +11."""
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(PRED, 5)          # initial source = PRED (unit 1)
    bits[24:29] = _bits_for(MOVE, 5)          # first group's connection-unit = MOVE (19)
    bits[29:32] = _bits_for(6, 3)              # weight_specifier = 6 -> weight +11
    genome = Genome(bits=bits)

    connections = develop(genome)
    assert connections == [(PRED, MOVE, 11)]


def test_develop_source_mode_shift_changes_current_unit_without_connecting():
    """weight_specifier == 0 (SOURCE_MODE_SHIFT) must retarget current_unit
    and create no connection."""
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(0, 5)  # initial source = TRUE (unit 0)
    bits[24:29] = _bits_for(5, 5)   # group 1: connection-unit = 5 (@2, an input unit)
    bits[29:32] = _bits_for(0, 3)   # weight_specifier = 0 -> mode shift, no connection
    # group 2: now current_unit should be 5; create a real connection from it
    bits[32:37] = _bits_for(MOVE, 5)
    bits[37:40] = _bits_for(4, 3)  # weight_specifier 4 -> weight +3
    genome = Genome(bits=bits)

    connections = develop(genome)
    assert connections == [(5, MOVE, 3)]


def test_develop_dest_mode_shift_skips_input_units():
    """weight_specifier == 7 (DEST_MODE_SHIFT) targeting an INPUT unit
    (index < 13) must be a complete no-op -- no mode change, no
    current_unit change, no connection."""
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(MOVE, 5)  # initial source = MOVE (19, a computed unit)
    bits[24:29] = _bits_for(PRED, 5)   # group 1: connection-unit = PRED (1, an INPUT unit)
    bits[29:32] = _bits_for(7, 3)      # weight_specifier = 7 -> dest-mode-shift attempt
    # group 2: current_unit should STILL be MOVE (unchanged), mode STILL source
    bits[32:37] = _bits_for(21, 5)     # connection-unit = 21 (a hidden unit)
    bits[37:40] = _bits_for(3, 3)      # weight_specifier 3 -> weight -3
    genome = Genome(bits=bits)

    connections = develop(genome)
    # If the dest-mode-shift had incorrectly taken effect, current_unit
    # would be PRED and mode DESTINATION, giving connection (21, PRED, -3)
    # instead of (MOVE, 21, -3).
    assert connections == [(MOVE, 21, -3)]


def test_develop_dest_mode_shift_works_for_computed_units():
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(21, 5)     # initial source = hidden unit 21
    bits[24:29] = _bits_for(22, 5)      # group 1: connection-unit = 22 (hidden, computed)
    bits[29:32] = _bits_for(7, 3)       # weight_specifier = 7 -> dest-mode-shift succeeds
    # now current_unit = 22, mode = DESTINATION
    bits[32:37] = _bits_for(PRED, 5)    # group 2: connection-unit = PRED
    bits[37:40] = _bits_for(1, 3)       # weight_specifier 1 -> weight -11
    genome = Genome(bits=bits)

    connections = develop(genome)
    # DESTINATION mode: connection FROM connection_unit TO current_unit
    assert connections == [(PRED, 22, -11)]


def test_mutate_zero_rate_is_noop(rng):
    g = founder_genome(rng)
    m = mutate(g, rng, rate=0.0)
    assert np.array_equal(m.bits, g.bits)


def test_mutate_full_rate_flips_every_bit(rng):
    g = founder_genome(rng)
    m = mutate(g, rng, rate=1.0)
    assert np.array_equal(m.bits, 1 - g.bits)


def test_crossover_output_is_448_bits(rng):
    a = founder_genome(rng)
    b = founder_genome(rng)
    child = crossover(a, b, rng)
    assert child.bits.shape == (448,)


def test_crossover_every_byte_comes_from_one_parent_or_the_other(rng):
    a = founder_genome(rng)
    b = founder_genome(rng)
    child = crossover(a, b, rng)
    for i in range(56):
        byte = child.bits[i * 8:(i + 1) * 8]
        from_a = np.array_equal(byte, a.bits[i * 8:(i + 1) * 8])
        from_b = np.array_equal(byte, b.bits[i * 8:(i + 1) * 8])
        assert from_a or from_b


def test_crossover_can_draw_from_both_parents_over_many_bytes(rng):
    # very distinct parents so "which parent" is unambiguous per byte
    a = Genome(bits=np.zeros(448, dtype=np.uint8))
    b = Genome(bits=np.ones(448, dtype=np.uint8))
    saw_a = saw_b = False
    for _ in range(200):
        child = crossover(a, b, rng, byte_cross_prob=0.5)
        if child.bits.sum() < 448:
            saw_a = True
        if child.bits.sum() > 0:
            saw_b = True
    assert saw_a and saw_b
