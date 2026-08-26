import numpy as np

from altruism.genome import Genome
from altruism.network import Network
from altruism.units import MOVE, PRED, TO_R, TRUE


def _bits_for(value: int, n: int) -> list[int]:
    return [int(c) for c in format(value, f"0{n}b")]


def _genome_with_single_connection(source_unit, dest_unit, weight_specifier) -> Genome:
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(source_unit, 5)
    bits[24:29] = _bits_for(dest_unit, 5)
    bits[29:32] = _bits_for(weight_specifier, 3)
    return Genome(bits=bits)


def test_reset_trial_sets_true_and_initial_states():
    genome = _genome_with_single_connection(PRED, MOVE, 6)
    genome.bits[0] = 1  # initial_states bit 0 -> unit 13 (first computed unit)
    net = Network(genome)
    net.reset_trial(genome)
    assert net.values[TRUE] == 1
    assert net.values[13] == 1
    assert net.values[14] == 0


def test_pred_drives_move_through_a_single_excitatory_connection():
    """PRED -> MOVE, weight +11 (weight_specifier=6). Pred=1 should turn
    Move on; Pred=0 should leave it off (threshold is strictly > 0)."""
    genome = _genome_with_single_connection(PRED, MOVE, 6)
    net = Network(genome)
    net.reset_trial(genome)

    hearing = np.zeros(6, dtype=np.int16)
    net.react(pred=1, food=0, location="L", hearing=hearing)
    assert net.move is True

    net.reset_trial(genome)
    net.react(pred=0, food=0, location="L", hearing=hearing)
    assert net.move is False


def test_inhibitory_connection_suppresses_activation():
    """PRED -> MOVE, weight -11 (weight_specifier=1): Pred=1 should never
    turn Move on (sum is 0 or negative, never > 0)."""
    genome = _genome_with_single_connection(PRED, MOVE, 1)
    net = Network(genome)
    net.reset_trial(genome)
    hearing = np.zeros(6, dtype=np.int16)
    net.react(pred=1, food=0, location="L", hearing=hearing)
    assert net.move is False


def test_two_pass_propagation_lets_hidden_unit_reach_effector_same_step():
    """Chain PRED -> hidden(21) -> MOVE within ONE step, via two
    connections and two synchronous passes. First pass: hidden(21) turns
    on from PRED. Second pass: MOVE turns on from hidden(21)'s NEW value
    -- this only works because of the two-pass update, not one."""
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(PRED, 5)
    bits[24:29] = _bits_for(21, 5)
    bits[29:32] = _bits_for(6, 3)  # PRED -> 21, weight +11
    bits[32:37] = _bits_for(MOVE, 5)
    bits[37:40] = _bits_for(6, 3)  # 21 -> MOVE, weight +11 (mode still SOURCE, current_unit now 21)
    genome = Genome(bits=bits)

    net = Network(genome)
    net.reset_trial(genome)
    hearing = np.zeros(6, dtype=np.int16)
    net.react(pred=1, food=0, location="L", hearing=hearing)
    assert net.hidden_state[0] == 1  # unit 21 activated in pass 1
    assert net.move is True          # unit MOVE activated in pass 2, same step


def test_location_sensors_are_one_hot():
    genome = _genome_with_single_connection(PRED, MOVE, 6)
    net = Network(genome)
    net.reset_trial(genome)
    hearing = np.zeros(6, dtype=np.int16)
    net.react(pred=0, food=0, location="2", hearing=hearing)
    from altruism.units import AT_L, AT_1, AT_2, AT_R
    assert net.values[AT_2] == 1
    assert net.values[AT_L] == 0
    assert net.values[AT_1] == 0
    assert net.values[AT_R] == 0


def test_hearing_values_can_exceed_one():
    """Hearing sensors take 0..8 (sum of speech across up to 8 individuals),
    not just 0/1 -- verify the raw value is stored, not thresholded."""
    genome = _genome_with_single_connection(PRED, MOVE, 6)
    net = Network(genome)
    net.reset_trial(genome)
    hearing = np.array([8, 0, 0, 0, 0, 0], dtype=np.int16)
    net.react(pred=0, food=0, location="L", hearing=hearing)
    from altruism.units import HEAR
    assert net.values[HEAR[0]] == 8
