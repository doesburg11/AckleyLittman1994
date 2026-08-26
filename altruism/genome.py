"""The 448-bit genome and its developmental process (Section 2.2).

Genome layout (448 bits total):
  - bits[0:19]   initial states for the 19 COMPUTED units (13..31, in
                 order), read at the start of every trial.
  - bits[19:24]  initial source: a 5-bit integer (0-31) naming the first
                 "current unit" for the developmental process.
  - bits[24:424] fifty 8-bit connection-specifier groups (400 bits): each
                 group is a 5-bit connection-unit index + 3-bit weight
                 specifier.
  - bits[424:448] 24 unused "pseudo genes" -- present so the genome has
                 its full, paper-specified length and crossover operates
                 over the whole thing, but never decoded into anything.

All bits are stored as a single uint8 numpy array of 0/1 values, MSB-first
within each field, for reproducible, spec-legible slicing.
"""

from dataclasses import dataclass

import numpy as np

from altruism.units import N_COMPUTED_UNITS, N_UNITS, is_input_unit

GENOME_BITS = 448
N_INITIAL_STATE_BITS = N_COMPUTED_UNITS  # 19
N_SOURCE_BITS = 5
N_CONNECTION_GROUPS = 50
BITS_PER_GROUP = 8  # 5-bit unit index + 3-bit weight specifier
N_SYNAPTIC_BITS = N_SOURCE_BITS + N_CONNECTION_GROUPS * BITS_PER_GROUP  # 405
N_PSEUDO_BITS = GENOME_BITS - N_INITIAL_STATE_BITS - N_SYNAPTIC_BITS  # 24
assert N_SYNAPTIC_BITS == 405
assert N_PSEUDO_BITS == 24

_INITIAL_STATE_START = 0
_SOURCE_START = N_INITIAL_STATE_BITS
_GROUPS_START = _SOURCE_START + N_SOURCE_BITS
_PSEUDO_START = _GROUPS_START + N_CONNECTION_GROUPS * BITS_PER_GROUP
assert _PSEUDO_START + N_PSEUDO_BITS == GENOME_BITS

# Weight specifier values 1-6 map to a fixed, symmetric set of connection
# weights (Figure 1's weight-specifier box). 0 and 7 are mode-shift codes,
# handled in develop(), not weights.
WEIGHT_TABLE = {1: -11, 2: -7, 3: -3, 4: 3, 5: 7, 6: 11}
SOURCE_MODE_SHIFT = 0
DEST_MODE_SHIFT = 7


def _bits_to_int(bits: np.ndarray) -> int:
    return int("".join(str(b) for b in bits), 2)


def _int_to_bits(value: int, n: int) -> np.ndarray:
    return np.array([int(c) for c in format(value, f"0{n}b")], dtype=np.uint8)


@dataclass
class Genome:
    bits: np.ndarray  # shape (448,), dtype uint8, values 0/1

    def copy(self) -> "Genome":
        return Genome(bits=self.bits.copy())

    @property
    def initial_states(self) -> np.ndarray:
        """Starting 0/1 value for each of the 19 computed units (13..31),
        in index order -- read once at the start of every trial."""
        return self.bits[_INITIAL_STATE_START:_INITIAL_STATE_START + N_INITIAL_STATE_BITS]

    @property
    def initial_source(self) -> int:
        return _bits_to_int(self.bits[_SOURCE_START:_SOURCE_START + N_SOURCE_BITS])

    def connection_groups(self):
        """Yields (connection_unit_index, weight_specifier) for the 50
        connection-specifier groups, in genome order."""
        for i in range(N_CONNECTION_GROUPS):
            start = _GROUPS_START + i * BITS_PER_GROUP
            group = self.bits[start:start + BITS_PER_GROUP]
            unit_index = _bits_to_int(group[0:5])
            weight_specifier = _bits_to_int(group[5:8])
            yield unit_index, weight_specifier


def founder_genome(rng: np.random.Generator) -> Genome:
    bits = (rng.random(GENOME_BITS) < 0.5).astype(np.uint8)
    return Genome(bits=bits)


def develop(genome: Genome) -> list[tuple[int, int, int]]:
    """Run the developmental process (Section 2.2) and return the grown
    connection list as (from_unit, to_unit, weight) triples.

    Starts in SOURCE mode with `current_unit` from the 5-bit initial
    source. Processes the 50 connection-specifier groups in order:
      - weight_specifier == 0 (SOURCE_MODE_SHIFT): current_unit :=
        connection_unit; mode := SOURCE. No connection created.
      - weight_specifier == 7 (DEST_MODE_SHIFT): if connection_unit is an
        input unit, do nothing (growing an incoming connection into a
        unit whose value is always set from the environment is
        meaningless -- see units.py::is_input_unit); otherwise
        current_unit := connection_unit; mode := DESTINATION.
      - weight_specifier in 1..6: create one connection with the
        corresponding fixed weight. In SOURCE mode: current_unit ->
        connection_unit. In DESTINATION mode: connection_unit ->
        current_unit. current_unit and mode are unchanged by this case.

    A connection whose destination happens to be an input unit (possible
    here, since this specific guard only applies to the mode-shift case)
    is still returned -- but harmless by construction, because
    `network.py` always sets input-unit values from the environment,
    never from accumulated weighted input.
    """
    current_unit = genome.initial_source % N_UNITS
    mode_source = True
    connections: list[tuple[int, int, int]] = []

    for connection_unit, weight_specifier in genome.connection_groups():
        connection_unit %= N_UNITS
        if weight_specifier == SOURCE_MODE_SHIFT:
            current_unit = connection_unit
            mode_source = True
        elif weight_specifier == DEST_MODE_SHIFT:
            if is_input_unit(connection_unit):
                continue
            current_unit = connection_unit
            mode_source = False
        else:
            weight = WEIGHT_TABLE[weight_specifier]
            if mode_source:
                connections.append((current_unit, connection_unit, weight))
            else:
                connections.append((connection_unit, current_unit, weight))

    return connections


def mutate(genome: Genome, rng: np.random.Generator, rate: float) -> Genome:
    """Flip each bit independently with probability `rate`. NOT used in
    the paper's own reported studies (they ran with mutation off entirely,
    crossover-only -- see Section 4's extended discussion of why) -- kept
    here only so a run CAN opt into mutation for exploration beyond the
    paper's own reported cases, off by default (rate=0.0)."""
    if rate <= 0.0:
        return genome.copy()
    flips = rng.random(GENOME_BITS) < rate
    bits = genome.bits.copy()
    bits[flips] = 1 - bits[flips]
    return Genome(bits=bits)


def crossover(a: Genome, b: Genome, rng: np.random.Generator, byte_cross_prob: float = 0.05) -> Genome:
    """Parameterized uniform crossover (Spears & de Jong, 1991), per byte:
    56 bytes total (448/8). For each byte, with probability
    `byte_cross_prob` a single crossover point is possible; the paper's
    own description guarantees "it is impossible to cross more than once
    in a single byte." This implementation makes the (documented,
    reasonable) simplification of switching at BYTE boundaries only, not
    sub-byte bit positions, since Spears & De Jong (1991) isn't available
    here to pin down the exact sub-byte mechanics -- byte-boundary
    switching already satisfies the paper's stated invariant by
    construction and preserves the low, per-byte crossover rate.
    """
    n_bytes = GENOME_BITS // 8
    assert GENOME_BITS % 8 == 0
    current = a if rng.random() < 0.5 else b
    out_bits = np.zeros(GENOME_BITS, dtype=np.uint8)
    for i in range(n_bytes):
        if rng.random() < byte_cross_prob:
            current = b if current is a else a
        out_bits[i * 8:(i + 1) * 8] = current.bits[i * 8:(i + 1) * 8]
    return Genome(bits=out_bits)
