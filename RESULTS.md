# Validation notes — AckleyLittman1994

## Local level (Section 2.1–2.2) — implemented and validated (2026-08-26)

25/25 tests passing across `test_genome.py`, `test_network.py`, and
`test_world.py`.

**Genome and developmental process** (`genome.py`): hand-crafted genomes
verify each of the developmental process's cases individually --
source-mode connection creation with the correct weight, the
`SOURCE_MODE_SHIFT` (weight specifier 0) retargeting `current_unit`
without creating a connection, the `DEST_MODE_SHIFT` (weight specifier 7)
correctly no-op'ing when the target is an input unit (growing an incoming
connection into a unit whose value is always set from the environment
would be meaningless) and correctly switching to destination mode
otherwise.

**Network execution** (`network.py`): a hand-built two-connection chain
(PRED → hidden unit → MOVE) confirms the two-pass synchronous update
lets a hidden unit reach an effector within a single step -- this only
works because of the second pass, not the first; a single-pass
implementation would fail this test.

**Trial structure and scoring** (`world.py`): the strongest single
validation so far -- **a never-moving individual scores exactly -696**
over a full day of 36 trials, matching the paper's own stated number
(p. 43–44: "the far the most probable resulting behavior pattern is never
to move at all. Such individuals receive behavioral scores of -696")
exactly. This number falls out purely from the score-vector table
(Figure 1) and the Latin-square location-assignment scheme (which
guarantees every individual visits each of the four locations exactly
once per stimulus pair, regardless of RNG) -- nothing was tuned to hit
it, and getting the wrong score-vector signs, the wrong stimulus-pair
enumeration, or the wrong location-coverage scheme would not have
produced this exact number by chance. Local (rank-based) reproduction
is also tested directly for top-half-only parentage and exactly-one
replacement per day.

## Known interpretive choices, documented rather than assumed

- **Which units count as "input" for the destination-mode-shift guard**:
  the paper's guard ("if the connection unit is a sensor unit, do
  nothing") is applied here to units 0–12 (TRUE plus the 12 proper
  sensors), not just the 12 sensors narrowly, since growing an incoming
  connection into TRUE would be equally meaningless (its value is also
  fixed, never computed).
- **Crossover mechanics**: "parameterized uniform crossover... crossing
  probability of 0.05 per byte... impossible to cross more than once in a
  single byte" is implemented as a per-byte (not sub-byte-bit-position)
  switch, since Spears & De Jong (1991) -- the paper this cites for the
  exact mechanism -- isn't available here. This satisfies the stated
  invariant by construction; it's a documented simplification, not
  confirmed to match their exact bit-level implementation.
- **Behavioral scoring when both L and R carry a stimulus in the same
  trial**: the paper states "add points from each applicable vector,"
  which this implementation takes literally (sum both sides' vectors at
  the final position) -- consistent with the -696 validation, which
  exercises exactly this combination logic across all 9 stimulus pairs.

## Not yet built or validated

Everything at the "global level" (Section 2.3): the population grid, wind
migration, and festival migration/reproduction with phase-shifted quads.
No comparative run (their Case 1/2/3 wind-only / wind+festival /
strictly-festival designs) has been attempted at any scale yet.
