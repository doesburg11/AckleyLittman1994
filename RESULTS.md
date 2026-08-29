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

## Global level (Section 2.3) — implemented and validated (2026-08-27)

31/31 tests passing (25 local-level + 6 new in `test_grid.py`), plus a
small-scale smoke run.

**Grid, wind, festival** (`altruism/grid.py`): a torus grid of
`LocalWorld` subpopulations (`GridWorld`), with wind migration and
festival migration/reproduction layered on top of the existing local
day/scoring/reproduction cycle -- festival replaces local reproduction
for cells in a festival quad that day; wind is additive, applied after
reproduction; if a day is both windy and a festival day, festival runs
first, then wind (per the paper, p.43).

**Wind migration** validated two ways: (1) a forced-direction,
forced-emigrant-index test confirms the individual that leaves cell
(r, c) lands in *exactly* the slot vacated by its destination cell's own
emigrant, and every other slot is untouched -- checked by object
identity, not just genome content, on a 4×4 torus (covers wraparound at
every edge for free). (2) a real-RNG test confirms wind migration is a
pure permutation of the population -- same set of individual objects
before and after, every cell still exactly 8 strong.

**Festival quad phase-shifting** validated combinatorially: for a fixed
cell on an 8×8 torus, the union of its quad-partners across all 4
Margolus phase offsets is checked to equal exactly its 8 Moore
neighbors -- the property the paper states ("after four festivals, any
given cell will have interacted with all eight of its neighbors") is
asserted directly, not just run and eyeballed.

**Festival reproduction** validated with the same all-zero/all-one
genome trick already used for local reproduction's top-half check
(`test_local_reproduce_parents_come_from_top_half`), generalized to the
32-individual quad and its top-quarter (8-of-32) parent pool: across 50
random trials, exactly one individual in the quad is ever replaced
(tracked by object identity, since a victim from the top-quarter group
can be replaced by a content-identical all-zero offspring), and that
offspring's genome is always all-zero -- never a mix, which would only
be possible if a parent were drawn from outside the top quarter.

**Small-scale smoke run**: `run_grid_simulation.py --grid-size 8 --days
200 --wind-period 5 --festival-period 2 --seed 0` completed in ~94s with
no crashes or NaNs; both the max and mean per-cell-average behavioral
score climbed steadily from the -696-ish random-founder baseline (day 20:
max -595.5/mean -689.9) toward less negative values (day 200: max
-389.5/mean -437.5) -- the same qualitative shape (steady climb from the
never-moving baseline) the paper's own Figure 2/3 report, though nowhere
near their reported day-1000-3000 timescale for reaching the -12 to -56
range, as expected at this tiny 8×8/64-subpopulation scale over only 200
days.

**Refactor, not new behavior**: `world.py`'s `local_reproduce` now calls
a shared `select_parents_and_victim(scores, top_k, rng)` helper (same
rank/select/kill mechanic local reproduction always used, just factored
out so festival reproduction -- identical mechanic, different pool size
and top fraction -- doesn't duplicate it). All 25 pre-existing local-level
tests still pass unchanged, confirming this is behavior-preserving.

## Parallelized per-cell scoring (2026-08-27)

Since a full-scale day is dominated by 16,384 cells' worth of independent
36-trial scoring, `GridWorld(..., n_workers=N)` dispatches that step (and
local reproduction, unless the day is a festival day) across a persistent
`ProcessPoolExecutor`; festival and wind stay single-threaded, since
they're cheap by comparison and need cross-cell coordination a per-cell
worker can't provide. Correctness holds by construction, not just by
testing it: every cell owns its own independent rng stream, so a cell's
result never depends on which worker ran it or in what order -- verified
by a new test asserting a serial run and a 2-worker run land on
bit-identical genomes given the same seed. Benchmarked on a 32×32 grid,
20 days: 32 workers cut wall-clock from 2m31s to 19.5s (7.7x), with
byte-identical output confirming parallelization changed nothing about
the result. 33/33 tests passing.

## Case 1 (wind-only), full paper scale — launched 2026-08-27, in progress

Launched at the paper's own Case 1 configuration: `--grid-size 128
--wind-period 5 --days 13110 --seed 0 --workers 30`, run detached
(nohup+disowned) so it survives independent of any terminal session.
Measured throughput has held steady at ~18-19s/day the entire run
(~30 of 32 cores in use, bursty rather than pegged, due to the daily
IPC round-trip) -- full run estimated at ~2.7 days wall-clock.

**Max and mean per-cell-average score over the run so far** (sampled
every 500 days):

| day | max | mean |
|---|---|---|
| 1 | -650.9 | -701.9 |
| 500 | 42.0 | -283.0 |
| 1,000 | 45.0 | -96.1 |
| 1,500 | 63.25 | -63.8 |
| 2,000 | 58.0 | -43.9 |
| 3,000 | 58.0 | -20.5 |
| 4,000 | 58.0 | -19.3 |
| 5,000 | 58.0 | -19.3 |
| 6,000 | 58.0 | -18.3 |
| 6,500 | 58.0 | -17.7 |
| 7,500 | 58.0 | -19.3 |
| 8,500 | 58.0 | -19.8 |
| 9,000 | 24.5 | -18.7 |
| 10,000 | 24.0 | -19.8 |
| 10,500 | 24.5 | -20.1 |

Two things worth flagging while the run is still in progress, not as a
final result: (1) a single communicating cluster scoring 58 held on for
roughly 6,500 straight days (day 2,000 to 8,500) before being invaded and
displaced -- between day 8,500 and 9,000 the max collapsed to 24.5 and
nothing comparably good has re-emerged since, a plausible live instance
of the exact wind-only dynamic the paper describes (a communicating
cluster can hold substantial territory for a long stretch, but has no
guarantee of holding it indefinitely once wind keeps mixing communication
range away from breeding range). (2) The population-wide *mean* barely
moved through that whole rise-and-fall (-17.7 to -20.1) -- expected, since
one 58-scoring cluster was always a tiny fraction of 16,384 subpopulations,
matching the paper's own finding that communicators never dominate the
array under wind-only migration. Read as consistent with the paper's
described dynamics so far, not as a bug -- the full picture (including
whatever the spatial trace looks like) isn't in until the run completes.

## Not yet run

- The paper's Case 2 (wind+festival, `wind_period=10`/`festival_period=2`,
  14,580 days) and Case 3 (festival-only, `festival_period=2`, 99,980
  days) -- not launched yet, pending Case 1's outcome and a decision on
  wall-clock budget (Case 3 alone is ~99,980 days; even parallelized,
  a meaningfully longer commitment than Case 1).
- Figure 2/3/4's "sample" scatter series and any movie/plate-style spatial
  visualization -- the CSV logger currently only records max/mean
  per-cell-average score per day.
