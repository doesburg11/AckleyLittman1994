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

## Case 1 (wind-only), full paper scale — completed (launched 2026-08-27, finished 2026-08-30)

Run at the paper's own Case 1 configuration: `--grid-size 128
--wind-period 5 --days 13110 --seed 0 --workers 30`, detached
(nohup+disowned). Completed cleanly in ~59 hours wall-clock (~18.8s/day
average, ~30 of 32 cores in use, bursty rather than pegged, due to the
daily IPC round-trip), no crashes.

**Four-phase structure**, from a 200-day-windowed re-analysis of the full
13,110-row log (min/max of the day-max column and mean of the day-mean
column per window, so nothing here is cherry-picked from a coarser
sample):

| phase | days | max score | mean score | character |
|---|---|---|---|---|
| 1. Rapid climb | 0-2,400 | -650 → 58 | -702 → -26.5 | fast improvement off the random-founder baseline |
| 2. First dominant clone | 2,400-6,600 (~4,200 days) | flat at 58.0, no exceptions | -26.5 → **-17.7** (best of the run) | one clone owns best-in-grid for 4,200 straight days |
| 3. Contested turnover | 6,600-8,800 (~2,200 days) | swings repeatedly through 24-66 | -18 to -19.7 | not a clean collapse -- 58 keeps reappearing and getting knocked back down by several other clone scores (40.5/44.5/49.25/24.5) before finally losing for good |
| 4. Second dominant clone | 8,800-13,110 (~4,300 days, through the end) | flat at 24.0-25.0, no exceptions | ~-19 to -20.3 | a *lower*-scoring but evidently more durable clone displaces the first and holds just as stably -- the run ends still in this regime |

The headline observation: the population's best discovered strategy
actually got *worse* in absolute score (58 → 24-25) even though its
replacement is competitively more durable under wind-only migration (it
had already outlasted the first clone's reign by the time the run ended:
4,300+ days vs. 4,200). The population-wide mean never moved far from
the -17 to -20 band across either era, and never came close to the
paper's own reported approach toward -12 -- consistent with, though not
proof of, the paper's central thesis for Case 1: under wind-only
migration a communicating-ish clone can hold substantial territory for
a long stretch, but has no guarantee of holding it indefinitely, and can
be permanently displaced by something that scores worse but doesn't pay
whatever cost the first type was paying.

**Important caveat**: this analysis is score data only. It does not
confirm the 58-clone was actually more communicative (used the alarm/
speech channels more) than the 24-25-clone that displaced it --
that requires the speech-activity logging added below, not available
for this run. Read the "communicators vs. cheaters" framing above as a
plausible, paper-consistent interpretation of the score shape, not a
directly verified claim.

## Speech-activity logging (2026-08-30)

To let a future run directly check *whether* a high-scoring clone is
actually communicating more than average -- rather than inferring it
from score shape alone, as Case 1's analysis above had to -- `LocalWorld`
now tracks `last_day_speech_activity`: total active speech-channel-bits
per individual over the day's 36 trials (max possible 6 channels x 3
steps x 36 trials = 648), refreshed as a side effect of `run_day()`.
`GridWorld` collects this into `last_day_speech` (same shape as scores),
through both the serial and parallel scoring paths. `run_grid_simulation.py`
now logs two additional CSV columns: `mean_cell_speech_activity` (grid-wide
average) and `best_cell_speech_activity` (the speech activity specifically
of whichever cell holds that day's top score) -- so a future run's CSV
alone can show whether the best-scoring cell talks more than the
population at large.

Validated with an exact check in the same style as the -696 score
validation: a hand-crafted genome with exactly one connection (TRUE ->
one speech channel, source mode) is deterministically active every step
regardless of stimuli, location, or hearing, giving an exact predicted
day-total of 108 (3 steps x 36 trials x 1 channel) -- confirmed exactly,
alongside a never-moving individual's total confirmed at exactly 0.
Parallel-vs-serial consistency for the new metric is also tested
directly. 36/36 tests passing.

## Not yet run

- The paper's Case 2 (wind+festival, `wind_period=10`/`festival_period=2`,
  14,580 days) and Case 3 (festival-only, `festival_period=2`, 99,980
  days) -- not launched yet. Now that speech-activity logging exists,
  worth deciding whether to re-run Case 1 with it enabled (to directly
  confirm or refute the communicators-vs-cheaters reading above) before
  or alongside launching Case 2/3 -- not yet decided.
- Figure 2/3/4's "sample" scatter series and any movie/plate-style spatial
  visualization.
