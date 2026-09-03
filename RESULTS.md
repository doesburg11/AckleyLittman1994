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

**What this case is actually testing.** The paper's central claim is
that kin selection can only favor costly communication if communication
range and breeding range are correlated -- but not perfectly correlated,
or the population just freezes at whatever it started with. Wind-only
migration ("anything goes," no group-cohesion mechanism at all, unlike
festival in Cases 2/3) is the crudest way to inject the imperfection
that keeps that from happening. Two questions this case is built to
answer, and the baseline Cases 2/3 get compared against:
- **Can communication even get started under the harshest disruption
  tested?** Yes -- communicating clones (the paper's "26's") repeatedly
  emerge.
- **Can it survive there?** No, not durably -- wind keeps scrambling
  who-you-hear vs. who-you're-related-to, so non-communicating
  "cheaters" ("-12's") keep re-invading before a communicating clone can
  grow large. The population mean climbs but plateaus well short of
  full communicator dominance.

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

## Checkpoint/resume (2026-08-30)

Long runs had no way to save progress, so adding one more metric to an
already-completed run meant re-running from day 1 -- exactly what
happened wanting speech data for Case 1. `GridWorld.save_checkpoint()`/
`load_checkpoint()` (pickle, atomic write via temp-file + `os.replace`)
close this: `__getstate__`/`__setstate__` exclude the unpicklable
`ProcessPoolExecutor` from the saved state without ever touching the
live object, so a checkpoint never disturbs a run in progress, and
`load_checkpoint`'s `n_workers` can differ from whatever the checkpoint
was saved with. `run_grid_simulation.py` gains `--checkpoint-every`
(periodic saves, default 500 days) and `--resume` (picks up at
`checkpoint.day + 1`, appends to the existing CSV). SIGINT/SIGTERM (a
plain `kill <pid>`) let the in-progress day finish, then save a final
checkpoint and exit cleanly -- never taken mid-day. Verified two ways:
3 new tests confirm a save/resume round trip (including changing worker
count across it) lands on bit-identical genomes and speech-activity
values as an uninterrupted run; a manual CLI smoke test killed a real
run via SIGTERM mid-flight, resumed it, and diffed the resulting CSV
against an uninterrupted reference run -- byte-identical, no duplicate
or missing rows. 39/39 tests passing.

## Full metric parity with the paper's own analysis (2026-08-30)

Asked directly whether the logging supported a *full* comparison
against the paper's own analysis -- it didn't. Five gaps, all closed:

- **Per-stimulus speech breakdown** (`LocalWorld.last_day_speech_by_stimulus`,
  shape `(8, 9)`): speech activity broken down by which of the 9
  stimulus pairs was active (index 4 is the paper's own "Left Pred/Right
  Pred" case), not just a flat daily total -- this is what would let a
  future run distinguish a clone that talks constantly from the paper's
  "cautious communicators," who mostly play the safe non-signaling
  strategy and only call in specific situations. `last_day_speech_activity`
  is now *derived* from this (`.sum(axis=1)`) rather than tracked
  separately. Validated exactly: the always-speaking genome shows
  exactly 12 (4 reps x 3 steps x 1 channel) in *every one* of the 9
  columns, summing to the already-validated 108.
- **Spatial snapshots** (`GridWorld.snapshot()`, `--snapshot-every` in
  the CLI, written to `{out-dir}/snapshots/day{N}.npz`): per-cell score,
  per-cell per-stimulus speech, and per-cell genetic **purity** (the
  largest group of a cell's 8 individuals sharing bit-identical genomes,
  divided by 8) -- the data behind the paper's "Plates" showing which
  clone occupies which cell over time.
- **Sample-scatter series**: `GridWorld.sample_cells`, 8 subpopulations
  chosen once at random (part of the reproducible seed sequence,
  preserved automatically across checkpoint/resume) and logged as 8
  extra CSV columns every day -- the paper's own third curve, showing
  spread across the array.
- **Mixing-zone/border analysis**: a new standalone script,
  `analyze_snapshot.py`, loads one snapshot and reports mean score/purity
  for border cells (any Moore neighbor's score differs by more than a
  threshold) vs. interior cells -- directly testable against the paper's
  claim that borders score lower from crosses between incompatible
  types. Pure post-processing over the snapshot data; no new runtime
  instrumentation needed.
- **Clone/lineage tracking -- a deliberate non-addition.** An exact
  genome-bit-hash registry was considered and rejected: the paper names
  a "clone" by the *score* it produces ("we refer to an organism type by
  the score it obtains... since that's deterministic"), not by exact
  genetic identity, and with zero mutation but crossover firing on
  ~94% of reproductions (`1 - 0.95^56`), almost every offspring is a
  genetically distinct byte-string from either parent even when its
  behavior is unchanged. A genome-hash registry would show constant
  turnover even during an era the paper's own standard would call one
  persisting clone (e.g. Case 1's 4,200-day "58-clone" era). The
  per-cell score map + purity above is the paper-consistent version of
  this same idea, without separate lineage bookkeeping.

All verified with a small-grid CLI smoke run (`--snapshot-every 10`):
snapshot `.npz` files load back with correct shapes and sane value
ranges (score, `speech_by_stimulus` (8,8,9), purity in [0,1]), the 8
sample + 9 speech-stimulus CSV columns are present, and
`analyze_snapshot.py` produces a border-vs-interior comparison without
error. 42/42 tests passing.

## Codex review (2026-08-30) -- 4 real bugs found and fixed

A second opinion on the checkpoint/metrics work above, requested before
launching Case 2, caught four genuine issues (plus two lower-priority
observations) that hadn't surfaced in testing:

- **`sample_cells` was drawing from the same rng as wind/festival**
  (`self.rng`). A logging-only feature was silently perturbing the
  actual simulated trajectory -- any run using wind or festival would
  diverge from what it would have produced without the sample-cell
  instrumentation, since every wind-direction/emigrant and
  festival-parent/victim draw downstream would be shifted by one extra
  rng consumption. Fixed: a dedicated `instrumentation_rng`, spawned
  separately, used only for sample-cell selection -- adding or changing
  logging from here on can never again perturb the mechanism.
- **`snapshot()`'s purity was computed from the *current* grid state
  when called, after that day's `local_reproduce` (and, on windy/
  festival days, migration/reproduction too) had already mutated every
  cell -- while `score` describes the population *before* those
  mutations.** A snapshot's own score and purity arrays described two
  different populations, on every single day (not just windy/festival
  ones, since local_reproduce runs on every non-festival day). Fixed:
  purity is now computed *during* the same per-cell scoring pass as
  score/speech, immediately after `run_day()` and before
  `local_reproduce` touches that cell -- `run_day(want_snapshot=True)`
  populates `last_day_purity`, which `snapshot()` now just reads (and
  raises clearly if called without it). Verified with a white-box
  regression test that spies on `local_reproduce` (a plain method,
  unlike the numpy Generator calls elsewhere that can't be
  monkeypatched) to confirm purity matches the population as it stood
  at that exact moment, not whatever reproduction left behind.
- **`--resume --workers 1` silently ignored the request to force
  serial.** The CLI collapsed any `--workers <= 1` to `None` before
  calling `load_checkpoint`, but `load_checkpoint`'s `n_workers=None`
  specifically means "keep whatever the checkpoint was saved with" --
  so resuming a checkpoint saved with 30 workers, asking for `--workers
  1`, would silently keep using 30. Fixed: the CLI no longer collapses
  to `None` at all; `GridWorld`/`load_checkpoint` already treat any
  value `<= 1` as serial, so passing the literal value through removes
  the ambiguity entirely.
- **Checkpoint/CSV filenames only encoded seed and grid size, not
  wind/festival period**, and `--resume` never validated the loaded
  checkpoint's config against the CLI's current arguments -- two
  different cases (e.g. Case 1 and Case 2) run into the same `--out-dir`
  with the same seed would silently collide or cross-resume. Fixed:
  filenames now also encode `w{wind_period}_f{festival_period}`, and
  `--resume` errors clearly if the loaded grid_size/wind_period/
  festival_period don't match what the CLI was just given.

Two more points raised, deliberately not changed: a signal-check
placement question (the actual race window -- between the existing
end-of-loop check and the next day starting -- can't be meaningfully
narrowed by relocating a flag check, so there was nothing to fix), and
a tie-breaking observation in `select_parents_and_victim`'s
`np.argsort` (ties bias toward low-index individuals; real, but
pre-existing, already-validated core mechanics from before this
session's changes, and low-impact since tied scores in practice mean
identical genomes anyway -- not touched without separate sign-off).

44/44 tests passing (2 new: the purity-timing regression test above,
and a check that `snapshot()` raises without `want_snapshot=True`);
re-verified end-to-end with a fresh CLI smoke run covering the new
filename tags, the resume-config mismatch error, and forcing serial
(`--workers 1`) across a resume.

## Closing 5 more gaps -- Codex, given the actual paper text, said metrics were still partial (2026-08-30)

Asked directly whether the metric set above supports a *full* comparison
against the paper's own Section 3 analysis, Codex was given the paper's
own verbatim excerpts (not my paraphrase) and independently judged: no,
still partial. Five concrete gaps, all closed:

- **Persistent clone/lineage identity.** Score alone can't distinguish
  "the same clone held this territory" from "an unrelated clone
  independently reached the same score." Not an exact-genome-hash
  registry (already rejected -- crossover on ~94% of reproductions,
  `1 - 0.95^56`, makes exact genetic identity the wrong signal for a
  persisting *strategy*). Instead: `Individual.lineage_id`, a
  single-parent-inheritance ancestry label (like mitochondrial descent)
  independent of the genome's actual bytes -- every founder gets a
  globally unique id (`itertools.count()` in `GridWorld.__init__`);
  every offspring (`local_reproduce`, `_run_festival`) inherits
  parent_a's unchanged. `dominant_lineage_id` / `lineage_purity` in the
  snapshot are the same Counter-based computation as genome purity,
  just grouped by ancestry instead.
- **The "cautious communicator" claim** (26's mostly play the safe -12
  strategy, but rely on signaling specifically in Left Pred/Right Pred
  trials) needs a policy check, not just speech volume. New
  `LocalWorld.last_day_hearing_response`, shape `(8, 9, 4)`: per
  individual per stimulus pair, counts of (heard-something & moved,
  heard-something & stayed, heard-nothing & moved, heard-nothing &
  stayed) across that pair's 4 daily reps -- "heard something" is a
  per-trial fact (hearing is subpopulation-wide in this model), "moved"
  is a movement *response* (attempted at least once during the trial),
  tallied across all 9 stimulus pairs so the paper's specific prediction
  (concentrated at index 4, not spread evenly) is actually checkable
  rather than assumed.
- **Signal honesty** (deception) needed no new instrumentation --
  derivable from data already logged: `analyze_snapshot.py`'s new
  `signal_informativeness()` compares speech during Pred-present
  stimulus pairs against speech during no-stimulus-at-all pairs.
- **Mixing-zone genetics**: the snapshot's new `dominant_genome`
  (each cell's modal genome, free from the same Counter computation
  purity already does) lets `analyze_snapshot.py`'s new
  `border_genetic_distance()` measure genome Hamming distance and
  lineage mismatch across a border, not just a score gap.
- **Exact paper Plate days**: `--snapshot-days "2000,3800,4150,..."` in
  the CLI, combined with the existing `--snapshot-every`.

## Codex review round 2 -- 3 more bugs found and fixed, plus test-coverage gaps closed (2026-08-30)

A second Codex pass on the work above (same reasoning as before: get an
independent check before calling substantial new logic done) found:

- **Stale snapshot stats could pair with the wrong day's scores.**
  `snapshot()` only checked "were purity/lineage/genome stats ever
  computed," not "were they computed *for this exact day*." A
  `run_day(want_snapshot=True)` day followed by a
  `run_day(want_snapshot=False)` day would let `snapshot()` silently
  return the new day's `score` next to the *previous* day's now-stale
  purity/lineage/genome data. Fixed: `GridWorld` now tracks
  `_snapshot_day`, set only when stats are actually (re)computed;
  `snapshot()` raises unless it matches the current day exactly.
- **`hearing_response_summary()` invented probabilities from an epsilon
  hack.** A stimulus pair with zero "heard" (or zero "unheard")
  observations was reported as `P(moved | ...) = 0`, when that
  conditional probability is genuinely undefined, not zero. Fixed with
  `np.divide(..., where=denom > 0)`, returning `nan` and the raw
  heard/unheard counts instead.
- **"Moved" was defined as net displacement, which can miss a real
  movement response.** An individual that fled and came back, or hit
  the boundary clamp and landed back at its start, counted as "stayed."
  Fixed: `moved` now means "attempted at least one move during the
  trial" (`ind.network.move` fired), which is what the hearing-response
  metric is actually meant to capture -- verified this doesn't change
  either existing exact test (`_never_moving_individual` and
  `_always_speaking_individual` both never trigger `move` at all, so
  both definitions agree for them).
- **Old-checkpoint compatibility** (a checkpoint saved before
  `lineage_id` existed would break on load) was flagged but
  deliberately not addressed with migration code: no such checkpoint
  currently exists (Case 1 completed entirely before checkpointing was
  even added), so this would be complexity for a currently-hypothetical
  case, not a present data-loss risk. Documented here instead.

Test gaps Codex found were real: the lineage-inheritance tests only
checked "came from the top-scoring pool," which would also pass if
parent_b's lineage were inherited instead of parent_a's. Fixed by
monkeypatching `select_parents_and_victim` (a plain function, unlike the
numpy Generator calls elsewhere in this codebase, which can't be
monkeypatched) to pin exact parent_a/parent_b/victim indices -- for both
`local_reproduce` (patched on `altruism.world`) and `_run_festival`
(patched on `altruism.grid`, since that's where its own imported
reference lives, not `altruism.world`'s). Also added: snapshot *value*
tests for `dominant_lineage_id`/`lineage_purity` (not just shapes,
using a lineage grouping deliberately different from the genome
grouping, to prove it's tracking something independent); extended the
pre-reproduction timing test to lineage stats too; and a new
`tests/test_analyze_snapshot.py` with exact fixtures for
`border_genetic_distance` (a 6-row two-block grid where the block split
is known exactly), `signal_informativeness`, and
`hearing_response_summary`'s nan-handling (reproducing Codex's own
demonstration case directly).

53/53 tests passing. Re-verified end-to-end with a fresh CLI smoke run.

## Case 2 (wind+festival), full paper scale -- completed (2026-08-30)

**What this case is actually testing.** Case 1 showed communication can
start but not durably survive under wind-only migration, because wind
scrambles who-you-hear vs. who-you're-related-to with no group-cohesion
mechanism at all. Festival migration is the paper's fix: 32-individual
quad tournaments that both migrate *and* reproduce together, so a
successful lineage's neighbors are more likely to actually be its kin --
communication range and breeding range pulled back into alignment. Case
2 asks whether that's enough for a communicating clone to hold territory
for much longer than Case 1's ~4,200-day run, and whether the paper's
own reported result (population mean exceeding the never-moving
baseline of -12 within a few thousand days) reproduces at this scale.

Run: paper's own configuration end to end -- 128x128 grid (131,072
individuals), `wind_period=10`, `festival_period=2`, 14,580 days,
complete metric set (speech, lineage, hearing-response, snapshots)
enabled from day 1. Completed in full, 14,580/14,580 days, with 58
snapshots (`--snapshot-every 250`).

**Trajectory (200-day windows of max/mean cell-average score):**

| Phase | Days | Shape |
|---|---|---|
| 1. Rapid climb | 0 -- 600 | mean rises from -585.8 to -119.1 as the initial random population is culled |
| 2. "42.0 floor" era | 600 -- 13,000+ | max cell score locks to exactly 42.0 and holds there for ~12,400 days straight; mean climbs slowly throughout, crossing the paper's -12 (never-moving) baseline around day 8,000-8,200 |
| 3. Transition | ~13,000 -- 13,295 | max drops 42.0 -> 26.0 over roughly 25-100 days, sharp by comparison to Case 1's transition |
| 4. "26.0 floor" era | 13,295 -- 14,580 (end) | max cell score locks to exactly 26.0 through the end of the run; mean stabilizes around -10 to -11 |

Compared to the paper's own reported Case 2 result (population mean
briefly exceeding -12 around day 3,000-5,000): this run's mean crosses
-12 later (~day 8,000) but, unlike the paper's "brief" exceedance, never
drops back below it again through the end of the run -- a sustained
rather than transient improvement, at this scale and seed.

**Headline finding: the "42.0 floor" was not one clone the whole
time.** A naive read of the trajectory table above -- "max score pinned
at exactly 42.0 for 12,400 days" -- suggests one dominant clone locked
in and stayed there. The lineage-tracking metric (closed as gap #1 of
the 5-gap review, specifically *because* score alone can't tell "same
clone" from "different clone, same score") says otherwise: across all
58 snapshots, the dominant `lineage_id` occupying the 42.0-scoring
territory changes repeatedly -- 81249, 81144, 82092, 51628, 51950,
56233, and (after the transition) a further mix of 51950 / 97048 /
112008 / 28265 / 79972. Multiple unrelated lineages independently
converged on the same 42.0 strategy and traded dominance of the same
territory, invisible in score data alone. The champion's *territory
size* was also shrinking well before its *score* changed: from roughly
260 cells at day 4,750 down to single digits by day 13,000 -- a
population in slow decline for thousands of days before the visible
42.0 -> 26.0 transition, another signal pure max/mean tracking misses
entirely.

**Hearing-response ("cautious communicator") check: a null result.**
The paper's specific claim is that a successful signaling clone plays
the safe -12 (never-move) strategy in general, but *specifically*
relies on hearing the alarm and moving in Left-Pred/Right-Pred trials
(stimulus pair index 4) -- i.e. heard-vs-moved correlation should be
concentrated at index 4 and near-absent elsewhere. Checked across 4
representative snapshots (days 12,000 / 13,000 / 13,500 / 14,500) using
`hearing_response_summary()`'s `P(moved | heard) - P(moved | unheard)`
diff at each of the 9 stimulus pairs: the effect is essentially zero
(~0.000) at *every* stimulus pair, including index 4. This run's
dominant lineages are not showing the paper's specific
hearing-triggers-movement signature at all, despite reaching high
scores and despite the strong signal-honesty result below. Reported
here without softening -- it's a real, somewhat unexpected result that
score/purity/honesty data alone would not have surfaced, and it's the
main reason this hearing-response metric was worth adding in the first
place.

**Why: traced directly to the champion genomes' wiring, not an
instrumentation bug.** Before accepting a clean null like this, checked
whether it was a counting artifact. It isn't -- `P(moved | heard)` and
`P(moved | unheard)` land on identical values to 3-4 decimal places at
every stimulus pair across every snapshot checked, which is what a real
structural absence of effect looks like, not sampling noise. Confirmed
directly by reconstructing each era's dominant genome from its
snapshot's `dominant_genome` bits, developing its network
(`altruism.network.Network`), and checking whether any HEAR unit
(7-12) has a path -- direct or indirect, any number of hops/steps -- to
the MOVE unit (19) via the genome's own connection graph:
- The **42.0-floor champion** (days 12,000 and 13,000 -- the genome
  that held the bulk of the run's territory for ~12,400 of its 14,580
  days) has **zero graph path from any hearing unit to MOVE at all**.
  Its movement decision cannot be influenced by hearing under any
  circumstance -- not weak, not overridden, structurally absent. This
  alone accounts for most of the null result, since this genome
  dominated the grid for most of the run's length.
- The **26.0-era champion** (days 13,500 and 14,500, post-transition)
  *does* have a wired path (hearing channel 'd' -> MOVE, weight -3) --
  so this genome is structurally capable of the paper's mechanism.
  Directly toggling hearing fully on/off, holding predator state and
  location fixed, across every combination: the MOVE decision never
  once flipped. The other inputs into MOVE (weights up to +-11) dominate
  the threshold regardless of what that one hearing channel says, so
  the wiring exists but is functionally vestigial.

Conclusion: this is a genuine finding about what evolved, not a
measurement problem. Whatever pushed these clones' scores above the
never-move baseline, it wasn't "hear the alarm, then move" -- most
likely their own direct sensing of the predator/location inputs, which
feed MOVE independently of anyone's speech.

**Signal honesty: strongly selective, but the raw ratio is an
artifact.** `signal_informativeness()` (Pred-present speech / no-Pred
speech, per cell) comes back in the hundreds of millions at all four
checked snapshots. That number is real but misleading on its face: it's
large because the no-Pred-speech denominator is close to zero, not
because Pred-present speech is enormous -- i.e. these cells speak
almost *only* when a predator stimulus is actually present, which is
exactly the honest-signaling behavior the paper describes, but the
astronomical ratio itself is a near-zero-denominator artifact and
should be read as "extremely selective," not literally "N times more."

**Mixing-zone genetics: borders modestly but consistently show
incompatible-neighbor signal.** `border_genetic_distance()` across the
same 4 snapshots: border cells (score-gap Moore-neighbor boundaries)
show mean Hamming distance ~0.14 vs. interior ~0.10-0.11, and lineage
mismatch rate ~0.49-0.50 vs. interior ~0.43-0.45 -- borders are
genetically less related and more often cross-lineage than interior
cells, and (at day 12,000) score at the border averages -8.09 vs. -1.71
interior. Modest in magnitude, but consistent in direction across every
snapshot checked -- direct genetic evidence for the paper's "mixing
zone" claim, not just a score-gap proxy for it.

**Case 1 vs. Case 2.** Festival migration did what it was meant to:
the dominant-clone-adjacent "42.0 floor" territory held for ~12,400
days here, roughly 3x longer than Case 1's best single era (~4,200 days
for the "58" clone), and the eventual transition to the next era was
much faster and cleaner (~25-100 days vs. Case 1's ~2,200-day contested
back-and-forth). But "held longer" is not the same as "one victorious
communicating clone" -- the lineage-succession finding above shows
Case 2's floor era was itself a *sequence* of unrelated lineages
cycling through the same score, and the hearing-response null result
means this run doesn't confirm the paper's specific cautious-
communicator mechanism was what got them there. Festival buys
durability; it doesn't, at least in this run, produce the single
long-lived kin-selected communicator the paper's mechanism describes.

## Not yet run

- Case 3 (festival-only, `festival_period=2`, 99,980 days) -- not
  launched yet; a much longer commitment even parallelized.
- A Case 1 redo (or a fresh Case 2 run at a different seed) with the
  full metric set now available from day 1 -- would let the
  hearing-response null result above be checked for seed-dependence,
  rather than concluding from a single run.
  Checkpoint/resume means this no longer requires deciding up front --
  a run can always be extended or re-analyzed later without restarting.
