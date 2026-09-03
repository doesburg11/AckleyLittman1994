# Altruism in the Evolution of Communication — Ackley & Littman (1994)

A from-scratch replication of:

> Ackley, D. H., & Littman, M. L. (1994). *Altruism in the Evolution of
> Communication.* In R. A. Brooks & P. Maes (Eds.), *Artificial Life IV*
> (pp. 40–48). MIT Press.

The paper's question: can costly, initially-arbitrary signaling evolve and
stabilize even when it provides no direct benefit to the signaler? Their
answer is yes, provided communication range and breeding range are mostly
(but not always) aligned — "you'll be talking to family" most of the time,
which lets kin selection favor the emergence of altruistic signaling,
while occasional misalignment keeps subpopulations from converging and the
search for improved forms from stalling.

This is a sibling project to
[AckleyLittman1991](https://github.com/doesburg11/AckleyLittman1991) (a
replication of the same authors' earlier ERL paper) and to
[HintonNowlan1987](https://github.com/doesburg11/HintonNowlan1987), but a
genuinely different model — no predator-prey-grass ecology, no individual
learning at all (genes fully and permanently determine behavior; there is
no plasticity in this paper). It's a separate, self-contained architecture:
8-individual subpopulations distributed across a grid, each individual a
32-unit linear-threshold network grown from a 448-bit genome via an
indirect, graph-traversal-style developmental process.

## Status: work in progress

**Done and unit-tested** (the paper's Section 2.1–2.2, "local level" — one
subpopulation of 8 individuals, no grid yet):
- The 448-bit genome and its developmental process (source/destination
  mode, mode-shift codes, the fixed six-value weight table).
- Two-pass synchronous network execution.
- The trial structure (36 trials/day: 9 stimulus combinations × 4
  repetitions, Latin-square-assigned starting locations), the exact
  scoring table from Figure 1, and local (rank-based, top-half-parent)
  reproduction.

A precise, independent validation: a never-moving individual scores
**exactly -696** in a full day of 36 trials — matching the paper's own
stated number (p. 43–44) exactly, computed from the score-vector and
trial-structure implementation with no fitting involved.

**Done and unit-tested** (the paper's Section 2.3, "global level" —
`altruism/grid.py`):
- The population grid: a torus of `LocalWorld` subpopulations
  (`GridWorld`, parameterizable size — the paper's own scale is
  128×128/131,072 individuals total).
- Wind migration: one global compass direction per windy day, one random
  emigrant per subpopulation, each immigrant landing in the exact slot
  its destination's own emigrant vacated (a simultaneous,
  population-preserving swap).
- Festival migration/reproduction: 32-individual quad tournaments (top
  quarter parents, one random victim replaced), with the quad grouping
  phase-shifted (Margolus-neighborhood style) across successive festivals
  so every cell interacts with all eight of its neighbors over 4
  festivals.
- A CLI runner (`run_grid_simulation.py`) with an optional
  `--workers N` multiprocess pool for the per-cell scoring step (the
  dominant cost by far) — measured at a 7.7× wall-clock speedup on 32
  cores, with byte-identical output to a single-threaded run.
- Checkpoint/resume: `GridWorld.save_checkpoint()`/`load_checkpoint()`
  (atomic pickle) plus `--checkpoint-every`/`--resume` in the CLI —
  `kill <pid>` finishes the in-progress day, checkpoints, and exits
  cleanly, so a long run can be stopped and resumed (or extended, or
  re-analyzed with a metric added afterward) without restarting from
  day 1. Verified against an uninterrupted reference run: byte-identical.
- Full metric parity with the paper's own analysis, closed in two
  passes (the second after Codex, given the actual paper text, judged
  the first pass still partial): per-stimulus speech breakdown and a
  hearing-response tally (heard-something × moved, per stimulus pair —
  tests the paper's specific "cautious communicator" claim directly,
  not just how much a cell talks); periodic spatial snapshots (per-cell
  score, per-stimulus speech, genetic purity, ancestry-based
  `lineage_id` tracking — distinguishes "the same clone held this
  territory" from "an unrelated clone reached the same score," which
  exact-genome hashing can't, since crossover makes exact genetic
  identity the wrong signal — and each cell's modal genome); the
  paper's own 8-subpopulation sample-scatter series; `--snapshot-days`
  to hit the paper's exact reported Plate days; and `analyze_snapshot.py`
  for border/mixing-zone genetics and signal-honesty analysis. See
  `RESULTS.md` for the full gap list and why ancestry (not exact genome
  bytes) is the right way to track a persisting clone.

**What Case 1 is actually testing.** It's the harshest, simplest version
of the paper's central question -- kin selection can only favor costly
communication if communication range and breeding range are correlated,
and wind-only migration ("anything goes," no group-cohesion mechanism at
all) is the crudest way to inject the imperfection that keeps a
population from freezing. Two questions it's built to answer, and the
reference point Case 2/3 get compared against:
- **Can communication even get started under the harshest disruption
  tested?** Yes -- communicating clones (the paper's "26's") repeatedly
  emerge.
- **Can it survive there?** No, not durably -- wind keeps scrambling
  who-you-hear vs. who-you're-related-to, so non-communicating "cheaters"
  ("-12's") keep re-invading before a communicating clone can grow large.
  The population mean climbs but plateaus well short of full
  communicator dominance.

**Done and run**: Case 1 (wind-only), the paper's own configuration and
full 128×128/131,072-individual scale, completed in ~59 hours wall-clock
— see `RESULTS.md` for the full result (a 4-phase trajectory: rapid
climb, a communicating-ish clone dominating for ~4,200 days, a ~2,200-day
contested turnover, then a lower-scoring but more durable clone
dominating for the remaining ~4,300+ days through the end of the run).
This run predates the full metric set above, so it can't itself confirm
whether the dominant clones were actually communicating.

**Done and run**: Case 2 (wind+festival), the paper's own configuration
(wind every 10th day, festival every 2nd day, 14,580 days), full
128×128/131,072-individual scale, completed 2026-08-30 with the
complete metric set (speech, lineage, hearing-response, snapshots)
enabled from day 1 — see `RESULTS.md` for the full result. Headline:
the dominant "42.0 floor" era held for ~12,400 days (~3× longer than
Case 1's best era), but lineage tracking shows it was a *succession* of
unrelated clones trading the same score, not one clone holding on —
invisible in score data alone. The paper's specific "cautious
communicator" hearing-response signature (heard-alarm → moved,
concentrated at the Left-Pred/Right-Pred stimulus) came back as an
essentially null result across every stimulus pair — traced directly
to the champion genomes' own network wiring (not an instrumentation
bug): the genome that held the bulk of the run's territory has zero
graph path from any hearing unit to its movement unit at all, and the
later champion that does have such wiring never actually uses it (its
movement decision is dominated by other, stronger inputs regardless of
what it hears). Signal honesty and mixing-zone genetics both came back positive
(extremely selective Pred-triggered speech; borders modestly but
consistently more genetically mismatched and lower-scoring than
interiors).

**Not yet done**:
- Case 3 (festival-only) — not launched yet; a much longer commitment
  even parallelized (99,980 days).
- A Case 1 redo, or a Case 2 rerun at a different seed, with the full
  metric set from day 1 — would check whether the hearing-response null
  result above is seed-dependent rather than concluding from one run.

See `RESULTS.md` for validation details and status as the global level is
built out.
