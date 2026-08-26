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

**Not yet built** (the paper's Section 2.3, "global level" — the actual
experimental variable in their case studies):
- The population grid (their scale: 128×128 subpopulations, 131,072
  individuals total).
- Wind migration (periodic, random-direction, one individual per
  subpopulation).
- Festival migration/reproduction (periodic 2×2-quad tournaments, with
  the quad phase shifted across four successive festivals so every cell
  eventually interacts with all eight of its neighbors).

See `RESULTS.md` for validation details and status as the global level is
built out.
