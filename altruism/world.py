"""Local level (Section 2.1): a subpopulation of 8 individuals, each in
its own track, living through a "day" of 36 trials, then (subject to the
global-level strategy) a local reproduction.
"""

from dataclasses import dataclass, field

import numpy as np

from altruism.genome import Genome, crossover, founder_genome
from altruism.network import Network
from altruism.units import LOCATIONS

N_INDIVIDUALS = 8
N_TRIALS_PER_DAY = 36
N_STEPS_PER_TRIAL = 3
MOVE_COST = 1

STIMULI = (None, "Pred", "Food")

# Score vectors: index by final location L=0,1=1,2=2,R=3 (Figure 1's table).
# Values not spelled out in the paper for a stimulus type default to the
# published ones (Pred, Food); "no stimulus" on a side contributes nothing.
_LEFT_PRED = np.array([-81, -42, -23, 0])
_RIGHT_PRED = np.array([0, -23, -42, -81])
_LEFT_FOOD = np.array([30, 0, 0, 0])
_RIGHT_FOOD = np.array([0, 0, 0, 30])


def _score_vector(left_stim: str | None, right_stim: str | None) -> np.ndarray:
    v = np.zeros(4)
    if left_stim == "Pred":
        v += _LEFT_PRED
    elif left_stim == "Food":
        v += _LEFT_FOOD
    if right_stim == "Pred":
        v += _RIGHT_PRED
    elif right_stim == "Food":
        v += _RIGHT_FOOD
    return v


@dataclass
class Individual:
    genome: Genome
    network: Network = field(default=None)

    def __post_init__(self):
        if self.network is None:
            self.network = Network(self.genome)


def _latin_square_locations(rng: np.random.Generator) -> list[list[int]]:
    """For the 4 repetitions of one stimulus pair: returns loc_by_rep[rep]
    = list of 8 location-indices (one per individual, track order),
    satisfying (a) exactly 2 individuals per location each rep, (b) each
    individual visits all 4 distinct locations across the 4 reps -- via
    two independently-shuffled groups of 4, each cycled through the 4
    locations."""
    order = rng.permutation(N_INDIVIDUALS)
    group_a, group_b = order[:4], order[4:]
    loc_by_rep = [[0] * N_INDIVIDUALS for _ in range(4)]
    for rep in range(4):
        for j, individual in enumerate(group_a):
            loc_by_rep[rep][individual] = (j + rep) % 4
        for j, individual in enumerate(group_b):
            loc_by_rep[rep][individual] = (j + rep) % 4
    return loc_by_rep


def select_parents_and_victim(
    scores: np.ndarray, top_k: int, rng: np.random.Generator
) -> tuple[int, int, int]:
    """Shared rank/select/kill mechanic behind both local reproduction
    (Section 2.1, top_k=4 of 8) and festival reproduction (Section 2.3,
    top_k=8 of 32) -- same rule, different pool size. Ranks `scores`
    descending; two parents are chosen uniformly (with replacement) from
    the top `top_k`; one victim is chosen uniformly from the whole pool.
    Returns (parent_a_index, parent_b_index, victim_index), indices into
    `scores`."""
    ranked = np.argsort(-scores)  # best first
    top = ranked[:top_k]
    parent_a, parent_b = rng.choice(top, size=2, replace=True)
    victim = int(rng.integers(0, len(scores)))
    return int(parent_a), int(parent_b), victim


class LocalWorld:
    """One subpopulation of 8 individuals. Owns the day's trial/scoring
    loop and local (non-festival) reproduction. Global-level concerns
    (migration, festivals across a grid of these) are handled one layer up
    -- see grid.py."""

    def __init__(self, rng: np.random.Generator, individuals: list[Individual] | None = None):
        self.rng = rng
        self.individuals = individuals or [Individual(genome=founder_genome(rng)) for _ in range(N_INDIVIDUALS)]
        assert len(self.individuals) == N_INDIVIDUALS
        # Total active speech-channel-bits emitted over the most recent
        # day's 36 trials, per individual (max possible: 6 channels x 3
        # steps x 36 trials = 648) -- a cheap communication-activity
        # proxy, set fresh by every run_day() call.
        self.last_day_speech_activity = np.zeros(N_INDIVIDUALS)

    def run_day(self) -> np.ndarray:
        """Runs the day's 36 trials (all 9 stimulus combinations x 4 reps
        each) and returns each individual's total behavioral score for
        the day, in track order. Also refreshes
        `last_day_speech_activity` as a side effect."""
        scores = np.zeros(N_INDIVIDUALS)
        speech_activity = np.zeros(N_INDIVIDUALS)
        stim_pairs = [(left, right) for left in STIMULI for right in STIMULI]
        for left_stim, right_stim in stim_pairs:
            loc_by_rep = _latin_square_locations(self.rng)
            for rep in range(4):
                scores += self._run_trial(left_stim, right_stim, loc_by_rep[rep], speech_activity)
        self.last_day_speech_activity = speech_activity
        return scores

    def _run_trial(
        self, left_stim, right_stim, start_locations: list[int], speech_activity: np.ndarray
    ) -> np.ndarray:
        for ind in self.individuals:
            ind.network.reset_trial(ind.genome)
        positions = list(start_locations)
        trial_scores = np.zeros(N_INDIVIDUALS)
        prev_speech = np.zeros((N_INDIVIDUALS, 6), dtype=np.int16)  # all zero on first step

        for _step in range(N_STEPS_PER_TRIAL):
            hearing = prev_speech.sum(axis=0)  # subpopulation-wide, non-localized
            this_step_speech = np.zeros((N_INDIVIDUALS, 6), dtype=np.int16)
            for i, ind in enumerate(self.individuals):
                loc_name = LOCATIONS[positions[i]]
                pred = 1 if loc_name in ("L", "R") and (
                    (loc_name == "L" and left_stim == "Pred") or (loc_name == "R" and right_stim == "Pred")
                ) else 0
                food = 1 if loc_name in ("L", "R") and (
                    (loc_name == "L" and left_stim == "Food") or (loc_name == "R" and right_stim == "Food")
                ) else 0
                ind.network.react(pred=pred, food=food, location=loc_name, hearing=hearing)
                this_step_speech[i] = ind.network.speech
                speech_activity[i] += int(this_step_speech[i].sum())
                if ind.network.move:
                    trial_scores[i] -= MOVE_COST
                    positions[i] += 1 if ind.network.to_r else -1
                    positions[i] = max(0, min(3, positions[i]))
            prev_speech = this_step_speech

        vector = _score_vector(left_stim, right_stim)
        for i in range(N_INDIVIDUALS):
            trial_scores[i] += vector[positions[i]]
        return trial_scores

    def local_reproduce(self, scores: np.ndarray):
        """Rank by score; two parents chosen uniformly from the top half;
        one offspring via crossover; one of the 8 killed uniformly at
        random and replaced (Section 2.1)."""
        parent_a, parent_b, dead = select_parents_and_victim(scores, top_k=N_INDIVIDUALS // 2, rng=self.rng)
        child_genome = crossover(self.individuals[parent_a].genome, self.individuals[parent_b].genome, self.rng)
        self.individuals[dead] = Individual(genome=child_genome)
