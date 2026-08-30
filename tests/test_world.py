import numpy as np
import pytest

from altruism.genome import Genome
from altruism.units import SPEAK, TRUE
from altruism.world import (
    Individual,
    LocalWorld,
    N_INDIVIDUALS,
    N_STEPS_PER_TRIAL,
    N_STIM_PAIRS,
    N_TRIALS_PER_DAY,
    _latin_square_locations,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _never_moving_individual() -> Individual:
    """An all-zero genome: empty initial states (MOVE starts at 0) and
    every connection group is a harmless SOURCE_MODE_SHIFT (weight
    specifier 0), so develop() yields zero connections -- MOVE can never
    turn on, by construction, at any point in any trial."""
    return Individual(genome=Genome(bits=np.zeros(448, dtype=np.uint8)))


def _bits_for(value: int, n: int) -> list[int]:
    return [int(c) for c in format(value, f"0{n}b")]


def _always_speaking_individual() -> Individual:
    """A hand-crafted genome with exactly one connection: TRUE -> SPEAK[0],
    weight +11 (source mode). TRUE is always 1, so this one speech
    channel is unconditionally active every step of every trial,
    regardless of stimuli, location, or hearing -- a fully deterministic
    speech-activity total, the same validation style as the -696 score
    check."""
    bits = np.zeros(448, dtype=np.uint8)
    bits[19:24] = _bits_for(TRUE, 5)      # initial source = TRUE (unit 0)
    bits[24:29] = _bits_for(SPEAK[0], 5)  # first group's connection-unit = SPEAK[0] (13)
    bits[29:32] = _bits_for(6, 3)         # weight_specifier = 6 -> weight +11
    return Individual(genome=Genome(bits=bits))


def test_never_moving_individual_scores_exactly_minus_696(rng):
    """A precise, independent check against the paper's own stated number
    (page 43-44): 'the far the most probable resulting behavior pattern is
    never to move at all. Such individuals receive behavioral scores of
    -696.' A never-moving individual's day-score is fully determined by
    the score vectors and the Latin-square guarantee that every individual
    visits each of the 4 locations exactly once per stimulus pair (36
    trials = 9 stimulus pairs x 4 locations), independent of RNG -- so this
    is an exact, not statistical, check."""
    world = LocalWorld(rng, individuals=[_never_moving_individual() for _ in range(N_INDIVIDUALS)])
    scores = world.run_day()
    assert np.allclose(scores, -696.0)


def test_latin_square_covers_all_locations_and_two_per_location(rng):
    loc_by_rep = _latin_square_locations(rng)
    assert len(loc_by_rep) == 4
    for rep_locations in loc_by_rep:
        counts = np.bincount(rep_locations, minlength=4)
        assert (counts == 2).all(), "exactly two individuals must start at each location"

    per_individual = np.array(loc_by_rep).T  # shape (8 individuals, 4 reps)
    for individual_locations in per_individual:
        assert sorted(individual_locations) == [0, 1, 2, 3], \
            "each individual must visit all four distinct locations across the four repetitions"


def test_run_day_returns_one_score_per_individual(rng):
    world = LocalWorld(rng)
    scores = world.run_day()
    assert scores.shape == (N_INDIVIDUALS,)


def test_local_reproduce_replaces_exactly_one_individual(rng):
    world = LocalWorld(rng)
    before_genomes = [ind.genome.bits.copy() for ind in world.individuals]
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3], dtype=float)
    world.local_reproduce(scores)

    changed = [
        not np.array_equal(before, world.individuals[i].genome.bits)
        for i, before in enumerate(before_genomes)
    ]
    assert sum(changed) == 1, "local reproduction must replace exactly one of the eight individuals"
    assert len(world.individuals) == N_INDIVIDUALS


def test_local_reproduce_parents_come_from_top_half(rng):
    """Indirect check (numpy Generator methods can't be monkeypatched):
    make the top-half individuals all-zero-genome and the bottom-half
    all-one-genome. If parents were ever drawn from the bottom half, the
    offspring would sometimes contain 1-bits; if selection is correctly
    restricted to the top half, the offspring must always be all-zero."""
    individuals = (
        [Individual(genome=Genome(bits=np.zeros(448, dtype=np.uint8))) for _ in range(4)]
        + [Individual(genome=Genome(bits=np.ones(448, dtype=np.uint8))) for _ in range(4)]
    )
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3], dtype=float)  # indices 0..3 rank top half
    # Under correct top-half-only parentage, every individual's genome bit-sum
    # stays either 0 (an untouched or newly-bred top-half genome) or 448 (an
    # untouched bottom-half genome) -- a sum strictly between the two would
    # mean the offspring mixed in bottom-half genetic material.
    for _ in range(100):
        world = LocalWorld(rng, individuals=[Individual(genome=ind.genome.copy()) for ind in individuals])
        world.local_reproduce(scores)
        sums = [int(ind.genome.bits.sum()) for ind in world.individuals]
        assert all(s in (0, 448) for s in sums), f"unexpected genome mix: {sums}"


def test_never_moving_individuals_have_zero_speech_activity(rng):
    world = LocalWorld(rng, individuals=[_never_moving_individual() for _ in range(N_INDIVIDUALS)])
    world.run_day()
    assert np.allclose(world.last_day_speech_activity, 0.0)


def test_always_speaking_individual_has_exact_speech_activity(rng):
    """One deterministically-always-speaking individual among seven
    never-moving (hence never-speaking) ones: the always-speaker's
    day total must be exactly 1 active channel x 3 steps x 36 trials =
    108, and every never-moving individual's total must be exactly 0 --
    an exact, not statistical, check on the speech-activity accounting,
    the same validation style as the -696 score check."""
    individuals = [_always_speaking_individual()] + [_never_moving_individual() for _ in range(N_INDIVIDUALS - 1)]
    world = LocalWorld(rng, individuals=individuals)
    world.run_day()
    assert world.last_day_speech_activity[0] == N_STEPS_PER_TRIAL * N_TRIALS_PER_DAY
    assert np.allclose(world.last_day_speech_activity[1:], 0.0)


def test_always_speaking_individual_has_exact_speech_per_stimulus(rng):
    """Same setup, but checking the per-stimulus breakdown: the
    always-speaker must show exactly 4 reps x 3 steps x 1 channel = 12
    in *every one* of the 9 stimulus-pair columns (regardless of which
    stimuli were present, since this genome's one connection is fed by
    TRUE, never by a sensor), summing to the already-validated 108."""
    individuals = [_always_speaking_individual()] + [_never_moving_individual() for _ in range(N_INDIVIDUALS - 1)]
    world = LocalWorld(rng, individuals=individuals)
    world.run_day()
    assert world.last_day_speech_by_stimulus.shape == (N_INDIVIDUALS, N_STIM_PAIRS)
    expected_per_stimulus = N_STEPS_PER_TRIAL * 4  # 3 steps x 4 reps = 12
    assert np.array_equal(world.last_day_speech_by_stimulus[0], np.full(N_STIM_PAIRS, expected_per_stimulus))
    assert np.allclose(world.last_day_speech_by_stimulus[1:], 0.0)
    assert world.last_day_speech_by_stimulus[0].sum() == world.last_day_speech_activity[0]
