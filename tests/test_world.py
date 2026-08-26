import numpy as np
import pytest

from altruism.genome import Genome
from altruism.world import Individual, LocalWorld, N_INDIVIDUALS, _latin_square_locations


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _never_moving_individual() -> Individual:
    """An all-zero genome: empty initial states (MOVE starts at 0) and
    every connection group is a harmless SOURCE_MODE_SHIFT (weight
    specifier 0), so develop() yields zero connections -- MOVE can never
    turn on, by construction, at any point in any trial."""
    return Individual(genome=Genome(bits=np.zeros(448, dtype=np.uint8)))


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
