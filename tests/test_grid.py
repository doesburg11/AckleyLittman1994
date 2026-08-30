from collections import Counter

import numpy as np
import pytest

from altruism.genome import Genome
from altruism.grid import COMPASS_OFFSETS, N_SAMPLE_CELLS, PHASE_OFFSETS, N_QUAD_INDIVIDUALS, GridWorld
from altruism.world import Individual, N_INDIVIDUALS, N_STIM_PAIRS


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_grid_initializes_every_cell_with_eight_individuals():
    world = GridWorld(seed=0, grid_size=4)
    assert len(world.cells) == 4
    for row in world.cells:
        assert len(row) == 4
        for cell in row:
            assert len(cell.individuals) == N_INDIVIDUALS


def test_wind_migration_moves_exact_slot_to_destination_vacated_slot():
    """Forces direction index 0 (COMPASS_OFFSETS[0] == (-1, 0), i.e. every
    cell's slot-0 individual moves one row 'north') and forces every cell's
    emigrant to be its own slot 0. The individual that left cell (r, c)
    must land in slot 0 of cell (r-1, c) -- the exact slot vacated there --
    and every other slot must be untouched."""
    world = GridWorld(seed=0, grid_size=4)
    n = world.grid_size
    assert COMPASS_OFFSETS[0] == (-1, 0)

    before = [[list(world.cells[r][c].individuals) for c in range(n)] for r in range(n)]

    class _FixedWindRNG:
        def integers(self, low, high=None, size=None):
            if size is None:
                return 0  # direction index -> COMPASS_OFFSETS[0]
            return np.zeros(size, dtype=int)  # every cell's emigrant is slot 0

    world.rng = _FixedWindRNG()
    world._run_wind()

    for r in range(n):
        for c in range(n):
            dest_r = (r - 1) % n
            assert world.cells[dest_r][c].individuals[0] is before[r][c][0]
            for slot in range(1, N_INDIVIDUALS):
                assert world.cells[r][c].individuals[slot] is before[r][c][slot]


def test_wind_migration_preserves_population_as_a_permutation(rng):
    world = GridWorld(seed=int(rng.integers(0, 1_000_000)), grid_size=6)
    before_ids = {id(ind) for row in world.cells for cell in row for ind in cell.individuals}
    world._run_wind()
    after_ids = {id(ind) for row in world.cells for cell in row for ind in cell.individuals}
    assert before_ids == after_ids
    for row in world.cells:
        for cell in row:
            assert len(cell.individuals) == N_INDIVIDUALS


def test_festival_quad_phase_cycle_covers_all_eight_neighbors():
    world = GridWorld(seed=2, grid_size=8)
    n = world.grid_size
    r, c = 3, 5
    expected_neighbors = {((r + dr) % n, (c + dc) % n) for dr, dc in COMPASS_OFFSETS}

    covered = set()
    for phase_offset in PHASE_OFFSETS:
        row_offset, col_offset = phase_offset
        g_row = ((r - row_offset) % n) // 2
        g_col = ((c - col_offset) % n) // 2
        quad = world._quad_cells(g_row, g_col, phase_offset)
        covered.update(cell for cell in quad if cell != (r, c))

    assert covered == expected_neighbors


def test_festival_reproduce_parents_come_from_top_quarter_of_quad(rng):
    """Same technique as test_world.py's
    test_local_reproduce_parents_come_from_top_half, generalized to the
    32-individual quad pool: the first cell (by quad_cells order) gets
    all-zero genomes and is given the highest scores (top quarter of 32),
    the other three cells get all-one genomes and lower scores. If
    festival reproduction ever drew a parent from outside the top
    quarter, the offspring's genome could contain 1-bits; under correct
    top-quarter-only parentage the offspring's bit-sum must always be
    exactly 0."""
    grid_size = 2
    for _ in range(50):
        world = GridWorld(seed=int(rng.integers(0, 1_000_000)), grid_size=grid_size)
        quad_cells = world._quad_cells(0, 0, PHASE_OFFSETS[0])
        assert len(quad_cells) == 4

        for i, (r, c) in enumerate(quad_cells):
            bit_value = 0 if i == 0 else 1
            world.cells[r][c].individuals = [
                Individual(genome=Genome(bits=np.full(448, bit_value, dtype=np.uint8)))
                for _ in range(N_INDIVIDUALS)
            ]

        scores = np.zeros((grid_size, grid_size, N_INDIVIDUALS))
        flat = 0
        for (r, c) in quad_cells:
            for slot in range(N_INDIVIDUALS):
                scores[r, c, slot] = N_QUAD_INDIVIDUALS - 1 - flat  # strictly descending by quad order
                flat += 1

        before = {
            (r, c, slot): id(world.cells[r][c].individuals[slot])
            for (r, c) in quad_cells
            for slot in range(N_INDIVIDUALS)
        }

        world._run_festival(scores)

        # Identity, not genome content, marks the replaced slot: a victim
        # drawn from the top-quarter (all-zero) group gets a new offspring
        # Individual whose genome is *also* all-zero, so content alone
        # can't distinguish "replaced" from "untouched" in that case.
        replaced = [
            (r, c, slot)
            for (r, c) in quad_cells
            for slot in range(N_INDIVIDUALS)
            if id(world.cells[r][c].individuals[slot]) != before[(r, c, slot)]
        ]
        assert len(replaced) == 1, f"exactly one individual across the quad must be replaced, got {len(replaced)}"
        r, c, slot = replaced[0]
        offspring_bits = world.cells[r][c].individuals[slot].genome.bits
        assert offspring_bits.sum() == 0, "offspring must be bred from top-quarter (all-zero) parents only"


def test_parallel_scoring_matches_serial_given_same_seed():
    """The per-cell scoring/local-reproduction step is safe to parallelize
    because each cell's outcome depends only on that cell's own rng
    stream, never on which worker ran it or what order cells were
    processed in. A multi-worker run must therefore end up bit-for-bit
    identical to a single-threaded run given the same seed."""

    def run(n_workers):
        with GridWorld(seed=3, grid_size=4, wind_period=3, festival_period=2, n_workers=n_workers) as world:
            for _ in range(20):
                world.run_day()
            return [
                world.cells[r][c].individuals[slot].genome.bits.copy()
                for r in range(world.grid_size)
                for c in range(world.grid_size)
                for slot in range(N_INDIVIDUALS)
            ]

    serial_result = run(n_workers=None)
    parallel_result = run(n_workers=2)

    for serial_bits, parallel_bits in zip(serial_result, parallel_result):
        assert np.array_equal(serial_bits, parallel_bits)


def test_grid_world_context_manager_shuts_down_executor():
    with GridWorld(seed=0, grid_size=4, n_workers=2) as world:
        world.run_day()
        assert world._executor is not None
    assert world._executor is None


def test_last_day_speech_has_correct_shape_and_matches_serial_vs_parallel():
    """`last_day_speech` and `last_day_speech_by_stimulus` (a
    communication-activity proxy, collected alongside scores) must have
    the expected shapes, and -- since they're collected through the same
    per-cell-independent-rng path -- be exactly reproducible between a
    serial and a parallel run, same as the genome bits already are."""

    def run(n_workers):
        with GridWorld(seed=5, grid_size=4, wind_period=3, festival_period=2, n_workers=n_workers) as world:
            for _ in range(10):
                world.run_day()
            assert world.last_day_speech.shape == (4, 4, N_INDIVIDUALS)
            assert world.last_day_speech_by_stimulus.shape == (4, 4, N_INDIVIDUALS, N_STIM_PAIRS)
            return world.last_day_speech.copy(), world.last_day_speech_by_stimulus.copy()

    serial_speech, serial_speech_by_stim = run(n_workers=None)
    parallel_speech, parallel_speech_by_stim = run(n_workers=2)
    assert np.array_equal(serial_speech, parallel_speech)
    assert np.array_equal(serial_speech_by_stim, parallel_speech_by_stim)


def test_end_to_end_small_grid_run_is_reproducible_and_conserves_population():
    def run(seed):
        world = GridWorld(seed=seed, grid_size=4, wind_period=3, festival_period=2)
        for _ in range(40):
            world.run_day()
        return world

    world_a = run(seed=7)
    world_b = run(seed=7)

    for r in range(4):
        for c in range(4):
            assert len(world_a.cells[r][c].individuals) == N_INDIVIDUALS
            for slot in range(N_INDIVIDUALS):
                bits_a = world_a.cells[r][c].individuals[slot].genome.bits
                bits_b = world_b.cells[r][c].individuals[slot].genome.bits
                assert np.array_equal(bits_a, bits_b)


def _grid_fingerprint(world):
    """Genome bits and speech activity for every individual on the grid
    -- enough to detect any divergence at all between two runs."""
    return [
        (
            world.cells[r][c].individuals[slot].genome.bits.copy(),
            world.last_day_speech[r, c, slot],
        )
        for r in range(world.grid_size)
        for c in range(world.grid_size)
        for slot in range(N_INDIVIDUALS)
    ]


def test_checkpoint_round_trip_matches_uninterrupted_run(tmp_path):
    """Saving mid-run and resuming from that checkpoint must be
    indistinguishable, day-for-day, from never having stopped: the same
    seed run straight through N+M days must land on bit-identical
    genomes and speech-activity values as N days followed by a
    save/load and M more days."""
    seed, grid_size, n, m = 11, 4, 12, 10

    uninterrupted = GridWorld(seed=seed, grid_size=grid_size, wind_period=3, festival_period=2)
    for _ in range(n + m):
        uninterrupted.run_day()

    world_a = GridWorld(seed=seed, grid_size=grid_size, wind_period=3, festival_period=2)
    for _ in range(n):
        world_a.run_day()
    checkpoint_path = str(tmp_path / "checkpoint.pkl")
    world_a.save_checkpoint(checkpoint_path)

    resumed = GridWorld.load_checkpoint(checkpoint_path)
    assert resumed.day == n
    for _ in range(m):
        resumed.run_day()

    expected = _grid_fingerprint(uninterrupted)
    actual = _grid_fingerprint(resumed)
    for (exp_bits, exp_speech), (act_bits, act_speech) in zip(expected, actual):
        assert np.array_equal(exp_bits, act_bits)
        assert exp_speech == act_speech


def test_checkpoint_resume_can_change_worker_count(tmp_path):
    """Saved serial, resumed parallel (or vice versa) must still match
    the same-seed serial-only equivalent -- the worker count used at any
    point never affects the outcome."""
    seed, grid_size, n, m = 13, 4, 8, 8

    reference = GridWorld(seed=seed, grid_size=grid_size, wind_period=3, festival_period=2)
    for _ in range(n + m):
        reference.run_day()

    with GridWorld(seed=seed, grid_size=grid_size, wind_period=3, festival_period=2, n_workers=None) as world:
        for _ in range(n):
            world.run_day()
        checkpoint_path = str(tmp_path / "checkpoint.pkl")
        world.save_checkpoint(checkpoint_path)

    with GridWorld.load_checkpoint(checkpoint_path, n_workers=2) as resumed:
        for _ in range(m):
            resumed.run_day()
        expected = _grid_fingerprint(reference)
        actual = _grid_fingerprint(resumed)
        for (exp_bits, exp_speech), (act_bits, act_speech) in zip(expected, actual):
            assert np.array_equal(exp_bits, act_bits)
            assert exp_speech == act_speech


def test_save_checkpoint_does_not_disturb_a_live_executor(tmp_path):
    """Taking a checkpoint mid-run must not touch the live object's own
    executor -- __getstate__ excludes it only from the pickled copy."""
    with GridWorld(seed=0, grid_size=4, n_workers=2) as world:
        world.run_day()
        world.save_checkpoint(str(tmp_path / "checkpoint.pkl"))
        assert world._executor is not None
        world.run_day()  # must still work after saving a checkpoint


def test_snapshot_shapes_and_purity():
    """`snapshot()` must return the documented shapes, and purity must
    be exactly computable: 1.0 for a cell whose 8 individuals share one
    bit-identical genome (a fully converged monoculture), and less than
    1.0 the moment even one individual's genome differs."""
    grid_size = 4
    with GridWorld(seed=0, grid_size=grid_size) as world:
        all_zero = [Individual(genome=Genome(bits=np.zeros(448, dtype=np.uint8))) for _ in range(N_INDIVIDUALS)]
        world.cells[0][0].individuals = [Individual(genome=ind.genome.copy()) for ind in all_zero]
        mixed = [Individual(genome=Genome(bits=np.zeros(448, dtype=np.uint8))) for _ in range(N_INDIVIDUALS - 1)]
        mixed.append(Individual(genome=Genome(bits=np.ones(448, dtype=np.uint8))))
        world.cells[0][1].individuals = mixed

        scores = world.run_day(want_snapshot=True)
        snap = world.snapshot(scores)

        assert snap["day"] == 1
        assert snap["score"].shape == (grid_size, grid_size)
        assert snap["speech_by_stimulus"].shape == (grid_size, grid_size, N_STIM_PAIRS)
        assert snap["purity"].shape == (grid_size, grid_size)
        assert snap["purity"][0, 0] == 1.0
        assert snap["purity"][0, 1] == (N_INDIVIDUALS - 1) / N_INDIVIDUALS


def test_snapshot_requires_want_snapshot():
    with GridWorld(seed=0, grid_size=4) as world:
        scores = world.run_day()  # want_snapshot defaults to False
        with pytest.raises(RuntimeError):
            world.snapshot(scores)


def test_snapshot_purity_reflects_pre_reproduction_population():
    """Purity must describe the same population that produced this day's
    score -- computed before that day's local_reproduce (or any later
    festival/wind) mutates the cell, not after. Verified by spying on
    local_reproduce (a plain instance method, unlike the numpy Generator
    calls elsewhere in this file, which can't be monkeypatched) to
    capture exactly what the population looked like at the moment
    reproduction was about to run, and checking last_day_purity matches
    that captured snapshot rather than whatever comes after it."""
    grid_size = 4
    with GridWorld(seed=0, grid_size=grid_size, wind_period=None, festival_period=None) as world:
        cell = world.cells[0][0]
        distinct = [Individual(genome=Genome(bits=np.eye(448, dtype=np.uint8)[i])) for i in range(N_INDIVIDUALS)]
        cell.individuals = distinct

        captured = {}
        original_reproduce = cell.local_reproduce

        def spy_reproduce(scores):
            captured["bits_at_call_time"] = [ind.genome.bits.copy() for ind in cell.individuals]
            return original_reproduce(scores)

        cell.local_reproduce = spy_reproduce

        world.run_day(want_snapshot=True)

        expected_purity = max(Counter(b.tobytes() for b in captured["bits_at_call_time"]).values()) / N_INDIVIDUALS
        assert world.last_day_purity[0, 0] == expected_purity


def test_sample_cells_reproducible_and_survives_checkpoint(tmp_path):
    world_a = GridWorld(seed=42, grid_size=8)
    world_b = GridWorld(seed=42, grid_size=8)
    assert len(world_a.sample_cells) == N_SAMPLE_CELLS
    assert world_a.sample_cells == world_b.sample_cells
    for row, col in world_a.sample_cells:
        assert 0 <= row < 8 and 0 <= col < 8

    checkpoint_path = str(tmp_path / "checkpoint.pkl")
    world_a.save_checkpoint(checkpoint_path)
    resumed = GridWorld.load_checkpoint(checkpoint_path)
    assert resumed.sample_cells == world_a.sample_cells
