"""Global level (Section 2.3): a torus grid of subpopulations, with wind
migration and festival migration/reproduction layered on top of the local
level's day/scoring/reproduction cycle (world.py).
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from altruism.genome import crossover
from altruism.world import Individual, LocalWorld, N_INDIVIDUALS, select_parents_and_victim

GRID_SIZE_DEFAULT = 128
N_QUAD_INDIVIDUALS = 4 * N_INDIVIDUALS  # 32
FESTIVAL_TOP_K = N_QUAD_INDIVIDUALS // 4  # top quarter = 8

# 8 compass directions as (row_offset, col_offset); torus wraparound is
# applied via `% grid_size` wherever these are used.
COMPASS_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Margolus-neighborhood phase cycle for festival quad grouping: the 2x2
# block origin shifts through these (row, col) offsets on successive
# festival days, so after 4 festivals a given cell has been grouped, in
# some quad, with all 8 of its neighbors.
PHASE_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _score_chunk(local_worlds: list[LocalWorld], do_local_reproduce: bool):
    """Worker-process entry point: runs each cell's day (and, unless a
    festival is about to override it, its local reproduction) in
    isolation. Each LocalWorld carries its own independent rng stream, so
    a cell's result depends only on that cell -- never on which worker
    ran it or what order cells within a chunk were processed in, which is
    what makes this safe to parallelize without changing the outcome."""
    scores = []
    for local_world in local_worlds:
        day_scores = local_world.run_day()
        if do_local_reproduce:
            local_world.local_reproduce(day_scores)
        scores.append(day_scores)
    return local_worlds, scores


def _chunk(items: list, n: int) -> list[list]:
    n = max(1, n)
    size = -(-len(items) // n)  # ceil division
    return [items[i:i + size] for i in range(0, len(items), size)]


class GridWorld:
    """A grid_size x grid_size torus of LocalWorld subpopulations. Runs
    the day cycle for every cell, then applies festival reproduction
    (replacing local reproduction for cells in a festival quad) and/or
    wind migration, per the periodic schedules given.

    `wind_period` / `festival_period` of None means that mechanism never
    fires; an integer period p means it fires when `day % p == 0` (day
    counting starts at 1, so period 5 fires on days 5, 10, 15, ...).

    `n_workers`, if given as an int > 1, runs each day's per-cell scoring
    (and local reproduction) across a persistent pool of that many worker
    processes instead of a single-threaded loop -- the dominant cost by
    far (the 36-trial scoring loop), and the one part of the day cycle
    that's embarrassingly parallel per-cell. Festival and wind stay
    single-threaded in the main process: they're cheap by comparison and
    need cross-cell coordination (a festival quad spans 4 cells; wind is
    one global, simultaneous swap), so parallelizing them would add real
    complexity for negligible time saved. Close the pool with `close()`
    or use `GridWorld(...) as world:` when done.
    """

    def __init__(
        self,
        seed: int,
        grid_size: int = GRID_SIZE_DEFAULT,
        wind_period: int | None = None,
        festival_period: int | None = None,
        n_workers: int | None = None,
    ):
        assert grid_size % 2 == 0, "grid_size must be even (2x2 quads must tile it exactly)"
        self.grid_size = grid_size
        self.wind_period = wind_period
        self.festival_period = festival_period
        self.day = 0
        self._festival_phase = 0
        self._n_workers = n_workers
        self._executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers and n_workers > 1 else None
        self.last_day_speech = np.zeros((grid_size, grid_size, N_INDIVIDUALS))

        # One independent child RNG stream per cell (for that cell's own
        # trial/local-reproduction randomness) plus one more for the grid
        # itself (wind direction/emigrant draws, festival parent/victim
        # draws) -- numpy's own recommended pattern for independent,
        # reproducible parallel streams.
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(grid_size * grid_size + 1)
        self.rng = np.random.default_rng(child_seeds[-1])
        self.cells: list[list[LocalWorld]] = [
            [
                LocalWorld(rng=np.random.default_rng(child_seeds[row * grid_size + col]))
                for col in range(grid_size)
            ]
            for row in range(grid_size)
        ]

    def run_day(self) -> np.ndarray:
        """Runs one full day: every cell's 36-trial scoring (plus local
        reproduction, unless today is a festival day for that cell),
        then festival reproduction if today is a festival day, then wind
        migration if today is a windy day. Returns today's scores, shape
        (grid_size, grid_size, N_INDIVIDUALS). Also refreshes
        `last_day_speech` (same shape) -- each cell's
        `last_day_speech_activity`, a cheap communication-activity proxy,
        collected as a side effect of scoring."""
        self.day += 1
        is_festival = self.festival_period is not None and self.day % self.festival_period == 0
        is_windy = self.wind_period is not None and self.day % self.wind_period == 0

        if self._executor is not None:
            scores, speech = self._run_scoring_parallel(do_local_reproduce=not is_festival)
        else:
            scores, speech = self._run_scoring_serial(do_local_reproduce=not is_festival)
        self.last_day_speech = speech

        if is_festival:
            self._run_festival(scores)
        if is_windy:
            self._run_wind()

        return scores

    def _run_scoring_serial(self, do_local_reproduce: bool) -> tuple[np.ndarray, np.ndarray]:
        scores = np.empty((self.grid_size, self.grid_size, N_INDIVIDUALS))
        speech = np.empty((self.grid_size, self.grid_size, N_INDIVIDUALS))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = self.cells[row][col]
                scores[row, col] = cell.run_day()
                speech[row, col] = cell.last_day_speech_activity
                if do_local_reproduce:
                    cell.local_reproduce(scores[row, col])
        return scores, speech

    def _run_scoring_parallel(self, do_local_reproduce: bool) -> tuple[np.ndarray, np.ndarray]:
        flat_cells = [self.cells[row][col] for row in range(self.grid_size) for col in range(self.grid_size)]
        chunks = _chunk(flat_cells, self._n_workers)
        futures = [self._executor.submit(_score_chunk, chunk, do_local_reproduce) for chunk in chunks]

        scores = np.empty((self.grid_size, self.grid_size, N_INDIVIDUALS))
        speech = np.empty((self.grid_size, self.grid_size, N_INDIVIDUALS))
        flat_index = 0
        for future in futures:
            updated_cells, chunk_scores = future.result()
            for local_world, day_scores in zip(updated_cells, chunk_scores):
                row, col = divmod(flat_index, self.grid_size)
                self.cells[row][col] = local_world
                scores[row, col] = day_scores
                speech[row, col] = local_world.last_day_speech_activity
                flat_index += 1
        return scores, speech

    def close(self):
        """Shuts down the worker pool, if one was created. Safe to call
        even when n_workers was never set (a no-op)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "GridWorld":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __getstate__(self) -> dict:
        """A ProcessPoolExecutor can't be pickled -- excluded from the
        saved state, never from the live object, so taking a checkpoint
        never disturbs a run in progress."""
        state = self.__dict__.copy()
        state["_executor"] = None
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    def save_checkpoint(self, path: str):
        """Atomic write (temp file + os.replace) so a crash or kill
        mid-write can never leave a corrupt checkpoint behind."""
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self, f)
        os.replace(tmp_path, path)

    @staticmethod
    def load_checkpoint(path: str, n_workers: int | None = None) -> "GridWorld":
        """`n_workers` overrides whatever pool size the checkpoint was
        saved with -- fine to resume with a different worker count,
        since the executor itself is never part of the pickled state."""
        with open(path, "rb") as f:
            world: GridWorld = pickle.load(f)
        world._n_workers = n_workers if n_workers is not None else world._n_workers
        world._executor = (
            ProcessPoolExecutor(max_workers=world._n_workers)
            if world._n_workers and world._n_workers > 1
            else None
        )
        return world

    def _run_wind(self):
        """One global wind direction; independently in every cell, one
        individual is picked uniformly at random to emigrate; all
        selected individuals move downwind simultaneously, each landing in
        the exact slot its destination cell's own emigrant vacated (a
        population-preserving swap, computed from a snapshot so the order
        cells are processed in cannot matter)."""
        direction = COMPASS_OFFSETS[self.rng.integers(0, 8)]
        row_offset, col_offset = direction
        emigrant_index = self.rng.integers(0, N_INDIVIDUALS, size=(self.grid_size, self.grid_size))

        emigrants = [
            [self.cells[row][col].individuals[emigrant_index[row, col]] for col in range(self.grid_size)]
            for row in range(self.grid_size)
        ]

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                dest_row = (row + row_offset) % self.grid_size
                dest_col = (col + col_offset) % self.grid_size
                dest_slot = emigrant_index[dest_row, dest_col]
                self.cells[dest_row][dest_col].individuals[dest_slot] = emigrants[row][col]

    def _quad_cells(self, g_row: int, g_col: int, phase_offset: tuple[int, int]) -> list[tuple[int, int]]:
        """The 4 member cells of quad (g_row, g_col) under the given
        (row_offset, col_offset) phase, on this torus grid."""
        row_offset, col_offset = phase_offset
        r0 = (2 * g_row + row_offset) % self.grid_size
        r1 = (2 * g_row + row_offset + 1) % self.grid_size
        c0 = (2 * g_col + col_offset) % self.grid_size
        c1 = (2 * g_col + col_offset + 1) % self.grid_size
        return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    def _run_festival(self, scores: np.ndarray):
        """Festivals held independently in each of the grid's non-
        overlapping 2x2 quads at the current phase: the 32 individuals
        across the quad's 4 cells are ranked by today's score; two parents
        drawn uniformly from the top quarter; one offspring via crossover;
        one of the 32 killed uniformly at random and replaced in whichever
        cell/slot it belonged to. Advances the phase (mod 4) once, after
        all quads for today are done."""
        phase_offset = PHASE_OFFSETS[self._festival_phase]
        for g_row in range(self.grid_size // 2):
            for g_col in range(self.grid_size // 2):
                quad_cells = self._quad_cells(g_row, g_col, phase_offset)
                quad_scores = np.array([scores[r, c, slot] for (r, c) in quad_cells for slot in range(N_INDIVIDUALS)])

                parent_a, parent_b, victim = select_parents_and_victim(
                    quad_scores, top_k=FESTIVAL_TOP_K, rng=self.rng
                )

                def genome_at(flat_index):
                    cell_index, slot = divmod(flat_index, N_INDIVIDUALS)
                    r, c = quad_cells[cell_index]
                    return self.cells[r][c].individuals[slot].genome

                child_genome = crossover(genome_at(parent_a), genome_at(parent_b), self.rng)
                victim_cell_index, victim_slot = divmod(victim, N_INDIVIDUALS)
                victim_row, victim_col = quad_cells[victim_cell_index]
                self.cells[victim_row][victim_col].individuals[victim_slot] = Individual(genome=child_genome)

        self._festival_phase = (self._festival_phase + 1) % 4
