"""Global level (Section 2.3): a torus grid of subpopulations, with wind
migration and festival migration/reproduction layered on top of the local
level's day/scoring/reproduction cycle (world.py).
"""

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


class GridWorld:
    """A grid_size x grid_size torus of LocalWorld subpopulations. Runs
    the day cycle for every cell, then applies festival reproduction
    (replacing local reproduction for cells in a festival quad) and/or
    wind migration, per the periodic schedules given.

    `wind_period` / `festival_period` of None means that mechanism never
    fires; an integer period p means it fires when `day % p == 0` (day
    counting starts at 1, so period 5 fires on days 5, 10, 15, ...).
    """

    def __init__(
        self,
        seed: int,
        grid_size: int = GRID_SIZE_DEFAULT,
        wind_period: int | None = None,
        festival_period: int | None = None,
    ):
        assert grid_size % 2 == 0, "grid_size must be even (2x2 quads must tile it exactly)"
        self.grid_size = grid_size
        self.wind_period = wind_period
        self.festival_period = festival_period
        self.day = 0
        self._festival_phase = 0

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
        """Runs one full day: every cell's 36-trial scoring, then
        reproduction (festival replaces local reproduction on a festival
        day), then wind migration on a windy day. Returns today's scores,
        shape (grid_size, grid_size, N_INDIVIDUALS)."""
        self.day += 1
        scores = np.empty((self.grid_size, self.grid_size, N_INDIVIDUALS))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                scores[row, col] = self.cells[row][col].run_day()

        is_festival = self.festival_period is not None and self.day % self.festival_period == 0
        is_windy = self.wind_period is not None and self.day % self.wind_period == 0

        if is_festival:
            self._run_festival(scores)
        else:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    self.cells[row][col].local_reproduce(scores[row, col])

        if is_windy:
            self._run_wind()

        return scores

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
