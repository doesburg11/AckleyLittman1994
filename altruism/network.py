"""Network execution: two synchronous linear-threshold passes per step
(Section 2.2's "reacts" description).
"""

import numpy as np

from altruism.genome import Genome, develop
from altruism.units import HEAR, HIDDEN, N_UNITS, PRED, FOOD, TO_R, MOVE, SPEAK, TRUE, LOCATION_UNIT


class Network:
    def __init__(self, genome: Genome):
        connections = develop(genome)
        self.incoming: dict[int, list[tuple[int, int]]] = {u: [] for u in range(13, N_UNITS)}
        for from_unit, to_unit, weight in connections:
            if 13 <= to_unit < N_UNITS:
                self.incoming[to_unit].append((from_unit, weight))
        self.values = np.zeros(N_UNITS, dtype=np.int16)

    def reset_trial(self, genome: Genome):
        """Called at the start of every trial -- initial states for the 19
        computed units (13..31) are read from the genome fresh each trial;
        nothing here persists across trials (only across steps WITHIN a
        trial)."""
        self.values[:] = 0
        self.values[13:N_UNITS] = genome.initial_states
        self.values[TRUE] = 1

    def react(self, pred: int, food: int, location: str, hearing: np.ndarray) -> np.ndarray:
        """One step: set input units from the environment, then two
        synchronous linear-threshold passes over units 13..31. Returns the
        full 32-value state array (callers read SPEAK/MOVE/TO_R off it)."""
        self.values[TRUE] = 1
        self.values[PRED] = pred
        self.values[FOOD] = food
        for loc, unit in LOCATION_UNIT.items():
            self.values[unit] = 1 if loc == location else 0
        self.values[HEAR[0]:HEAR[-1] + 1] = hearing

        for _pass in range(2):
            new_values = self.values.copy()
            for unit in range(13, N_UNITS):
                total = 0
                for from_unit, weight in self.incoming[unit]:
                    total += int(self.values[from_unit]) * weight
                new_values[unit] = 1 if total > 0 else 0
            self.values = new_values

        return self.values

    @property
    def speech(self) -> np.ndarray:
        return self.values[SPEAK[0]:SPEAK[-1] + 1]

    @property
    def move(self) -> bool:
        return bool(self.values[MOVE])

    @property
    def to_r(self) -> bool:
        return bool(self.values[TO_R])

    @property
    def hidden_state(self) -> np.ndarray:
        return self.values[HIDDEN[0]:HIDDEN[-1] + 1]
