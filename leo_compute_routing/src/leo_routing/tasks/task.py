from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Task:
    task_id: int
    source_sat: int
    data_bits: float
    cycles_per_bit: float
    deadline_seconds: float
    arrival_slot: int

    def __post_init__(self):
        for value in (self.task_id, self.source_sat, self.arrival_slot):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Task identifiers and arrival slots must be nonnegative integers")
        for value in (self.data_bits, self.cycles_per_bit, self.deadline_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Task size, complexity and deadline must be positive")
        if not math.isfinite(self.total_cycles):
            raise ValueError("Task cycles overflowed")

    @property
    def total_cycles(self):
        return self.data_bits * self.cycles_per_bit
