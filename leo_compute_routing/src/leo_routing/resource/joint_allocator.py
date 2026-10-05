from dataclasses import dataclass

from .cpu_allocator import allocate_cpu
from .link_allocator import allocate_links


@dataclass(frozen=True)
class Allocation:
    cpu: dict
    links: dict

    def cpu_rates(self):
        return {task: rate for group in self.cpu.values() for task, rate in group.items()}

    def link_rates(self):
        return {task: rate for group in self.links.values() for task, rate in group.items()}


def allocate_resources(jobs, graph, cpu_capacities, mode="sqrt"):
    jobs = tuple(jobs)
    return Allocation(allocate_cpu(jobs, cpu_capacities, mode), allocate_links(jobs, graph, mode))
