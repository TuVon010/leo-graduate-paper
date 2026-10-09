"""Prediction-only link/CPU calendars, seeded with currently committed jobs.

No unseen task arrivals are used. Exclusive reference-rate bookings approximate
competition; the real engine continues fluid sharing with KKT allocation.
"""
from dataclasses import dataclass
from types import SimpleNamespace

from ..network.contact_plan import EPS


def first_free_interval(bookings, earliest, duration):
    start = earliest
    for left, right in sorted(bookings):
        if right <= start + EPS:
            continue
        if left >= start + duration - EPS:
            break
        start = max(start, right)
    return start, start + duration


@dataclass(frozen=True)
class ReservationEstimate:
    route: object
    cpu_interval: tuple
    completion_seconds: float
    deadline_margin_seconds: float
    contact_ok: bool
    fully_checked: bool
    deadline_ok: bool

    @property
    def feasible(self):
        return self.contact_ok and self.deadline_ok


class ReservationCalendar:
    def __init__(self, plan, capacities, rate_fraction, allow_unverified=True, deadline_mask=True):
        self.plan, self.capacities, self.rate_fraction = plan, capacities, rate_fraction
        self.allow_unverified, self.deadline_mask = allow_unverified, deadline_mask
        self.links, self.cpu = {}, {}

    @classmethod
    def from_observation(cls, observation):
        calendar = cls(observation.contact_plan, observation.cpu_capacities,
                       observation.reference_rate_fraction, observation.allow_unverified_future,
                       observation.deadline_mask_enabled)
        now = calendar.plan.start
        # CPU-resident work has arrived; do not count it again as in-flight work.
        for satellite, cycles in enumerate(observation.cpu_queue_cycles):
            if cycles > 0:
                calendar.cpu[satellite] = [(now, now + cycles / calendar.capacities[satellite])]
        for job in sorted(observation.active_jobs, key=lambda j: (j.deadline_remaining_seconds, j.task_id)):
            if job.stage == "cpu":
                continue
            task = SimpleNamespace(data_bits=job.original_bits, total_cycles=job.remaining_cycles,
                                   deadline_seconds=job.deadline_remaining_seconds)
            start = now + job.seconds_to_wake if job.stage == "prop" else now
            path = job.path[job.hop + 1:] if job.stage == "prop" else job.path[job.hop:]
            route = calendar.plan.predict(task, path, calendar.rate_fraction, calendar.links, start,
                                          job.remaining_bits if job.stage == "tx" else None)
            arrival = start + route.route_seconds
            cpu = first_free_interval(calendar.cpu.get(job.compute_sat, ()), arrival,
                                      job.remaining_cycles / calendar.capacities[job.compute_sat])
            calendar._commit(route, job.compute_sat, cpu)
        return calendar

    def estimate(self, task, action, route=None):
        if route is None:
            route = self.plan.predict(task, action.path, self.rate_fraction, self.links)
        arrival = self.plan.start + route.route_seconds
        cpu = first_free_interval(self.cpu.get(action.compute_sat, ()), arrival,
                                  task.total_cycles / self.capacities[action.compute_sat])
        completion = cpu[1] - self.plan.start
        contact_ok = route.topology_feasible and (route.fully_checked or self.allow_unverified or not self.plan.future_enabled)
        margin = task.deadline_seconds - completion
        return ReservationEstimate(route, cpu, completion, margin, contact_ok, route.fully_checked,
                                   margin >= -EPS or not self.deadline_mask)

    def _commit(self, route, satellite, cpu):
        for hop in route.hops:
            self.links.setdefault(hop.edge, []).extend(hop.segments)
            self.links[hop.edge].sort()
            if not hop.feasible:
                break  # known failed input cannot consume downstream service
        if route.topology_feasible:
            self.cpu.setdefault(satellite, []).append(cpu)
            self.cpu[satellite].sort()

    def commit(self, task, action, estimate=None):
        if estimate is None:
            estimate = self.estimate(task, action)
        self._commit(estimate.route, action.compute_sat, estimate.cpu_interval)
        return estimate
