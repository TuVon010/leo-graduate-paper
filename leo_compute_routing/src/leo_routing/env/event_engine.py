"""Fluid store-and-forward simulator with exact events within piecewise slots.

Tasks share only their CURRENT link/CPU. Rates are reallocated on arrivals,
completions and topology changes, so past-slot tasks continue consuming budget.
Prediction features never substitute for measured completion timestamps.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..compute.compute_queue import queue_cycles, inflight_cycles
from ..network.link_model import edge_key
from ..resource.joint_allocator import allocate_resources

ACTIVE_STAGES = ("tx", "prop", "cpu")
TIME_EPS = 1e-10


@dataclass
class RuntimeTask:
    task: object
    action: object
    arrival_time: float
    stage: str
    remaining_bits: float
    remaining_cycles: float
    hop: int = 0
    wake_time: float = float("inf")
    propagation_seconds: float = 0.0
    cpu_arrival_time: object = None
    terminal_time: object = None
    deadline_reported: bool = False
    failure_reason: str = ""
    transmission_seconds: float = 0.0
    propagation_elapsed_seconds: float = 0.0
    cpu_service_seconds: float = 0.0

    @property
    def absolute_deadline(self):
        return self.arrival_time + self.task.deadline_seconds

    @property
    def edge(self):
        return edge_key(*self.action.path[self.hop:self.hop + 2])


class EventEngine:
    def __init__(self, trace, cpu_capacities, allocation_mode="sqrt", drop_at_deadline=False,
                 measurement_window=None):
        self.trace = trace
        self.cpu_capacities = np.asarray(cpu_capacities, dtype=float).copy()
        if (self.cpu_capacities.shape != (trace.satellite_count,) or
                np.any(self.cpu_capacities <= 0) or not np.all(np.isfinite(self.cpu_capacities))):
            raise ValueError("One positive finite CPU capacity is required per satellite")
        if allocation_mode not in ("sqrt", "equal"):
            raise ValueError("Unknown resource allocation mode")
        self.allocation_mode, self.drop_at_deadline = allocation_mode, drop_at_deadline
        self.now, self.jobs = 0.0, {}
        self._active_jobs = {}
        self.holding_cost_seconds = 0.0
        self.deadline_misses = self.route_failures = self.completions = 0
        self.cpu_cycles_processed = self.transmitted_bits = 0.0
        self.cpu_busy_capacity_integral = self.cpu_total_capacity_integral = 0.0
        self.link_busy_capacity_integral = self.link_total_capacity_integral = 0.0
        self.max_cpu_budget_ratio = self.max_link_budget_ratio = 0.0
        self.measurement_window = measurement_window
        if measurement_window is not None:
            start, end = measurement_window
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("Invalid measurement window")
        self.measured_seconds = 0.0
        self.measured_cpu_busy_integral = self.measured_cpu_total_integral = 0.0
        self.measured_link_busy_integral = self.measured_link_total_integral = 0.0

    @property
    def active(self):
        return tuple(self._active_jobs.values())

    def queue_cycles(self):
        return queue_cycles(self.active, self.trace.satellite_count)

    def inflight_cycles(self):
        return inflight_cycles(self.active, self.trace.satellite_count)

    def admit(self, tasks, actions):
        tasks = tuple(tasks)
        expected = {task.task_id for task in tasks}
        if len(expected) != len(tasks) or set(actions) != expected:
            raise ValueError("Each arriving task requires exactly one action")
        # Validate the entire batch before changing any state.
        graph = self.trace.graph(self.trace.slot_at(self.now))
        for task in tasks:
            action = actions[task.task_id]
            if task.task_id in self.jobs or action.task_id != task.task_id:
                raise ValueError("Duplicate task or mismatched action ID")
            if abs(task.arrival_slot * self.trace.slot_seconds - self.now) > TIME_EPS:
                raise ValueError("Tasks must arrive at their declared slot boundary")
            if action.path[0] != task.source_sat or action.compute_sat >= self.trace.satellite_count:
                raise ValueError("Invalid action source/destination")
            if any(not graph.has_edge(i, j) for i, j in zip(action.path, action.path[1:])):
                raise ValueError("Action path uses an unavailable current link")
        for task in tasks:
            action = actions[task.task_id]
            stage = "cpu" if action.is_local else "tx"
            job = RuntimeTask(task, action, self.now, stage, task.data_bits, task.total_cycles)
            if stage == "cpu":
                job.cpu_arrival_time = self.now
            self.jobs[task.task_id] = job
            self._active_jobs[task.task_id] = job

    def _finish(self, job, status, reason=""):
        job.stage, job.terminal_time, job.failure_reason = status, self.now, reason
        self._active_jobs.pop(job.task.task_id, None)
        if status == "completed":
            self.completions += 1
        elif status == "route_failed":
            self.route_failures += 1

    @staticmethod
    def _residual_done(value, original):
        return value <= max(1e-7, original * 1e-12)

    def _resolve_instantaneous_events(self):
        # Finish an interval's transmission BEFORE applying new-slot link loss.
        # A last bit sent exactly at the old slot end is therefore delivered.
        changed = True
        while changed:
            changed = False
            for job in self.active:
                if job.stage == "cpu" and self._residual_done(job.remaining_cycles, job.task.total_cycles):
                    job.remaining_cycles = 0.0
                    self._finish(job, "completed")
                    changed = True
                elif job.stage == "tx" and self._residual_done(job.remaining_bits, job.task.data_bits):
                    job.remaining_bits = 0.0
                    job.stage, job.wake_time = "prop", self.now + job.propagation_seconds
                    changed = True
                elif job.stage == "prop" and job.wake_time <= self.now + TIME_EPS:
                    job.hop += 1
                    if job.hop == job.action.hops:
                        job.stage, job.cpu_arrival_time = "cpu", self.now
                    else:
                        job.stage, job.remaining_bits = "tx", job.task.data_bits
                    changed = True
        for job in self.active:
            if not job.deadline_reported and self.now >= job.absolute_deadline - TIME_EPS:
                job.deadline_reported = True
                self.deadline_misses += 1
                if self.drop_at_deadline:
                    self._finish(job, "timed_out", "deadline")
        if self.active:
            graph = self.trace.graph(self.trace.slot_at(self.now))
            for job in self.active:
                if job.stage == "tx" and not graph.has_edge(*job.edge):
                    self._finish(job, "route_failed", "link_unavailable")

    def advance(self, until):
        if not math.isfinite(until) or until < self.now - TIME_EPS:
            raise ValueError("Engine time must advance monotonically")
        if until >= self.trace.slots * self.trace.slot_seconds - TIME_EPS:
            raise ValueError("Insufficient topology coverage")
        before = (self.holding_cost_seconds, self.deadline_misses, self.route_failures, self.completions)
        while self.now < until - TIME_EPS:
            self._resolve_instantaneous_events()
            slot = self.trace.slot_at(self.now)
            graph, active = self.trace.graph(slot), self.active
            allocation = allocate_resources(active, graph, self.cpu_capacities, self.allocation_mode)
            cpu_rates, link_rates = allocation.cpu_rates(), allocation.link_rates()
            # Large absolute times can make a tiny residual's completion fall
            # below one floating-point time ULP. Account for that residual and
            # resolve its event at the same representable timestamp.
            roundoff_completion = False
            for job in active:
                if job.stage == "cpu" and self.now + job.remaining_cycles / cpu_rates[job.task.task_id] == self.now:
                    self.cpu_cycles_processed += job.remaining_cycles
                    job.remaining_cycles = 0.0
                    roundoff_completion = True
                elif job.stage == "tx" and self.now + job.remaining_bits / link_rates[job.task.task_id] == self.now:
                    self.transmitted_bits += job.remaining_bits
                    job.remaining_bits = 0.0
                    job.propagation_seconds = graph.edges[job.edge]["propagation_seconds"]
                    roundoff_completion = True
            if roundoff_completion:
                continue
            boundary = (slot + 1) * self.trace.slot_seconds
            next_time = min(until, boundary)
            for job in active:
                if not job.deadline_reported:
                    next_time = min(next_time, job.absolute_deadline)
                if job.stage == "cpu":
                    next_time = min(next_time, self.now + job.remaining_cycles / cpu_rates[job.task.task_id])
                elif job.stage == "tx":
                    job.propagation_seconds = graph.edges[job.edge]["propagation_seconds"]
                    next_time = min(next_time, self.now + job.remaining_bits / link_rates[job.task.task_id])
                else:
                    next_time = min(next_time, job.wake_time)
            delta = next_time - self.now
            if delta <= 0:
                raise RuntimeError("Event engine failed to advance at time=%s, next=%s, slot=%s" %
                                   (self.now, next_time, slot))
            self.holding_cost_seconds += len(active) * delta
            self.cpu_total_capacity_integral += float(self.cpu_capacities.sum()) * delta
            self.link_total_capacity_integral += sum(d["capacity_bps"] for _, _, d in graph.edges(data=True)) * delta
            measured_delta = delta if self.measurement_window is None else max(
                0.0, min(next_time, self.measurement_window[1]) - max(self.now, self.measurement_window[0]))
            self.measured_seconds += measured_delta
            self.measured_cpu_total_integral += float(self.cpu_capacities.sum()) * measured_delta
            self.measured_link_total_integral += sum(d["capacity_bps"] for _, _, d in graph.edges(data=True)) * measured_delta
            for satellite, group in allocation.cpu.items():
                amount = sum(group.values())
                self.cpu_busy_capacity_integral += amount * delta
                self.measured_cpu_busy_integral += amount * measured_delta
                self.max_cpu_budget_ratio = max(self.max_cpu_budget_ratio, amount / self.cpu_capacities[satellite])
            for edge, group in allocation.links.items():
                amount = sum(group.values())
                self.link_busy_capacity_integral += amount * delta
                self.measured_link_busy_integral += amount * measured_delta
                self.max_link_budget_ratio = max(self.max_link_budget_ratio, amount / graph.edges[edge]["capacity_bps"])
            for job in active:
                if job.stage == "cpu":
                    processed = min(job.remaining_cycles, cpu_rates[job.task.task_id] * delta)
                    self.cpu_cycles_processed += processed
                    job.remaining_cycles = max(0.0, job.remaining_cycles - processed)
                    job.cpu_service_seconds += delta
                elif job.stage == "tx":
                    processed = min(job.remaining_bits, link_rates[job.task.task_id] * delta)
                    self.transmitted_bits += processed
                    job.remaining_bits = max(0.0, job.remaining_bits - processed)
                    job.transmission_seconds += delta
                else:
                    job.propagation_elapsed_seconds += delta
            self.now = next_time
        self.now = until
        self._resolve_instantaneous_events()
        after = (self.holding_cost_seconds, self.deadline_misses, self.route_failures, self.completions)
        return dict(zip(("holding_cost_seconds", "new_deadline_misses", "new_route_failures", "new_completions"),
                        (a - b for a, b in zip(after, before))))

    def censor_remaining(self):
        for job in self.active:
            self._finish(job, "censored", "drain_limit")

    def task_records(self):
        records = []
        for task_id, job in sorted(self.jobs.items()):
            terminal = job.terminal_time
            elapsed = (self.now if terminal is None else terminal) - job.arrival_time
            complete = job.stage == "completed"
            records.append({"task_id": task_id, "arrival_slot": job.task.arrival_slot,
                            "source_sat": job.task.source_sat, "compute_sat": job.action.compute_sat,
                            "path": list(job.action.path), "hops": job.action.hops,
                            "status": job.stage, "failure_reason": job.failure_reason,
                            "deadline_seconds": job.task.deadline_seconds,
                            "completion_delay_s": elapsed if complete else None,
                            "sojourn_s": elapsed, "deadline_missed": job.deadline_reported,
                            "success": complete and elapsed <= job.task.deadline_seconds + TIME_EPS,
                            "transmission_s": job.transmission_seconds,
                            "propagation_s": job.propagation_elapsed_seconds,
                            "cpu_service_s": job.cpu_service_seconds,
                            "remaining_cycles": job.remaining_cycles})
        return records
