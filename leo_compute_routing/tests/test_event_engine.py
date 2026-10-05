import numpy as np
import pytest

from leo_routing.env.event_engine import EventEngine
from leo_routing.routing.action_builder import RoutingAction
from leo_routing.tasks.task import Task


def make_task(task_id=0, source=0, bits=100.0, complexity=1.0, deadline=10.0, slot=0):
    return Task(task_id, source, bits, complexity, deadline, slot)


def test_local_task_matches_analytic_time_and_exact_deadline(trace_factory):
    engine = EventEngine(trace_factory(), [100, 100])
    task = make_task(deadline=1.0)
    engine.admit([task], {0: RoutingAction(0, 0, (0,))})
    engine.advance(2.0)
    record = engine.task_records()[0]
    assert record["completion_delay_s"] == pytest.approx(1.0)
    assert record["success"] and engine.deadline_misses == 0
    assert engine.cpu_cycles_processed == pytest.approx(100)


def test_transmission_and_propagation_precede_cpu_admission(trace_factory):
    engine = EventEngine(trace_factory(propagation_seconds=0.2), [1000, 1000])
    task = make_task()
    engine.admit([task], {0: RoutingAction(0, 1, (0, 1))})
    engine.advance(1.1)
    assert engine.queue_cycles().sum() == 0
    assert engine.inflight_cycles()[1] == 100
    assert engine.cpu_cycles_processed == 0
    engine.advance(1.2)
    assert engine.queue_cycles()[1] == pytest.approx(100)
    engine.advance(1.5)
    row = engine.task_records()[0]
    assert row["completion_delay_s"] == pytest.approx(1.3)
    assert row["transmission_s"] + row["propagation_s"] + row["cpu_service_s"] == pytest.approx(1.3)


def test_store_and_forward_multihop(trace_factory):
    engine = EventEngine(trace_factory(satellites=3, edges=((0, 1), (1, 2)), propagation_seconds=0.1),
                         [1000, 1000, 1000])
    engine.admit([make_task()], {0: RoutingAction(0, 2, (0, 1, 2))})
    engine.advance(3.0)
    assert engine.task_records()[0]["completion_delay_s"] == pytest.approx(2.3)
    assert engine.transmitted_bits == pytest.approx(200)


def test_opposite_directions_share_one_capacity(trace_factory):
    engine = EventEngine(trace_factory(), [1000, 1000])
    tasks = [make_task(), make_task(1, source=1)]
    engine.admit(tasks, {0: RoutingAction(0, 1, (0, 1)), 1: RoutingAction(1, 0, (1, 0))})
    engine.advance(1.0)
    assert [job.remaining_bits for job in engine.active] == pytest.approx([50, 50])
    assert engine.queue_cycles().sum() == 0
    engine.advance(3.0)
    assert [r["completion_delay_s"] for r in engine.task_records()] == pytest.approx([2.1, 2.1])
    assert engine.max_link_budget_ratio <= 1 + 1e-12


def test_old_and_new_slot_tasks_compete(trace_factory):
    engine = EventEngine(trace_factory(capacity=10.0), [1000, 1000])
    first = make_task(bits=20.0)
    engine.admit([first], {0: RoutingAction(0, 1, (0, 1))})
    engine.advance(1.0)
    second = make_task(1, bits=10.0, slot=1)
    engine.admit([second], {1: RoutingAction(1, 1, (0, 1))})
    engine.advance(2.0)
    assert engine.jobs[0].remaining_bits > 0
    assert engine.jobs[1].remaining_bits > 0
    engine.advance(4.0)
    assert engine.jobs[1].terminal_time == pytest.approx(3.01)
    assert engine.transmitted_bits == pytest.approx(30)


def test_cpu_reallocation_and_workload_conservation(trace_factory):
    engine = EventEngine(trace_factory(), [100, 100])
    tasks = [make_task(bits=100), make_task(1, bits=400)]
    engine.admit(tasks, {0: RoutingAction(0, 0, (0,)), 1: RoutingAction(1, 0, (0,))})
    for time in (1.0, 2.0, 3.0, 4.0, 6.0):
        engine.advance(time)
        residual = sum(job.remaining_cycles for job in engine.jobs.values())
        assert engine.cpu_cycles_processed + residual == pytest.approx(500)
        assert np.all(engine.queue_cycles() >= 0)
    assert [r["completion_delay_s"] for r in engine.task_records()] == pytest.approx([3, 5])
    assert engine.max_cpu_budget_ratio <= 1 + 1e-12


@pytest.mark.parametrize("bits,expected_status", [(100.0, "completed"), (101.0, "route_failed")])
def test_link_disappearing_at_exact_boundary(trace_factory, bits, expected_status):
    trace = trace_factory(propagation_seconds=0.2, unavailable_slots=(1, 2, 3, 4, 5, 6, 7))
    engine = EventEngine(trace, [1000, 1000])
    task = make_task(bits=bits)
    engine.admit([task], {0: RoutingAction(0, 1, (0, 1))})
    engine.advance(2.0)
    assert engine.task_records()[0]["status"] == expected_status
    if expected_status == "route_failed":
        assert engine.jobs[0].terminal_time == pytest.approx(1.0)
        assert engine.cpu_cycles_processed == 0


def test_missed_deadline_reported_once_and_late_task_finishes(trace_factory):
    engine = EventEngine(trace_factory(), [100, 100])
    engine.admit([make_task(deadline=0.5)], {0: RoutingAction(0, 0, (0,))})
    engine.advance(0.6)
    assert engine.deadline_misses == 1
    engine.advance(2.0)
    row = engine.task_records()[0]
    assert row["status"] == "completed" and row["deadline_missed"] and not row["success"]
    assert engine.deadline_misses == 1


def test_deadline_drop_releases_capacity(trace_factory):
    engine = EventEngine(trace_factory(), [100, 100], drop_at_deadline=True)
    engine.admit([make_task(deadline=0.5)], {0: RoutingAction(0, 0, (0,))})
    engine.advance(2.0)
    row = engine.task_records()[0]
    assert row["status"] == "timed_out" and row["completion_delay_s"] is None
    assert row["sojourn_s"] == pytest.approx(0.5)
    assert engine.cpu_cycles_processed == pytest.approx(50)
    assert not engine.active


def test_batch_validation_is_atomic(trace_factory):
    engine = EventEngine(trace_factory(), [100, 100])
    tasks = [make_task(), make_task(1, slot=1)]
    with pytest.raises(ValueError):
        engine.admit(tasks, {0: RoutingAction(0, 0, (0,)), 1: RoutingAction(1, 0, (0,))})
    assert engine.jobs == {}


def test_small_high_rate_residual_at_large_absolute_time(trace_factory):
    trace = trace_factory(slots=300, capacity=1e9)
    engine = EventEngine(trace, [1e10, 1e10])
    engine.advance(200.0)
    task = make_task(bits=500123.456, complexity=500, deadline=1, slot=200)
    engine.admit([task], {0: RoutingAction(0, 1, (0, 1))})
    engine.advance(201.0)
    assert engine.jobs[0].stage == "completed"
    assert engine.transmitted_bits == pytest.approx(task.data_bits)
    assert engine.cpu_cycles_processed == pytest.approx(task.total_cycles)
