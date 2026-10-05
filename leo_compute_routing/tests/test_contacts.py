from copy import deepcopy

import numpy as np
import pytest

from leo_routing.network.contact_plan import ContactPlan
from leo_routing.routing.action_builder import RoutingAction
from leo_routing.routing.candidate_builder import CandidateBuilder
from leo_routing.routing.reservations import ReservationCalendar
from leo_routing.tasks.task import Task
from leo_routing.topology.topology_cache import TopologyTrace


def modified(trace, changes):
    adjacency, capacities = trace.adjacency.copy(), trace.capacities.copy()
    for slots, i, j, capacity in changes:
        adjacency[slots, i, j] = adjacency[slots, j, i] = capacity > 0
        capacities[slots, i, j] = capacities[slots, j, i] = capacity
    return TopologyTrace(trace.positions.copy(), adjacency, trace.distances.copy(), capacities, trace.slot_seconds)


def test_contact_windows_are_contiguous_bounded_and_half_open(trace_factory):
    trace = trace_factory(unavailable_slots=(2, 3))
    plan = ContactPlan.from_trace(trace, 0, 4)
    assert [(w.start, w.end, w.capacity_bits, w.end_known) for w in plan.windows] == [
        (0, 2, 200, True), (4, 5, 100, False)]
    exact = plan.predict(Task(0, 0, 200, 1, 20, 0), (0, 1), 1.0)
    assert exact.topology_feasible and exact.fully_checked
    assert exact.route_seconds == pytest.approx(2)
    assert not plan.predict(Task(0, 0, 201, 1, 20, 0), (0, 1), 1.0).topology_feasible
    # There is a later contact, but this implementation must not wait for it.
    assert not plan.transmit(0, 1, 2.5, 10, 1.0).feasible


def test_contact_integrates_variable_rate_and_excludes_propagation(trace_factory):
    trace = modified(trace_factory(propagation_seconds=0.5, unavailable_slots=(2,)), [(slice(1, 2), 0, 1, 50)])
    plan = ContactPlan.from_trace(trace, 0, 3)
    result = plan.predict(Task(0, 0, 150, 1, 20, 0), (0, 1), 1.0)
    assert result.topology_feasible and result.fully_checked
    assert result.route_seconds == pytest.approx(2.5)
    assert result.capacity_margin_ratio == pytest.approx(0)


def test_unknown_tail_never_reads_beyond_horizon(trace_factory):
    trace = trace_factory()
    changed = modified(trace, [(slice(2, None), 0, 1, 0)])
    task = Task(0, 0, 350, 1, 20, 0)
    a = ContactPlan.from_trace(trace, 0, 1).predict(task, (0, 1), 1.0)
    b = ContactPlan.from_trace(changed, 0, 1).predict(task, (0, 1), 1.0)
    assert a == b and a.topology_feasible and not a.fully_checked
    calendar = ReservationCalendar(ContactPlan.from_trace(trace, 0, 1), [100, 100], 1.0, False)
    assert not calendar.estimate(task, RoutingAction(0, 1, (0, 1))).contact_ok


def test_contact_search_changes_path_with_task_size(trace_factory, tiny_config):
    trace = trace_factory(satellites=4, edges=((0, 1), (1, 3), (0, 2), (2, 3)))
    trace = modified(trace, [(slice(2, None), 1, 3, 0), (slice(None), 0, 2, 80), (slice(None), 2, 3, 80)])
    settings = deepcopy(tiny_config["routing"])
    settings.update(candidate_generation="contact", k_paths=1, max_path_hops=2,
                    reference_rate_fraction=1.0, lookahead_slots=5)
    builder = CandidateBuilder(trace, [1000] * 4, settings)
    small = builder.build(Task(0, 0, 10, 1, 20, 0), 0, [0] * 4, [0] * 4)
    large = builder.build(Task(1, 0, 150, 1, 20, 0), 0, [0] * 4, [0] * 4)
    assert next(c.action.path for c in small if c.action.compute_sat == 3) == (0, 1, 3)
    assert next(c.action.path for c in large if c.action.compute_sat == 3) == (0, 2, 3)
    settings["path_expansion_limit"] = 1
    items = CandidateBuilder(trace, [1000] * 4, settings).build(Task(2, 0, 150, 1, 20, 0), 0, [0]*4, [0]*4)
    assert items[0].search_truncated  # Budget exhaustion is not called optimality.


def test_bidirectional_calendar_catches_batch_overbooking(trace_factory):
    plan = ContactPlan.from_trace(trace_factory(unavailable_slots=(2, 3, 4, 5, 6, 7)), 0, 5)
    calendar = ReservationCalendar(plan, [1000, 1000], 1.0)
    first = Task(0, 0, 150, 1, 10, 0)
    second = Task(1, 1, 150, 1, 10, 0)
    a, b = RoutingAction(0, 1, (0, 1)), RoutingAction(1, 0, (1, 0))
    assert calendar.estimate(first, a).feasible and calendar.estimate(second, b).feasible
    calendar.commit(first, a)
    assert not calendar.estimate(second, b).contact_ok
    assert sum(b - a for a, b in calendar.links[(0, 1)]) == pytest.approx(1.5)
    assert calendar.links[(0, 1)][0][0] == 0 and calendar.links[(0, 1)][-1][1] == 1.5


def test_cpu_bookings_start_after_data_arrival_and_share_capacity(trace_factory):
    plan = ContactPlan.from_trace(trace_factory(), 0, 5)
    calendar = ReservationCalendar(plan, [100, 100], 1.0)
    remote = Task(0, 0, 100, 1, 10, 0)
    estimate = calendar.commit(remote, RoutingAction(0, 1, (0, 1)))
    assert estimate.cpu_interval == pytest.approx((1, 2))
    # An idle CPU before remote data arrives remains available to a local task.
    local = Task(1, 1, 50, 1, 10, 0)
    assert calendar.estimate(local, RoutingAction(1, 1, (1,))).cpu_interval == pytest.approx((0, 0.5))
    longer = Task(2, 1, 150, 1, 2, 0)
    assert not calendar.estimate(longer, RoutingAction(2, 1, (1,))).deadline_ok


def test_calendar_seeds_residual_work_without_cpu_double_count(tiny_config):
    from leo_routing.env.leo_env import LeoEnv
    config = deepcopy(tiny_config)
    config["routing"]["reference_rate_fraction"] = 1.0
    jobs = ((Task(0, 0, 1000, 1, 30, 0), Task(1, 1, 1000, 1, 30, 0)), (), ())
    env = LeoEnv(config, task_trace=jobs, cpu_capacities=[100]*6)
    env.reset()
    obs, *_ = env.step({0: 0, 1: 0})
    calendar = ReservationCalendar.from_observation(obs)
    task = Task(2, 0, 100, 1, 20, 1)
    estimate = calendar.estimate(task, RoutingAction(2, 0, (0,)))
    assert estimate.cpu_interval == pytest.approx((10, 11))


def test_known_failed_input_does_not_reserve_downstream_cpu(trace_factory, tiny_config):
    from leo_routing.env.leo_env import LeoEnv
    config = deepcopy(tiny_config)
    config["routing"].update(reference_rate_fraction=1.0, lookahead_slots=5)
    trace = trace_factory(satellites=6, slots=24, edges=((0, 1), (1, 2)), unavailable_slots=tuple(range(2, 24)))
    jobs = ((Task(0, 0, 150, 1, 10, 0),), (), ())
    env = LeoEnv(config, trace, jobs, [1000]*6)
    obs, _ = env.reset()
    remote = next(i for i, item in enumerate(obs.candidates[0]) if item.action.compute_sat == 2)
    obs, *_ = env.step({0: remote})
    calendar = ReservationCalendar.from_observation(obs)
    assert 2 not in calendar.cpu
    assert calendar.links[(0, 1)] and calendar.links[(1, 2)]


def test_active_cache_keeps_terminal_records_without_serving_them(trace_factory):
    from leo_routing.env.event_engine import EventEngine
    engine = EventEngine(trace_factory(), [100, 100])
    tasks = [Task(i, i, 10, 1, 10, 0) for i in (0, 1)]
    engine.admit(tasks, {i: RoutingAction(i, i, (i,)) for i in (0, 1)})
    engine.advance(1)
    assert engine.active == () and len(engine.jobs) == 2
    assert all(j.stage == "completed" for j in engine.jobs.values())
    assert len(engine.task_records()) == 2


@pytest.mark.parametrize("horizon", [0, 1, 5])
@pytest.mark.parametrize("bits", [50, 150, 350])
def test_cached_contact_search_prefix_matches_full_route(trace_factory, horizon, bits):
    trace = trace_factory(satellites=3, edges=((0, 1), (1, 2)), propagation_seconds=0.25,
                          unavailable_slots=(3,))
    plan = ContactPlan.from_trace(trace, 1, horizon)
    task = Task(0, 0, bits, 1, 20, 1)
    prefix = plan.predict(task, (0,), 0.5)
    for i, j in ((0, 1), (1, 2)):
        prefix = plan.append_hop(task, prefix, i, j, 0.5)
    full = plan.predict(task, (0, 1, 2), 0.5)
    assert prefix.topology_feasible == full.topology_feasible
    assert prefix.fully_checked == full.fully_checked
    assert prefix.route_seconds == pytest.approx(full.route_seconds)
    assert prefix.contact_margin_seconds == pytest.approx(full.contact_margin_seconds)
    assert prefix.capacity_margin_ratio == pytest.approx(full.capacity_margin_ratio)
