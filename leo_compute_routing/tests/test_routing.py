from copy import deepcopy
import networkx as nx
import pytest
from leo_routing.network.contact_plan import ContactPlan
from leo_routing.env.leo_env import LeoEnv
from leo_routing.routing.contact_aware_router import ContactAwareRouter
from leo_routing.routing.reservations import ReservationCalendar
from leo_routing.tasks.task import Task


def test_router_matches_exhaustive_cost_on_small_static_graph(trace_factory, tiny_config):
    trace = trace_factory(satellites=6, slots=24, edges=((0, 1), (1, 3), (0, 2), (2, 3), (1, 2)))
    config = deepcopy(tiny_config)
    config["routing"].update(reference_rate_fraction=1., lookahead_slots=5,
                             contact_risk_weight_seconds=0., link_load_weight_seconds=0.)
    obs, _ = LeoEnv(config, trace, ((), (), ()), [1000] * 6).reset()
    router, calendar = ContactAwareRouter(config["routing"]), ReservationCalendar.from_observation(obs)
    task = Task(0, 0, 100, 1, 20, 0)
    for target in (1, 2, 3):
        result = router.find_route(obs, task, target, calendar)
        expected = min((len(path) - 1, tuple(path)) for path in
                       nx.all_simple_paths(obs.graph, 0, target, cutoff=config["routing"]["max_path_hops"]))
        assert result.action.path == expected[1]
        assert result.cost_seconds == pytest.approx(expected[0])
        assert not result.search_truncated
    assert router.find_route(obs, task, 0, calendar).action.is_local
    absent = router.find_route(obs, task, 5, calendar)
    assert absent.action is None and absent.reason == "no_verified_route"
    fallback = router.resolve(obs, task, 5, calendar)
    assert fallback.action.is_local and fallback.action.requested_compute_sat == 5


def test_future_check_covers_transmission_not_only_arrival(trace_factory):
    trace = trace_factory(unavailable_slots=(1,))
    task = Task(0, 0, 150, 1, 10, 0)
    checked = ContactPlan.from_trace(trace, 0, 3).predict(task, (0, 1), 1.0)
    assert not checked.topology_feasible
    # First-hop arrival time is zero; checking only that snapshot would miss the loss.
    snapshot = ContactPlan.from_trace(trace, 0, 0).predict(task, (0, 1), 1.0)
    assert snapshot.topology_feasible and not snapshot.fully_checked


def test_future_interval_half_open_and_no_propagation_contact_needed(trace_factory):
    trace = trace_factory(propagation_seconds=0.5, unavailable_slots=(1,))
    checked = ContactPlan.from_trace(trace, 0, 3).predict(Task(0, 0, 100, 1, 10, 0), (0, 1), 1.0)
    assert checked.topology_feasible and checked.fully_checked
    assert checked.route_seconds == pytest.approx(1.5)


def test_prediction_beyond_cache_is_unknown(trace_factory):
    trace = trace_factory(slots=2)
    checked = ContactPlan.from_trace(trace, 0, 10).predict(Task(0, 0, 400, 1, 10, 0), (0, 1), 1.0)
    assert checked.topology_feasible and not checked.fully_checked


def test_shared_resources_can_invalidate_a_prediction(trace_factory):
    from leo_routing.env.event_engine import EventEngine
    from leo_routing.routing.action_builder import RoutingAction
    trace = trace_factory(unavailable_slots=(2, 3, 4, 5, 6, 7))
    tasks = [Task(i, 0, 150, 1, 10, 0) for i in (0, 1)]
    assert all(ContactPlan.from_trace(trace, 0, 4).predict(task, (0, 1), 1.0).topology_feasible for task in tasks)
    engine = EventEngine(trace, [1000, 1000])
    engine.admit(tasks, {i: RoutingAction(i, 1, (0, 1)) for i in (0, 1)})
    engine.advance(3.0)
    assert engine.route_failures == 2
