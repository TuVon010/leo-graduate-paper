import networkx as nx
import pytest

from leo_routing.network.feasibility import predict_route
from leo_routing.network.ksp import bounded_k_shortest_paths, k_shortest_paths
from leo_routing.routing.candidate_builder import CandidateBuilder, effective_mask
from leo_routing.tasks.task import Task


def test_ksp_endpoints_simple_and_sorted(trace_factory):
    trace = trace_factory(satellites=4, edges=((0, 1), (1, 3), (0, 2), (2, 3), (1, 2)))
    graph = trace.graph(0)
    paths = k_shortest_paths(graph, 0, 3, 3, 100, 3, 20)
    assert len(paths) == 3
    assert all(path[0] == 0 and path[-1] == 3 and len(set(path)) == len(path) for path in paths)
    costs = [sum(100 / graph.edges[i, j]["capacity_bps"] for i, j in zip(path, path[1:])) for path in paths]
    assert costs == sorted(costs)
    assert k_shortest_paths(graph, 0, 0, 3, 100, 3, 20) == ((0,),)


def test_ksp_disconnected(trace_factory):
    graph = trace_factory(satellites=3).graph(0)
    assert k_shortest_paths(graph, 0, 2, 3, 100, 3, 20) == ()


def test_future_check_covers_transmission_not_only_arrival(trace_factory):
    trace = trace_factory(unavailable_slots=(1,))
    task = Task(0, 0, 150, 1, 10, 0)
    checked = predict_route(task, (0, 1), 0, trace, 1.0, 3)
    assert not checked.topology_feasible
    # First-hop arrival time is zero; checking only that snapshot would miss the loss.
    snapshot = predict_route(task, (0, 1), 0, trace, 1.0, 0)
    assert snapshot.topology_feasible and not snapshot.fully_checked


def test_future_interval_half_open_and_no_propagation_contact_needed(trace_factory):
    trace = trace_factory(propagation_seconds=0.5, unavailable_slots=(1,))
    checked = predict_route(Task(0, 0, 100, 1, 10, 0), (0, 1), 0, trace, 1.0, 3)
    assert checked.topology_feasible and checked.fully_checked
    assert checked.route_seconds == pytest.approx(1.5)


def test_prediction_beyond_cache_is_unknown(trace_factory):
    trace = trace_factory(slots=2)
    checked = predict_route(Task(0, 0, 400, 1, 10, 0), (0, 1), 0, trace, 1.0, 10)
    assert checked.topology_feasible and not checked.fully_checked


def test_all_masked_has_explicit_local_fallback(trace_factory, tiny_config):
    trace = trace_factory(satellites=6)
    builder = CandidateBuilder(trace, [1.0] * 6, tiny_config["routing"])
    items = builder.build(Task(0, 0, 100, 100, 0.01, 0), 0, [0.0] * 6, [0.0] * 6)
    assert not any(candidate.feasible for candidate in items)
    mask, fallback = effective_mask(items)
    assert mask[0] and sum(mask) == 1 and fallback
    assert not items[0].feasible  # Fallback permission is not a feasibility certificate.


def test_bounded_paths_match_exhaustive_networkx_costs(trace_factory):
    trace = trace_factory(satellites=4, edges=((0, 1), (1, 3), (0, 2), (2, 3), (1, 2)))
    graph = trace.graph(0)
    paths = bounded_k_shortest_paths(graph, 0, [0, 1, 2, 3], 3, 100, 3, 100)
    for target in (1, 2, 3):
        expected = sorted((len(path) - 1, tuple(path)) for path in nx.all_simple_paths(graph, 0, target, cutoff=3))
        assert paths[target] == tuple(path for _, path in expected[:3])
    with pytest.raises(ValueError):
        bounded_k_shortest_paths(graph, 0, [3], 3, 100, 3, 1)


def test_shared_resources_can_invalidate_a_prediction(trace_factory):
    from leo_routing.env.event_engine import EventEngine
    from leo_routing.routing.action_builder import RoutingAction
    trace = trace_factory(unavailable_slots=(2, 3, 4, 5, 6, 7))
    tasks = [Task(i, 0, 150, 1, 10, 0) for i in (0, 1)]
    assert all(predict_route(task, (0, 1), 0, trace, 1.0, 4).topology_feasible for task in tasks)
    engine = EventEngine(trace, [1000, 1000])
    engine.admit(tasks, {i: RoutingAction(i, 1, (0, 1)) for i in (0, 1)})
    engine.advance(3.0)
    assert engine.route_failures == 2
