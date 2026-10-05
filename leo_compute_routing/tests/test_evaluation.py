from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from conftest import ROOT
from leo_routing.baselines import make_policy
from leo_routing.config import load_config, validate_config
from leo_routing.env.event_engine import EventEngine
from leo_routing.env.leo_env import LeoEnv
from leo_routing.evaluation.calibration import audit_scenario
from leo_routing.evaluation.statistics import METRICS, aggregate_seed_results, run_multiseed
from leo_routing.routing.action_builder import CandidateAction, RoutingAction
from leo_routing.tasks.task import Task


def test_batch_bookings_reduce_cpu_herding(tiny_config):
    trace = ((Task(0, 0, 100, 1, 3, 0), Task(1, 0, 100, 1, 3, 0)), (), ())
    config = deepcopy(tiny_config)
    config["topology"]["link_capacity_bps"] = 1e9
    env = LeoEnv(config, task_trace=trace, cpu_capacities=[100, 200, 1, 1, 1, 1])
    observation, _ = env.reset()
    def destinations(policy):
        actions = make_policy(policy).select(observation)
        return [observation.candidates[t.task_id][actions[t.task_id]].action.compute_sat for t in trace[0]]
    assert destinations("computing_aware") == [1, 1]
    assert destinations("batch_greedy") == [1, 0]
    assert env.engine.jobs == {}  # Virtual bookings cannot mutate runtime state.
    assert np.all(observation.cpu_queue_cycles == 0)


def test_batch_link_bookings_change_path():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 3), (0, 2), (2, 3)], capacity_bps=100)
    tasks = (Task(0, 0, 100, 1, 3, 0), Task(1, 0, 100, 1, 3, 0))
    candidates = {t.task_id: tuple(CandidateAction(RoutingAction(t.task_id, 3, path), delay, 0, 0.1,
                                                 2, 100, True, True, True)
                                  for path, delay in [((0, 1, 3), 0.1), ((0, 2, 3), 0.2)]) for t in tasks}
    obs = SimpleNamespace(cpu_capacities=np.full(4, 1e9), active_jobs=(), graph=graph,
                          tasks=tasks, candidates=candidates)
    assert make_policy("batch_greedy").select(obs) == {0: 0, 1: 1}


def test_batch_future_deadline_ablation_really_disables_deadline_filter(tiny_config):
    trace = ((Task(0, 0, 100, 1, 0.1, 0),), (), ())
    choices = []
    for mask in [True, False]:
        config = deepcopy(tiny_config)
        config["routing"]["deadline_mask"] = mask
        config["topology"]["link_capacity_bps"] = 1e9
        env = LeoEnv(config, task_trace=trace, cpu_capacities=[100, 200, 1, 1, 1, 1])
        obs, _ = env.reset()
        actions = make_policy("batch_greedy_future").select(obs)
        choices.append(obs.candidates[0][actions[0]].action.compute_sat)
    assert choices == [0, 1]


def test_measurement_clips_events_and_excludes_drain(trace_factory):
    engine = EventEngine(trace_factory(), [100, 100], measurement_window=(0.25, 0.75))
    task = Task(0, 0, 100, 1, 10, 0)
    engine.admit([task], {0: RoutingAction(0, 0, (0,))})
    engine.advance(3)
    assert engine.measured_seconds == pytest.approx(0.5)
    assert engine.measured_cpu_busy_integral == pytest.approx(50)
    assert engine.measured_cpu_total_integral == pytest.approx(100)
    assert engine.cpu_busy_capacity_integral / engine.cpu_total_capacity_integral == pytest.approx(1 / 6)


def test_warmup_excludes_tasks_but_keeps_their_competition(tiny_config):
    config = deepcopy(tiny_config)
    config["evaluation"]["warmup_slots"] = 1
    trace = ((Task(0, 0, 1000, 1, 20, 0),), (Task(1, 0, 100, 1, 20, 1),), ())
    env = LeoEnv(config, task_trace=trace, cpu_capacities=[100] * 6)
    obs, _ = env.reset()
    while True:
        obs, _, done, truncated, info = env.step(make_policy("local").select(obs))
        if done or truncated:
            break
    metrics = info["episode_metrics"]
    assert metrics["task_count"] == 1 and metrics["all_admitted_task_count"] == 2
    assert metrics["utilization_measurement_seconds"] == pytest.approx(2)
    assert metrics["cpu_utilization"] == pytest.approx(1 / 6)
    assert metrics["mean_completion_delay_s"] > 1  # Warmup job is still sharing this CPU.


def test_legacy_audit_and_intrinsic_deadline_bound(trace_factory):
    config = load_config(ROOT / "configs/legacy_overload.yaml")
    topology = trace_factory(satellites=24, slots=250)
    task = Task(0, 0, 5e6, 1500, 0.5, 0)
    report = audit_scenario(config, topology, ((task,),), np.full(24, 8e9))
    assert report["expected_hotspot_task_fraction"] == pytest.approx(0.7375)
    assert report["expected_local_rho_per_satellite"][0] == pytest.approx(4.05625)
    assert report["optimistic_intrinsic_deadline_impossible_count"] == 1
    base = load_config(ROOT / "configs/base.yaml")
    report = audit_scenario(base, topology, ((task,),), np.full(24, 2e10))
    assert max(report["expected_local_rho_per_satellite"]) == pytest.approx(0.8525)
    assert report["optimistic_intrinsic_deadline_impossible_count"] == 0


def test_paired_seed_bootstrap_preserves_pairing_and_undefined_metrics():
    rows = []
    for seed, value in [(42, 0.1), (43, 0.7), (44, 0.4)]:
        for algorithm, extra in [("reference", 0), ("other", 0.1)]:
            rows.append({"algorithm": algorithm, "seed": seed,
                         **{metric: value + extra for metric in METRICS}})
    summaries, paired = aggregate_seed_results(rows, "reference", bootstrap_samples=200)
    assert all(row["mean_difference"] == pytest.approx(0.1) for row in paired)
    assert all(row["ci95_low"] == pytest.approx(0.1) and row["ci95_high"] == pytest.approx(0.1) for row in paired)
    rows[0]["mean_completion_delay_s"] = None
    summaries, paired = aggregate_seed_results(rows, "reference", bootstrap_samples=200)
    row = next(r for r in summaries if r["algorithm"] == "reference" and r["metric"] == "mean_completion_delay_s")
    assert row["valid_seed_count"] == 2 and row["mean"] is None and row["ci95_low"] is None
    with pytest.raises(ValueError):
        aggregate_seed_results(rows + [rows[0]], "reference")


def test_multiseed_preserves_raw_files_and_refuses_overwrite(tiny_config, tmp_path):
    summaries, paired = run_multiseed(tiny_config, ["local", "batch_greedy"], [42, 43], tmp_path,
                                      bootstrap_samples=100)
    assert summaries and paired
    for seed in [42, 43]:
        assert (tmp_path / ("seed_%s" % seed) / "calibration.json").exists()
        assert (tmp_path / ("seed_%s" % seed) / "batch_greedy/tasks.csv").exists()
    with pytest.raises(ValueError):
        run_multiseed(tiny_config, ["batch_greedy"], [42, 43], tmp_path)


@pytest.mark.parametrize("warmup", [-1, 3, True])
def test_invalid_warmup_is_rejected(tiny_config, warmup):
    tiny_config["evaluation"]["warmup_slots"] = warmup
    with pytest.raises(ValueError):
        validate_config(tiny_config)
