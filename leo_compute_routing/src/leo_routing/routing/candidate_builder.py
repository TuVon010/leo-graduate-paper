import networkx as nx

from ..network.feasibility import predict_route
from ..network.contact_plan import ContactPlan
from ..network.ksp import bounded_k_shortest_paths, k_shortest_paths
from .action_builder import CandidateAction, RoutingAction
from .contact_search import contact_paths


class CandidateBuilder:
    def __init__(self, trace, cpu_capacities, settings):
        self.trace, self.cpu_capacities, self.settings = trace, cpu_capacities, settings
        self._cache_slot, self._path_cache = None, {}
        self._contact_plan = None

    def contact_plan(self, slot):
        if self._contact_plan is None or abs(self._contact_plan.start - slot * self.trace.slot_seconds) > 1e-10:
            self._contact_plan = ContactPlan.from_trace(self.trace, slot, self.settings["lookahead_slots"])
        return self._contact_plan

    def build(self, task, slot, cpu_queue_cycles, in_transit_cycles):
        graph, settings = self.trace.graph(slot), self.settings
        if self._cache_slot != slot:
            self._cache_slot, self._path_cache = slot, {}
        reachable = nx.single_source_shortest_path_length(graph, task.source_sat,
                                                         cutoff=settings["max_compute_hops"])
        candidates = []
        # Local is always index zero, including when its deadline estimate fails.
        destinations = [task.source_sat] + sorted(s for s in reachable if s != task.source_sat)
        contact_mode = settings.get("candidate_generation", "ksp") == "contact"
        plan = self.contact_plan(slot)
        truncated = False
        if contact_mode:
            paths_by_destination, truncated, predictions = contact_paths(graph, task, destinations, plan, settings,
                self.cpu_capacities, cpu_queue_cycles + in_transit_cycles
                if hasattr(cpu_queue_cycles, "shape") else
                [a + b for a, b in zip(cpu_queue_cycles, in_transit_cycles)])
        elif settings["path_backend"] == "bounded" and (task.source_sat, task.source_sat) not in self._path_cache:
            source_paths = bounded_k_shortest_paths(graph, task.source_sat, destinations, settings["k_paths"],
                                                   settings["reference_data_bits"], settings["max_path_hops"],
                                                   settings["path_expansion_limit"])
            self._path_cache.update({(task.source_sat, destination): paths for destination, paths in source_paths.items()})
        for destination in destinations:
            key = (task.source_sat, destination)
            if not contact_mode and key not in self._path_cache:
                self._path_cache[key] = k_shortest_paths(
                    graph, task.source_sat, destination, settings["k_paths"],
                    settings["reference_data_bits"], settings["max_path_hops"], settings["path_search_limit"])
            for path in paths_by_destination[destination] if contact_mode else self._path_cache[key]:
                prediction = (predictions[path] if contact_mode else
                              predict_route(task, path, slot, self.trace, settings["reference_rate_fraction"],
                                            settings["lookahead_slots"]))
                contact_prediction = prediction if contact_mode else plan.predict(task, path, settings["reference_rate_fraction"])
                capacity = self.cpu_capacities[destination]
                # Include already committed in-flight work, without admitting it to CPU early.
                backlog = (cpu_queue_cycles[destination] + in_transit_cycles[destination]) / capacity
                execution = task.total_cycles / capacity
                margin = task.deadline_seconds - (prediction.route_seconds + backlog + execution)
                future_ok = (prediction.topology_feasible and
                             (prediction.fully_checked or settings["allow_unverified_future"] or
                              settings["lookahead_slots"] == 0))
                feasible = future_ok and (margin >= 0 or not settings["deadline_mask"])
                candidates.append(CandidateAction(RoutingAction(task.task_id, destination, path),
                                                   prediction.route_seconds, float(backlog), float(execution),
                                                   float(margin), prediction.bottleneck_bps,
                                                   prediction.topology_feasible, prediction.fully_checked, feasible,
                                                   contact_prediction.contact_margin_seconds,
                                                   contact_prediction.capacity_margin_ratio, truncated))
        return tuple(candidates)


def effective_mask(candidates):
    """Explicit local fallback prevents an all-masked categorical distribution."""
    mask = tuple(candidate.feasible for candidate in candidates)
    fallback = not any(mask)
    if fallback:
        mask = (True,) + (False,) * (len(candidates) - 1)
    return mask, fallback
