"""Route to ONE selected computing satellite; never construct PPO path actions.

Time-dependent, bounded loop-free label search. Contacts are checked at each
hop's predicted sending time, including committed link service. The best
completed label seen within the search budget is returned, not a global optimum.
"""
from dataclasses import dataclass, replace
import heapq
import math
import networkx as nx

from .action_builder import RoutingAction
from ..network.link_model import edge_key


@dataclass(frozen=True)
class RouteResult:
    action: object
    prediction: object
    cost_seconds: float
    search_truncated: bool
    reason: str
    estimate: object = None


class ContactAwareRouter:
    def __init__(self, settings):
        self.settings = settings
        self._observation = None
        self._target_distances = {}

    def _prepare(self, observation):
        if self._observation is observation:
            return
        self._observation = observation
        self._target_distances = {}
        self._neighbors = {s: tuple(sorted(observation.graph.neighbors(s))) for s in observation.graph}
        self._link_work = {}
        for job in observation.active_jobs:
            if job.stage == "tx":
                edge = edge_key(*job.path[job.hop:job.hop + 2])
                self._link_work[edge] = self._link_work.get(edge, 0) + job.remaining_bits

    def find_route(self, observation, task, destination, calendar):
        if destination == task.source_sat:
            action = RoutingAction(task.task_id, destination, (destination,))
            return RouteResult(action, calendar.plan.predict(task, action.path, calendar.rate_fraction), 0, False, "local")
        graph, plan = observation.graph, calendar.plan
        mode = self.settings.get("mode", "contact")
        self._prepare(observation)
        link_work = self._link_work
        hop_limit = self.settings["max_path_hops"]
        if destination not in self._target_distances:
            self._target_distances[destination] = nx.single_source_shortest_path_length(graph, destination, cutoff=hop_limit)
        distance_to_target = self._target_distances[destination]
        if task.source_sat not in distance_to_target:
            return RouteResult(None, None, math.inf, False, "no_verified_route")
        prefix = plan.predict(task, (task.source_sat,), calendar.rate_fraction)
        frontier = [(0.0, (task.source_sat,), prefix)]
        best, expanded = None, 0
        while frontier and expanded < self.settings["path_expansion_limit"]:
            # Every added hop has nonnegative delay/risk/load cost. Remaining
            # labels cannot improve the winner; strict comparison preserves ties.
            if best is not None and frontier[0][0] > best[0]:
                break
            score, path, prediction = heapq.heappop(frontier)
            expanded += 1
            if mode == "contact" and (not prediction.topology_feasible or
                    (plan.future_enabled and not prediction.fully_checked and
                     not self.settings["allow_unverified_future"])):
                continue
            if path[-1] == destination:
                item = (score, len(path), path, prediction)
                if best is None or item[:3] < best[:3]:
                    best = item
                continue
            if len(path) - 1 >= self.settings["max_path_hops"]:
                continue
            for neighbor in self._neighbors[path[-1]]:
                if neighbor in path:
                    continue
                extended = path + (neighbor,)
                if distance_to_target.get(neighbor, hop_limit + 1) > hop_limit - (len(extended) - 1):
                    continue  # A shortest-hop lower bound cannot discard a valid bounded route.
                route = plan.append_hop(task, prediction, path[-1], neighbor, calendar.rate_fraction, calendar.links)
                if mode == "snapshot":
                    edge = graph.edges[path[-1], neighbor]
                    cost = score + task.data_bits / edge["capacity_bps"] + edge["propagation_seconds"]
                else:
                    if not route.topology_feasible or (plan.future_enabled and not route.fully_checked and
                                                       not self.settings["allow_unverified_future"]):
                        continue
                    # Dimensionless preference penalties, converted to seconds by
                    # configured coefficients. They are not extra physical delay.
                    risk = sum(1 / (1 + max(0, hop.contact_margin_seconds)) for hop in route.hops) if plan.future_enabled else 0
                    load = sum(link_work.get(edge_key(i, j), 0) /
                               task.data_bits
                               for i, j in zip(extended, extended[1:]))
                    cost = (route.route_seconds + self.settings.get("contact_risk_weight_seconds", 0.1) * risk +
                            self.settings.get("link_load_weight_seconds", 0.1) * load)
                heapq.heappush(frontier, (cost, extended, route))
        truncated = bool(frontier) and (best is None or frontier[0][0] <= best[0])
        if best is None:
            return RouteResult(None, None, math.inf, truncated, "no_verified_route")
        return RouteResult(RoutingAction(task.task_id, destination, best[2]), best[3], best[0], truncated, "routed")

    def resolve(self, observation, task, destination, calendar):
        """Selected node -> route -> optional deadline admission check -> local fallback."""
        result = self.find_route(observation, task, destination, calendar)
        if result.action is not None:
            estimate = calendar.estimate(task, result.action, result.prediction)
            if (not observation.deadline_mask_enabled or estimate.deadline_ok or result.action.is_local):
                return replace(result, estimate=estimate)
            reason = "predicted_deadline"
        else:
            reason = result.reason
        local = RoutingAction(task.task_id, task.source_sat, (task.source_sat,), int(destination), reason)
        return RouteResult(local, calendar.plan.predict(task, local.path, calendar.rate_fraction),
                           0, result.search_truncated, reason)
