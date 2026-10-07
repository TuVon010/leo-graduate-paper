"""Node selection baselines, all using the same physical service engine."""
from copy import deepcopy
import networkx as nx
import numpy as np

from ..routing.action_builder import RoutingAction
from ..routing.contact_aware_router import ContactAwareRouter
from ..routing.reservations import ReservationCalendar


class NodeHeuristic:
    def __init__(self, name, criterion="completion", use_future=False, booking=False, config=None):
        self.name, self.criterion = name, criterion
        self.use_future, self.booking, self.config = use_future, booking, config

    def configure_environment(self, config):
        configured = deepcopy(config)
        if self.name == "node_greedy":
            configured["routing"]["mode"] = (self.config or configured)["routing"]["mode"]
        else:
            configured["routing"]["mode"] = "contact" if self.use_future else "snapshot"
        if not self.use_future:
            configured["routing"].update(lookahead_slots=0, deadline_mask=False)
        return configured

    def select(self, observation):
        actions = {}
        calendar = ReservationCalendar.from_observation(observation)
        router = ContactAwareRouter(observation.routing_settings)
        prefix = np.zeros_like(observation.cpu_capacities)
        for task in sorted(observation.tasks, key=lambda t: (t.deadline_seconds, t.task_id)):
            reachable = nx.single_source_shortest_path_length(observation.graph, task.source_sat,
                                                             cutoff=observation.max_compute_hops)
            choices = []
            for destination in sorted(reachable):
                if self.criterion == "network" and destination == task.source_sat and len(reachable) > 1:
                    continue
                result = router.find_route(observation, task, destination, calendar)
                if result.action is None:
                    continue
                estimate = calendar.estimate(task, result.action)
                if self.use_future and observation.deadline_mask_enabled and not estimate.deadline_ok:
                    continue
                load = ((observation.cpu_queue_cycles[destination] + observation.inflight_cycles[destination] +
                         prefix[destination]) / observation.cpu_capacities[destination])
                score = (result.cost_seconds if self.criterion == "network" else
                         load if self.criterion == "load" else estimate.completion_seconds)
                choices.append((score, result.cost_seconds, destination, result.action))
            action = (min(choices, key=lambda item: item[:3])[-1] if choices else
                      RoutingAction(task.task_id, task.source_sat, (task.source_sat,)))
            actions[task.task_id] = action
            if self.booking:
                prefix[action.compute_sat] += task.total_cycles
                calendar.commit(task, action)
        return actions
