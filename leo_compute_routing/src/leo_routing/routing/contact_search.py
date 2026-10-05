"""Task-sized bounded path search over the current graph and rolling contacts."""
import heapq


def contact_paths(graph, task, destinations, plan, settings, capacities, backlog):
    """Enumerate loop-free labels, retaining K ranked paths per destination.

    Known-broken labels rank after predicted contact-feasible labels. They remain
    available to unshielded ablations. No later-contact waiting edges are added.
    Search truncation is explicit rather than claiming exhaustive optimality.
    """
    targets = set(destinations)
    paths = {target: [] for target in targets}
    paths[task.source_sat] = [(0.0, (task.source_sat,))]
    frontier = [(0.0, (task.source_sat,))]
    predictions = {(task.source_sat,): plan.predict(task, (task.source_sat,), settings["reference_rate_fraction"])}
    expansion_count = 0
    while frontier and expansion_count < settings["path_expansion_limit"]:
        _, path = heapq.heappop(frontier)
        prediction = predictions[path]
        expansion_count += 1
        if len(path) > 1 and path[-1] in targets:
            target = path[-1]
            completion = prediction.route_seconds + (backlog[target] + task.total_cycles) / capacities[target]
            risk = not prediction.topology_feasible
            paths[target].append((risk, completion, path))
        if len(path) - 1 >= settings["max_path_hops"]:
            continue
        for neighbor in sorted(graph.neighbors(path[-1])):
            if neighbor in path:
                continue
            extended = path + (neighbor,)
            next_prediction = plan.append_hop(task, prediction, path[-1], neighbor, settings["reference_rate_fraction"])
            predictions[extended] = next_prediction
            risk_penalty = (plan.end - plan.start + task.deadline_seconds) if not next_prediction.topology_feasible else 0.0
            heapq.heappush(frontier, (next_prediction.route_seconds + risk_penalty, extended))
    result = {target: tuple(item[-1] for item in sorted(items)[:settings["k_paths"]])
              for target, items in paths.items()}
    return result, bool(frontier), predictions
