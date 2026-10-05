from itertools import islice

import networkx as nx


def bounded_k_shortest_paths(graph, source, destinations, k, reference_bits, max_hops, expansion_limit):
    """Exact weighted K shortest SIMPLE paths under a hop bound, per target.

    Enumerate a bounded DFS tree once per source rather than rerunning Yen
    for every target. For max degree 4 and hop bound 4 this tree is small.
    Raise on the expansion guard rather than silently returning biased paths.
    """
    targets = set(destinations)
    paths = {target: [] for target in targets}
    paths.setdefault(source, []).append((0.0, (source,)))
    stack, expansions = [((source,), 0.0)], 0
    while stack:
        path, cost = stack.pop()
        if len(path) - 1 >= max_hops:
            continue
        for neighbor in sorted(graph.neighbors(path[-1])):
            if neighbor in path:
                continue
            expansions += 1
            if expansions > expansion_limit:
                raise ValueError("Path expansion guard reached; reduce max_path_hops or increase path_expansion_limit")
            edge = graph.edges[path[-1], neighbor]
            updated = path + (neighbor,)
            total = cost + reference_bits / edge["capacity_bps"] + edge["propagation_seconds"]
            if neighbor in targets:
                paths[neighbor].append((total, updated))
            stack.append((updated, total))
    return {target: tuple(path for _, path in sorted(items)[:k]) for target, items in paths.items()}


def k_shortest_paths(graph, source, destination, k, reference_bits, max_hops, search_limit):
    """Yen-style simple paths ranked at a fixed reference task size.

    Enumerate at most search_limit paths, then retain at most k within the hop
    bound. Weighted ordering does not imply monotonic hop counts, so a long
    path does not terminate the search. Fewer than k candidates is legitimate.
    """
    if source == destination:
        return ((source,),)
    weight = lambda i, j, attributes: (reference_bits / attributes["capacity_bps"] +
                                       attributes["propagation_seconds"])
    try:
        paths = nx.shortest_simple_paths(graph, source, destination, weight=weight)
        selected = []
        for path in islice(paths, search_limit):
            if len(path) - 1 <= max_hops:
                selected.append(tuple(path))
                if len(selected) == k:
                    break
        return tuple(selected)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return ()
