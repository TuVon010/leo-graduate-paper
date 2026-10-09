"""Pack immutable autoregressive observations once per PPO update.

Tasks remain conditioned on their sampled prefixes. Packing only changes tensor
execution; probabilities are still summed into ONE joint action per slot.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..agents.features import EDGE_DIM, TASK_DIM, DESTINATION_DIM


@dataclass
class RolloutTensors:
    nodes: object
    context: object
    adjacency: object
    edges: object
    tasks: object
    destinations: object
    masks: object
    sources: object
    actions: object
    offsets: np.ndarray

    @classmethod
    def pack(cls, batches, device, with_edges=True):
        if not batches:
            raise ValueError("Cannot pack an empty rollout")
        count = len(batches[0].graph.nodes)
        if any(len(b.graph.nodes) != count for b in batches):
            raise ValueError("A PPO rollout must use one satellite count")
        offsets = np.r_[0, np.cumsum([len(b.decisions) for b in batches])]
        decisions = [d for b in batches for d in b.decisions]
        if any(len(b.decisions) != len(b.actions) for b in batches):
            raise ValueError("Decision/action lengths differ")
        if any(not d.mask.any() for d in decisions):
            raise ValueError("All-masked decision must have an explicit local fallback")
        adjacency = edges = None
        if with_edges:
            adjacency = np.broadcast_to(np.eye(count, dtype=bool), (len(batches), count, count)).copy()
            edges = np.zeros((len(batches), count, count, EDGE_DIM), dtype=np.float32)
            for i, batch in enumerate(batches):
                src, dst = batch.graph.edge_index
                adjacency[i, dst, src] = True
                edges[i, dst, src] = batch.graph.edges

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(array, dtype=dtype, device=device)

        return cls(tensor(np.stack([b.graph.nodes for b in batches])),
                   tensor(np.stack([b.graph.context for b in batches])),
                   tensor(adjacency, torch.bool) if with_edges else None,
                   tensor(edges) if with_edges else None,
                   tensor(np.stack([d.task for d in decisions]) if decisions else np.empty((0, TASK_DIM))),
                   tensor(np.stack([d.destination_features for d in decisions]) if decisions else
                          np.empty((0, count, DESTINATION_DIM))),
                   tensor(np.stack([d.mask for d in decisions]) if decisions else
                          np.empty((0, count)), torch.bool),
                   tensor([d.source for d in decisions], torch.long),
                   tensor([a for b in batches for a in b.actions], torch.long), offsets)

    def select(self, indices):
        """Select ragged tasks without consulting GPU scalars or changing order."""
        indices = np.asarray(indices, dtype=np.int64)
        sizes = self.offsets[indices + 1] - self.offsets[indices]
        task_indices = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in indices])
        local_slots = np.repeat(np.arange(len(indices)), sizes)
        device = self.nodes.device
        return (torch.as_tensor(indices, device=device),
                torch.as_tensor(task_indices, device=device, dtype=torch.long),
                torch.as_tensor(local_slots, device=device, dtype=torch.long))
