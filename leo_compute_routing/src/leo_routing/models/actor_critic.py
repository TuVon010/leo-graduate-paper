import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..agents.features import EDGE_DIM, CONTEXT_DIM, TASK_DIM, feature_dimensions
from .destination_scorer import DestinationScorer, TaskGatedDestinationScorer
from .gat_encoder import GATEncoder
from .rollout_tensors import RolloutTensors


class ActorCritic(nn.Module):
    def __init__(self, settings):
        super().__init__()
        hidden = settings["hidden_dim"]
        node_dim, self.destination_dim = feature_dimensions(settings)
        self.is_gat = settings["encoder"] in ("gat", "gated_gat")
        self.is_gated = settings["encoder"] == "gated_gat"
        self.self_only = settings["graph_neighbors"] == "self"
        self.graph_encoder = (GATEncoder(node_dim, EDGE_DIM, hidden, settings["gat_heads"], settings["gat_layers"])
                              if self.is_gat else
                              nn.Sequential(nn.Linear(node_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()))
        if self.is_gated:
            self.self_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.Tanh(),
                                               nn.Linear(hidden, hidden), nn.Tanh())
        self.context_encoder = nn.Sequential(nn.Linear(2 * hidden + CONTEXT_DIM, hidden), nn.Tanh())
        self.scorer = (TaskGatedDestinationScorer(hidden, TASK_DIM, self.destination_dim, settings["graph_gate_bias"])
                       if self.is_gated else DestinationScorer(hidden, TASK_DIM, self.destination_dim))
        self.value_network = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value_scale = settings["value_scale"]
        if self.value_scale != 1.0:
            # Start near zero in physical reward units despite the scaled value head.
            nn.init.orthogonal_(self.value_network[-1].weight, gain=0.01)
            nn.init.zeros_(self.value_network[-1].bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def tensor(self, array, dtype=torch.float32):
        # Copy read-only NumPy inputs to avoid undefined writes through shared memory.
        return torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=self.device)

    def encode(self, state):
        inputs = self.tensor(state.nodes)
        edge_index = state.edge_index[:, :0] if self.self_only else state.edge_index
        edges = state.edges[:0] if self.self_only else state.edges
        graph_nodes = (self.graph_encoder(inputs, self.tensor(edge_index, torch.long), self.tensor(edges))
                 if self.is_gat else self.graph_encoder(inputs))
        nodes = self.self_encoder(inputs) if self.is_gated else graph_nodes
        pooled = torch.cat((nodes.mean(0), nodes.max(0).values, self.tensor(state.context)))
        context = self.context_encoder(pooled)
        # Rollout/GAE always use original reward units.
        result = (nodes, context, self.value_network(context).squeeze(-1) * self.value_scale)
        return result + (graph_nodes,) if self.is_gated else result

    def logits(self, encoded, decision):
        nodes, context = encoded[:2]
        arguments = (nodes, context, self.tensor(decision.task),
                     self.tensor(decision.destination_features), decision.source)
        return self.scorer(*arguments, encoded[3]) if self.is_gated else self.scorer(*arguments)

    def distribution(self, encoded, decision):
        logits = self.logits(encoded, decision)
        mask = self.tensor(decision.mask, torch.bool)
        if not bool(mask.any()):
            raise ValueError("All-masked decision must have an explicit local fallback")
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def evaluate_batch(self, batch):
        encoded = self.encode(batch.graph)
        log_prob, entropy = encoded[2] * 0, encoded[2] * 0
        for decision, action in zip(batch.decisions, batch.actions):
            distribution = self.distribution(encoded, decision)
            log_prob = log_prob + distribution.log_prob(torch.tensor(action, device=self.device))
            entropy = entropy + distribution.entropy()
        return log_prob, entropy, encoded[2]

    def prepare_rollout(self, batches):
        return RolloutTensors.pack(batches, self.device, self.is_gat, self.destination_dim)

    def evaluate_minibatch(self, rollout, indices):
        slots, tasks, local_slots = rollout.select(indices)
        inputs = rollout.nodes[slots]
        adjacency = rollout.adjacency[slots] if self.is_gat else None
        edges = rollout.edges[slots] if self.is_gat else None
        if self.self_only:
            adjacency = torch.eye(inputs.shape[1], dtype=torch.bool, device=self.device)[None].expand(len(slots), -1, -1)
            edges = torch.zeros_like(edges)
        graph_nodes = (self.graph_encoder.forward_dense(inputs, adjacency, edges)
                 if self.is_gat else self.graph_encoder(inputs))
        nodes = self.self_encoder(inputs) if self.is_gated else graph_nodes
        pooled = torch.cat((nodes.mean(1), nodes.max(1).values, rollout.context[slots]), -1)
        context = self.context_encoder(pooled)
        values = self.value_network(context).squeeze(-1) * self.value_scale
        # Preserve differentiable zero log probabilities for task-free slots.
        joint_logs, joint_entropy = values * 0, values * 0
        if len(tasks):
            arguments = (nodes, context, rollout.tasks[tasks],
                         rollout.destinations[tasks], rollout.sources[tasks], local_slots)
            logits = (self.scorer.forward_many(*arguments, graph_nodes) if self.is_gated
                      else self.scorer.forward_many(*arguments))
            masks = rollout.masks[tasks]
            log_probs = torch.log_softmax(logits.masked_fill(~masks, -torch.inf), -1)
            chosen = log_probs.gather(1, rollout.actions[tasks, None]).squeeze(1)
            entropy = -(log_probs.exp() * log_probs.masked_fill(~masks, 0)).sum(-1)
            joint_logs = joint_logs.scatter_add(0, local_slots, chosen)
            joint_entropy = joint_entropy.scatter_add(0, local_slots, entropy)
        return joint_logs, joint_entropy, values
