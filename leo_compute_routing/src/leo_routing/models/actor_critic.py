import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..agents.features import NODE_DIM, EDGE_DIM, CONTEXT_DIM, TASK_DIM, DESTINATION_DIM
from .destination_scorer import DestinationScorer
from .gat_encoder import GATEncoder
from .rollout_tensors import RolloutTensors


class ActorCritic(nn.Module):
    def __init__(self, settings):
        super().__init__()
        hidden = settings["hidden_dim"]
        self.graph_encoder = (GATEncoder(NODE_DIM, EDGE_DIM, hidden, settings["gat_heads"], settings["gat_layers"])
                              if settings["encoder"] == "gat" else
                              nn.Sequential(nn.Linear(NODE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()))
        self.is_gat = settings["encoder"] == "gat"
        self.context_encoder = nn.Sequential(nn.Linear(2 * hidden + CONTEXT_DIM, hidden), nn.Tanh())
        self.scorer = DestinationScorer(hidden, TASK_DIM, DESTINATION_DIM)
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
        nodes = (self.graph_encoder(inputs, self.tensor(state.edge_index, torch.long), self.tensor(state.edges))
                 if self.is_gat else self.graph_encoder(inputs))
        pooled = torch.cat((nodes.mean(0), nodes.max(0).values, self.tensor(state.context)))
        context = self.context_encoder(pooled)
        # Rollout/GAE always use original reward units.
        return nodes, context, self.value_network(context).squeeze(-1) * self.value_scale

    def logits(self, encoded, decision):
        nodes, context, _ = encoded
        return self.scorer(nodes, context, self.tensor(decision.task),
                           self.tensor(decision.destination_features), decision.source)

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
        return RolloutTensors.pack(batches, self.device, self.is_gat)

    def evaluate_minibatch(self, rollout, indices):
        slots, tasks, local_slots = rollout.select(indices)
        inputs = rollout.nodes[slots]
        nodes = (self.graph_encoder.forward_dense(inputs, rollout.adjacency[slots], rollout.edges[slots])
                 if self.is_gat else self.graph_encoder(inputs))
        pooled = torch.cat((nodes.mean(1), nodes.max(1).values, rollout.context[slots]), -1)
        context = self.context_encoder(pooled)
        values = self.value_network(context).squeeze(-1) * self.value_scale
        # Preserve differentiable zero log probabilities for task-free slots.
        joint_logs, joint_entropy = values * 0, values * 0
        if len(tasks):
            logits = self.scorer.forward_many(nodes, context, rollout.tasks[tasks],
                rollout.destinations[tasks], rollout.sources[tasks], local_slots)
            masks = rollout.masks[tasks]
            log_probs = torch.log_softmax(logits.masked_fill(~masks, -torch.inf), -1)
            chosen = log_probs.gather(1, rollout.actions[tasks, None]).squeeze(1)
            entropy = -(log_probs.exp() * log_probs.masked_fill(~masks, 0)).sum(-1)
            joint_logs = joint_logs.scatter_add(0, local_slots, chosen)
            joint_entropy = joint_entropy.scatter_add(0, local_slots, entropy)
        return joint_logs, joint_entropy, values
