import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..agents.features import NODE_DIM, EDGE_DIM, CONTEXT_DIM, TASK_DIM, CANDIDATE_DIM
from .candidate_scorer import CandidateScorer
from .gat_encoder import GATEncoder


class ActorCritic(nn.Module):
    def __init__(self, settings):
        super().__init__()
        hidden = settings["hidden_dim"]
        self.graph_encoder = (GATEncoder(NODE_DIM, EDGE_DIM, hidden, settings["gat_heads"], settings["gat_layers"])
                              if settings["encoder"] == "gat" else
                              nn.Sequential(nn.Linear(NODE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()))
        self.is_gat = settings["encoder"] == "gat"
        self.context_encoder = nn.Sequential(nn.Linear(2 * hidden + CONTEXT_DIM, hidden), nn.Tanh())
        self.scorer = CandidateScorer(hidden, TASK_DIM, CANDIDATE_DIM)
        self.value_network = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

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
        return nodes, context, self.value_network(context).squeeze(-1)

    def logits(self, encoded, decision):
        nodes, context, _ = encoded
        return self.scorer(nodes, context, self.tensor(decision.task), self.tensor(decision.candidates),
                             decision.source, self.tensor(decision.destinations, torch.long), self.tensor(decision.path_pool))

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
