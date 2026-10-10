from torch import nn
import torch


class DestinationScorer(nn.Module):
    """One shared scorer per physical satellite; no fixed-ID output layer."""
    def __init__(self, hidden, task_dim, destination_dim):
        super().__init__()
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden), nn.Tanh())
        self.destination_encoder = nn.Sequential(nn.Linear(destination_dim, hidden), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.score[-1].weight, gain=0.01)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, nodes, context, task, features, source):
        count = len(nodes)
        return self.score(torch.cat((nodes, nodes[source].expand(count, -1),
            context.expand(count, -1), self.task_encoder(task).expand(count, -1),
            self.destination_encoder(features)), -1)).squeeze(-1)

    def forward_many(self, nodes, context, tasks, features, sources, slots):
        count = nodes.shape[1]
        return self.score(torch.cat((nodes[slots], nodes[slots, sources][:, None].expand(-1, count, -1),
            context[slots, None].expand(-1, count, -1),
            self.task_encoder(tasks)[:, None].expand(-1, count, -1),
            self.destination_encoder(features)), -1)).squeeze(-1)


class TaskGatedDestinationScorer(DestinationScorer):
    """Task-conditioned graph correction to a preserved own-node MLP branch.

    No satellite IDs, task-specific graph recomputation, or handcrafted logits.
    The critic uses the own branch, keeping graph gradients actor-specific.
    """
    def __init__(self, hidden, task_dim, destination_dim, gate_bias=-3.0):
        super().__init__(hidden, task_dim, destination_dim)
        self.gate = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Sigmoid())
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, gate_bias)

    def _score_many(self, nodes, graphs, context, tasks, features, sources):
        count = nodes.shape[1]
        task = self.task_encoder(tasks)[:, None].expand(-1, count, -1)
        destination = self.destination_encoder(features)
        shared_context = context[:, None].expand(-1, count, -1)
        batch = torch.arange(len(nodes), device=nodes.device)
        source = nodes[batch, sources][:, None].expand(-1, count, -1)
        gate_inputs = torch.cat((nodes, source, shared_context, task, destination), -1)
        gates = self.gate(gate_inputs)
        self.last_gate_mean = gates.detach().mean()
        fused = nodes + gates * graphs
        fused_source = fused[batch, sources][:, None].expand(-1, count, -1)
        return self.score(torch.cat((fused, fused_source, shared_context, task, destination), -1)).squeeze(-1)

    def forward(self, nodes, context, task, features, source, graph_nodes):
        sources = torch.tensor([source], dtype=torch.long, device=nodes.device)
        return self._score_many(nodes[None], graph_nodes[None], context[None], task[None],
                                features[None], sources)[0]

    def forward_many(self, nodes, context, tasks, features, sources, slots, graph_nodes):
        return self._score_many(nodes[slots], graph_nodes[slots], context[slots], tasks, features, sources)
