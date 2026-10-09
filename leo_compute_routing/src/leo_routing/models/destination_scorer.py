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
