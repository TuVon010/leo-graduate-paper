import torch
from torch import nn


class CandidateScorer(nn.Module):
    def __init__(self, hidden_dim, task_dim, candidate_dim):
        super().__init__()
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden_dim), nn.Tanh(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.path_encoder = nn.Sequential(nn.Linear(candidate_dim + hidden_dim, hidden_dim), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(5 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        nn.init.orthogonal_(self.score[-1].weight, gain=0.01)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, nodes, context, task, candidate_features, source, destinations, path_pool):
        count = candidate_features.shape[0]
        task_hidden = self.task_encoder(task).expand(count, -1)
        paths = self.path_encoder(torch.cat((candidate_features, path_pool @ nodes), dim=-1))
        return self.score(torch.cat((task_hidden, nodes[source].expand(count, -1), nodes[destinations],
                                     paths, context.expand(count, -1)), dim=-1)).squeeze(-1)
