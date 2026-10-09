"""Dense edge-aware multi-head GAT for the current small (6--72 node) graphs.

Self loops guarantee a valid neighborhood for isolated satellites. There is no
dropout: PPO rescoring must match sampling before the first optimizer update.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, heads, edge_dim):
        super().__init__()
        self.heads, self.channels = heads, hidden_dim // heads
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, self.channels))
        self.target_attention = nn.Parameter(torch.empty(heads, self.channels))
        self.edge_attention = nn.Linear(edge_dim, heads, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, nodes, edge_index, edges):
        count = nodes.shape[0]
        transformed = self.projection(nodes).reshape(count, self.heads, self.channels)
        source = (transformed * self.source_attention).sum(-1).T
        target = (transformed * self.target_attention).sum(-1).T
        edge_scores = nodes.new_zeros(self.heads, count, count)
        adjacency = torch.eye(count, dtype=torch.bool, device=nodes.device)
        if edge_index.shape[1]:
            src, dst = edge_index
            adjacency[dst, src] = True
            edge_scores[:, dst, src] = self.edge_attention(edges).T
        scores = F.leaky_relu(target[:, :, None] + source[:, None, :] + edge_scores, negative_slope=0.2)
        attention = torch.softmax(scores.masked_fill(~adjacency[None], -torch.inf), dim=-1)
        aggregated = torch.einsum("hij,jhd->ihd", attention, transformed).reshape(count, -1)
        return self.norm(nodes + F.elu(self.output(aggregated)))

    def forward_dense(self, nodes, adjacency, edges):
        """The same attention equations for a batch of independent graphs."""
        batch, count, _ = nodes.shape
        transformed = self.projection(nodes).reshape(batch, count, self.heads, self.channels)
        source = (transformed * self.source_attention).sum(-1).transpose(1, 2)
        target = (transformed * self.target_attention).sum(-1).transpose(1, 2)
        edge_scores = self.edge_attention(edges).permute(0, 3, 1, 2)
        scores = F.leaky_relu(target[:, :, :, None] + source[:, :, None, :] + edge_scores,
                             negative_slope=0.2)
        attention = torch.softmax(scores.masked_fill(~adjacency[:, None], -torch.inf), dim=-1)
        aggregated = torch.einsum("bhij,bjhd->bihd", attention, transformed).reshape(batch, count, -1)
        return self.norm(nodes + F.elu(self.output(aggregated)))


class GATEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, heads=4, layers=2):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([AttentionLayer(hidden_dim, heads, edge_dim) for _ in range(layers)])

    def forward(self, nodes, edge_index, edges):
        hidden = F.elu(self.input(nodes))
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edges)
        return hidden

    def forward_dense(self, nodes, adjacency, edges):
        hidden = F.elu(self.input(nodes))
        for layer in self.layers:
            hidden = layer.forward_dense(hidden, adjacency, edges)
        return hidden
