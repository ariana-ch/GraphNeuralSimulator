import torch
from torch import nn

from config import _C as C


class MessagePassing(nn.Module):
    def __init__(self):
        super(MessagePassing, self).__init__()
        self.hidden_size = C.NET.HIDDEN_SIZE
        self.leakyReLU = nn.LeakyReLU(0.2)

        self.edge_model = nn.Sequential(
            nn.Linear((self.hidden_size + 2 * self.hidden_size), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.node_model = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

    def forward(self, nodes, edges, senders, receivers):
        src_nodes = nodes[senders]
        dst_nodes = nodes[receivers]
        edges = self.edge_model(torch.cat([dst_nodes, src_nodes, edges], 1))

        effects = torch.zeros_like(nodes).scatter_add_(0, receivers[:, None].expand(edges.shape), edges)
        nodes = torch.cat([nodes, effects], 1)
        nodes = self.node_model(nodes)

        return nodes, edges


class GraphNet(nn.Module):
    def __init__(self, layers=1):
        super(GraphNet, self).__init__()
        self.gn_list = nn.ModuleList([MessagePassing() for i in range(layers)])

    def forward(self, nodes, edges, senders, receivers):
        for i, l in enumerate(self.gn_list):
            nodes_out, edges_out = l(nodes, edges, senders, receivers)
            nodes = nodes + nodes_out
            edges = edges + edges_out

        return nodes, edges