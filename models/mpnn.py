import torch
from torch_geometric.nn import MessagePassing, global_mean_pool


class MPNN(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers=2,
            activation=torch.sigmoid,
        ):
        super(MPNN, self).__init__()
        self.activation = activation
        self.layers = (
            [MPNNLayer(in_channels, hidden_channels)] + \
            [MPNNLayer(hidden_channels, hidden_channels)
             for _ in range(num_layers-2)] + \
            [MPNNLayer(hidden_channels, 1)]
        )

    def forward(self, x, edge_index, edge_weight, batch):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight, batch)
            x = self.activation(x)
        x = self.layers[-1](x, edge_index, edge_weight, batch)
        x = global_mean_pool(x, batch)
        return x
    

class MPNNLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(in_channels, out_channels)
        # TODO: Maybe change initialization!
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, edge_weight, batch):
        num_nodes_per_graph = torch.bincount(batch).float()
        num_nodes_for_each_node = num_nodes_per_graph[batch]
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out / num_nodes_for_each_node[:, None]
        return self.linear(out)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out
