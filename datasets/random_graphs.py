import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx


class ERGraphDataset(InMemoryDataset):
    
    def __init__(
        self,
        node_sizes,
        edge_prob,
        edge_weight,
        fully_connected=False,
        transform=None,
        graphs_per_size=100,
    ):
        self.node_sizes = node_sizes
        self.edge_prob = edge_prob
        self.edge_weight = edge_weight
        self.fully_connected = fully_connected
        self.graphs_per_size = graphs_per_size
        super(ERGraphDataset, self).__init__('.', transform=transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []
        for num_nodes in self.node_sizes:
            for _ in range(self.graphs_per_size):
                if self.fully_connected:
                    nx_graph = nx.complete_graph(num_nodes)
                    for i in range(num_nodes):
                        nx_graph.add_edge(i, i, edge_weight=0.5)
                    edge_weight_value = 0.5
                else:
                    nx_graph = nx.erdos_renyi_graph(num_nodes, self.edge_prob)
                    for i in range(num_nodes):
                        if not nx_graph.has_edge(i, i):
                            nx_graph.add_edge(i, i, edge_weight=1.0)
                    edge_weight_value = 1.0
                
                for u, v, edge_data in nx_graph.edges(data=True):
                    edge_data['edge_weight'] = edge_weight_value
                data = from_networkx(nx_graph)
                data.x = torch.ones(num_nodes, 1)
                data.edge_weight = data.edge_weight.float()
                data_list.append(data)
        return self.collate(data_list)
