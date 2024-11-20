import math
import os.path as osp
from typing import Callable, List, Optional, Dict, Tuple, Literal

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm


class GraphonDataset(InMemoryDataset):
    
    def __init__(
        self,
        node_sizes: List[int],
        graphs_per_size: int,
        signal: Literal['constant', 'random'] = 'constant',
        graph_type: Literal['sbm', 'smooth',
                            'smooth_narrow', 'triangular'] = 'sbm',
        params: Optional[Dict[str, float]] = None,
        weighted: bool = True,
        name: Optional[str] = None,
        transform: Optional[Callable] = None,
        root='datasets',
    ) -> None:
        self.node_sizes = node_sizes
        self.graphs_per_size = graphs_per_size
        self.signal = signal
        self.graph_type = graph_type
        self.params = params or {}
        self.weighted = weighted
        weighted_type = 'weighted' if weighted else 'simple'
        if name is None: name = self.graph_type
        self.folder = osp.join(root, f'{name}_{weighted_type}')

        super().__init__(self.folder, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self) -> Tuple[List[Data], Optional[Dict]]:
        print(f'Generating {self.graph_type}...')
        data_list: List[Data] = []
        for num_nodes in self.node_sizes:
            for _ in tqdm(range(self.graphs_per_size)):
                if self.graph_type == 'sbm':
                    data = self.generate_sbm(num_nodes)
                elif self.graph_type == 'smooth':
                    data = self.generate_smooth(num_nodes)
                elif self.graph_type == 'smooth_narrow':
                    data = self.generate_smooth_narrow(num_nodes)
                elif self.graph_type == 'triangular':
                    data = self.generate_triangular(num_nodes)
                else:
                    raise ValueError(
                        f"Unsupported graph type: {self.graph_type}")
                
                data.x = initialize_signal(num_nodes, self.signal)
                data_list.append(data)
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def graphon_to_data(self, graphon_matrix: np.ndarray) -> Data:
        graphon_matrix = graphon_matrix.astype(np.float32)
        if not self.weighted:
            lower_tri = np.tril(
                np.random.uniform(size=graphon_matrix.shape) < graphon_matrix
            ).astype(np.float32)
            graphon_matrix = (
                lower_tri + lower_tri.T - np.diag(np.diag(lower_tri)))
        nx_graph = nx.from_numpy_array(graphon_matrix)
        return from_networkx(nx_graph)

    def generate_sbm(self, num_nodes: int) -> Data:
        num_blocks = self.params.get('num_blocks', 3)
        intra_prob = self.params.get('intra_prob', 0.7)
        inter_prob = self.params.get('inter_prob', 0.3)
        sizes = [num_nodes // num_blocks] * num_blocks
        graphon_matrix = np.zeros((num_nodes, num_nodes))
        block_indices = np.cumsum([0] + sizes)
        for i in range(num_blocks):
            for j in range(num_blocks):
                prob = intra_prob if i == j else inter_prob
                block = slice(block_indices[i], block_indices[i + 1])
                block_other = slice(block_indices[j], block_indices[j + 1])
                graphon_matrix[block, block_other] = prob
        return self.graphon_to_data(graphon_matrix)

    def generate_smooth(self, num_nodes: int) -> Data:
        graphon_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                graphon_matrix[i, j] = abs(
                    math.sin(math.pi * i / num_nodes) * \
                    math.sin(math.pi * j / num_nodes))
        return self.graphon_to_data(graphon_matrix)
    
    def generate_smooth_narrow(self, num_nodes: int) -> Data:
        graphon_matrix = np.zeros((num_nodes, num_nodes))
        width = self.params.get('width', 0.05)
        for i in range(num_nodes):
            for j in range(num_nodes):
                x, y = i / num_nodes, j / num_nodes
                graphon_matrix[i, j] = math.exp(
                    -(math.sin((x - y) ** 2 / width) ** 2))
        return self.graphon_to_data(graphon_matrix)

    def generate_triangular(self, num_nodes: int) -> Data:
        graphon_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                graphon_matrix[i, j] = (i + j) / (2 * num_nodes)
        return self.graphon_to_data(graphon_matrix)
    
    def save_dataset(self, save_path: str) -> None:
        torch.save((self.data, self.slices), save_path)
        print(f"Dataset saved to {save_path}")

    @classmethod
    def load_dataset(cls, load_path: str, **kwargs) -> 'GraphonDataset':
        data, slices = torch.load(load_path)
        dataset = cls(**kwargs)
        dataset.data, dataset.slices = data, slices
        print(f"Dataset loaded from {load_path}")
        return dataset


def initialize_signal(
    num_nodes: int,
    which: Literal['constant', 'random'],
) -> torch.Tensor:
    if which == 'constant':
        return torch.ones(num_nodes, 1)
    elif which == 'random':
        return torch.randn(num_nodes, 1)
    else:
        raise ValueError(f'Unsupported signal type: {which}.')
