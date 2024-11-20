from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj


class IWN2(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int = 2,
            activation: Callable = torch.sigmoid,
        ) -> None:
        super(IWN2, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [IWN2Layer(in_channels, hidden_channels)] + \
            [IWN2Layer(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1)]
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        x_2d_A = self.to_dense_adj_and_signal(
            x, edge_index, edge_weight, batch)
        for i, layer in enumerate(self.layers):
            x_2d_A = layer(x_2d_A)
            if i < self.num_layers - 1:
                x_2d_A = self.activation(x_2d_A)
        return x_2d_A.mean([1, 2, 3])

    def to_dense_adj_and_signal(self,
                                x: torch.Tensor,
                                edge_index: torch.Tensor,
                                edge_weight: torch.Tensor,
                                batch: torch.Tensor) -> torch.Tensor:
        A = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)
        num_nodes_per_graph = A.size(-1)
        feature_dim = x.size(-1)
        batch_size = batch.max().item() + 1
        x_tensor = (
            x.view(batch.max().item()+1, -1, feature_dim)
            .transpose(1, 2)
            .unsqueeze(2)
            .expand(batch_size, feature_dim, num_nodes_per_graph,
                    num_nodes_per_graph)
        )
        return torch.cat([A.unsqueeze(1), x_tensor], dim=1)


class IWN2Layer(nn.Module):
    """
    Adapted from https://github.com/Haggaim/InvariantGraphNetworks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: bool = True,
        device: str = 'cpu',
    ) -> None:
        super(IWN2Layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.device = device
        self.basis_dim = 7

        self.coeffs = torch.nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, self.basis_dim),
            requires_grad=True,
        ).to(device = self.device)
        self.bias = torch.nn.Parameter(
            torch.zeros(1, self.out_channels, 1, 1)
        ).to(device = self.device)

    def forward(self, x_2d_A: torch.Tensor) -> torch.Tensor:

        ops_out = self.operators_2_to_2(x_2d_A)
        ops_out = torch.stack(ops_out, dim=2)

        output = torch.einsum('der,bdrij->beij', self.coeffs, ops_out)
        output = output + self.bias

        return output

    def operators_2_to_2(self, x_2d_A: torch.Tensor) -> torch.Tensor:
        """
        ij->ik
        ij->jk
        ij->ki
        ij->kj
        ij->ij
        ij->ji
        ij->kl
        """

        # x_2d_A shape: (b, d, n, n)
        n = x_2d_A.shape[-1]

        sum_of_rows = torch.sum(x_2d_A, dim=3)
        sum_of_cols = torch.sum(x_2d_A, dim=2)
        sum_all = torch.sum(sum_of_rows, dim=2)

        # op1 -- ij->ik (sum of cols on cols)
        op1 = torch.cat([sum_of_cols.unsqueeze(dim=3) for d in range(n)], dim=3)

        # op2 -- ij->jk (sum of rows on cols)
        op2 = torch.cat([sum_of_rows.unsqueeze(dim=3) for d in range(n)], dim=3)

        # op3 -- ij->ki (sum of cols on rows)
        op3 = torch.cat([sum_of_cols.unsqueeze(dim=2) for d in range(n)], dim=2)

        # op4 -- ij->kj (sum of rows on rows)
        op4 = torch.cat([sum_of_rows.unsqueeze(dim=2) for d in range(n)], dim=2)

        # op5 -- ij->ij (identity)
        op5 = x_2d_A

        # op6 -- ij->ji (transpose)
        op6 = x_2d_A.transpose(3, 2)

        # op7 -- ij->kl (total sum)
        op7 = torch.cat([sum_all.unsqueeze(dim=2) for d in range(n)], dim=2)
        op7 = torch.cat([op7.unsqueeze(dim=3) for d in range(n)], dim=3)
        
        if self.normalization:
            op1 = op1 / n
            op2 = op2 / n
            op3 = op3 / n
            op4 = op4 / n
            op7 = op7 / (n ** 2)

        return [op1, op2, op3, op4, op5, op6, op7]
