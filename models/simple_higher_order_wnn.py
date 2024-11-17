import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj


class SimpleGNN3(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers=2,
            activation=torch.sigmoid,
        ):
        super(SimpleGNN3, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [SimpleGNN3Layer(in_channels, hidden_channels)] + \
            [SimpleGNN3Layer(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1)]
        )

    def forward(self, x, edge_index, edge_weight, batch):
        x_2d = self.to_2d_tensor(x, batch)
        A = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)
        for i, layer in enumerate(self.layers):
            x_2d = layer(x_2d, A)
            if i < self.num_layers - 1:
                x_2d = self.activation(x_2d)
        return x_2d.mean([1, 2, 3])

    def to_2d_tensor(self, x, batch):
        feature_dim = x.shape[-1]
        b = batch.max().item() + 1
        x_2d = x.view(b, -1, feature_dim).transpose(1, 2)
        n = x_2d.shape[-1]
        return torch.cat([
            x_2d[:, :, :, None].expand(b, feature_dim, n, n),
            x_2d[:, :, None, :].expand(b, feature_dim, n, n),
        ], dim=1)


class SimpleGNN3Layer(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            normalization=True,
            device='cpu',
        ):
        super(SimpleGNN3Layer, self).__init__()

        self.in_channels = in_channels * 2  # each k (from features)
        self.out_channels = out_channels * 2
        self.normalization = normalization
        self.device = device
        self.basis_dim = 20

        self.coeffs = torch.nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, self.basis_dim) * \
            np.sqrt(2.0) / (self.in_channels + self.out_channels),
            requires_grad=True,
        ).to(device = self.device)
        self.bias = torch.nn.Parameter(
            torch.zeros(1, self.out_channels, 1, 1)
        ).to(device = self.device)

    def forward(self, x_2d, A):

        ops_out = self.operators_3_to_2(x_2d, A)
        ops_out = torch.stack(ops_out, dim=2)

        output = torch.einsum('der,bdrij->beij', self.coeffs, ops_out)
        output = output + self.bias

        return output

    def operators_3_to_2(self, x_2d, A):
        """
        (*) -- adjoint also contained

        ij,{ij},{ik}->jk  (*)
        ij,{ij},{jk}->ik  (*)
        ij,{ik},{jk}->ij
        ij,{ij}->ik       (*)
        ij,{ij}->jk       (*)
        ij,{ik}->jk       (*)
        ij,{jk}->ik       (*)
        ij,{ik}->ij
        ij,{jk}->ij
        ij->ik            (*)
        ij->jk            (*)
        ij->ij
        """

        # x_2d shape: (b, d, n, n)
        # A shape:   (n, n)
        xdim = x_2d.shape
        n = x_2d.shape[-1]
        norm = n if self.normalization else 1

        # op1 -- ij,{ij},{ik}->jk (*)
        op1 = torch.einsum('bdij,bij,bik->bdjk', x_2d, A, A) / norm
        ops1 = [op1, op1.transpose(2, 3)]

        # op2 -- ij,{ij},{jk}->ik (*)
        op2 = torch.einsum('bdij,bij,bjk->bdik', x_2d, A, A) / norm
        ops2 = [op2, op2.transpose(2, 3)]

        # op3 -- ij,{ik},{jk}->ij
        ops3 = [torch.einsum('bdij,bik,bjk->bdij', x_2d, A, A) / norm]

        # op4 -- ij,{ij}->ik (*)
        op4 = torch.einsum('bdij,bij->bdi', x_2d, A)[..., None].expand(xdim) / norm
        ops4 = [op4, op4.transpose(2, 3)]

        # op5 -- ij,{ij}->jk (*)
        op5 = torch.einsum('bdij,bij->bdj', x_2d, A)[..., None].expand(xdim) / norm
        ops5 = [op5, op5.transpose(2, 3)]

        # op6 -- ij,{ik}->jk (*)
        op6 = torch.einsum('bdij,bik->bdjk', x_2d, A) / norm
        ops6 = [op6, op6.transpose(2, 3)]

        # op7 -- ij,{jk}->ik (*)
        op7 = torch.einsum('bdij,bjk->bdik', x_2d, A) / norm
        ops7 = [op6, op6.transpose(2, 3)]

        # op8 -- ij,{ik}->ij
        ops8 = [torch.einsum('bdij,bik->bdij', x_2d, A) / norm]

        # op9 -- ij,{jk}->ij
        ops9 = [torch.einsum('bdij,bjk->bdij', x_2d, A) / norm]

        # op10 -- ij->ik (*)
        op10 = x_2d.sum(3, keepdim=True).expand(xdim) / norm
        ops10 = [op10, op10.transpose(2, 3)]

        # op11 -- ij->jk (*)
        op11 = x_2d.sum(2, keepdim=True).expand(xdim) / norm
        ops11 = [op11, op11.transpose(2, 3)]

        # op12 -- ij->ij
        ops12 = [x_2d]

        all_ops = [ops1, ops2, ops3, ops4, ops5, ops6,
                   ops7, ops8, ops9, ops10, ops11, ops12]

        return sum(all_ops, [])
