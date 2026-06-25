import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add


class NWHCLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm_mode: str = "wm_ew") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.norm_mode = norm_mode
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    @staticmethod
    def _to_1d_float(x, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32).view(-1)
        return torch.tensor(np.asarray(x), device=device, dtype=torch.float32).view(-1)
    
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        weight_matrix,
        edges_weights,
    ) -> Tensor:
        device = x.device
        row, col = hyperedge_index[0].long(), hyperedge_index[1].long()
        num_messages = row.numel()
        num_edges = int(col.max().item()) + 1
        num_nodes = x.size(0)
    
        wm_e = self._to_1d_float(weight_matrix, device)
        ew_m = self._to_1d_float(edges_weights, device)
    
        if wm_e.numel() != num_edges:
            raise RuntimeError(
                f"`weight_matrix` must have length {num_edges}, got {wm_e.numel()}."
            )
        if ew_m.numel() != num_messages:
            raise RuntimeError(
                f"`edges_weights` must have length {num_messages}, got {ew_m.numel()}."
            )
    
        x = x.matmul(self.weight) + self.bias
    
        if self.norm_mode == "orig":
            node_degree = scatter_add(wm_e[col], row, dim=0, dim_size=num_nodes).clamp_min(1e-8)
            hyper_degree = scatter_add(ew_m, col, dim=0, dim_size=num_edges).clamp_min(1e-8)
        elif self.norm_mode == "wm_only":
            node_degree = scatter_add(wm_e[col], row, dim=0, dim_size=num_nodes).clamp_min(1e-8)
            hyper_degree = scatter_add(
                torch.ones_like(col, dtype=torch.float32, device=device),
                col,
                dim=0,
                dim_size=num_edges,
            ).clamp_min(1e-8)
        elif self.norm_mode == "ones":
            node_degree = scatter_add(
                torch.ones_like(row, dtype=torch.float32, device=device),
                row,
                dim=0,
                dim_size=num_nodes,
            ).clamp_min(1e-8)
            hyper_degree = scatter_add(
                torch.ones_like(col, dtype=torch.float32, device=device),
                col,
                dim=0,
                dim_size=num_edges,
            ).clamp_min(1e-8)
        elif self.norm_mode == "wm_ew":
            node_degree = scatter_add((wm_e[col] * ew_m), row, dim=0, dim_size=num_nodes).clamp_min(1e-8)
            hyper_degree = scatter_add(ew_m, col, dim=0, dim_size=num_edges).clamp_min(1e-8)
        else:
            raise ValueError(f"Unknown norm_mode={self.norm_mode}")
    
        norm = node_degree.pow(-0.5)[row] * hyper_degree.pow(-0.5)[col]
    
        node_to_edge = x[row] * ew_m.unsqueeze(-1) * norm.unsqueeze(-1)
        hyper_feat = scatter_add(node_to_edge, col, dim=0, dim_size=num_edges)
    
        edge_to_node = (
            hyper_feat[col]
            * wm_e[col].unsqueeze(-1)
            * ew_m.unsqueeze(-1)
            * norm.unsqueeze(-1)
        )
        out = scatter_add(edge_to_node, row, dim=0, dim_size=num_nodes)
        return out


class NWHCEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        node_dim: int,
        num_layers: int,
        dropout: float,
        n_class: int,
        norm_mode: str = "wm_ew",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.fc = nn.Linear(in_dim, node_dim)
        self.convs = nn.ModuleList(
            [NWHCLayer(node_dim, node_dim, norm_mode=norm_mode) for _ in range(num_layers)]
        )
        self.outLayer = nn.Linear(node_dim, n_class)
    
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        weight_matrix,
        edges_weights,
    ) -> Tensor:
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        for conv in self.convs:
            z = conv(x, hyperedge_index, weight_matrix, edges_weights)
            x = F.relu(x + z)
            x = F.dropout(x, p=self.dropout, training=self.training)
    
        return x


class dualChannelArchitecture(nn.Module):
    def __init__(
        self,
        featureDim: int,
        dropout: float,
        nhid: int = 256,
        nclass: int = 2,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(featureDim * 2, nhid)
        self.cls = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        x = torch.cat([x_1, x_2], dim=1)
        x = F.relu(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        res = self.cls(x)
        return F.log_softmax(res, dim=1)


def xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Classifier_1(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2) -> None:
        super().__init__()
        self.clf = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.clf(x)
        return F.log_softmax(x, dim=1)

