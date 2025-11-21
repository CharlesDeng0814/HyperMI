import torch
import torch.nn as nn 
import torch.nn.functional as F  
from torch.nn.parameter import Parameter #
from torch.nn.modules.module import Module 
from torch_scatter import scatter_add
from torch import Tensor
import math
class BNHCEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers, dropout, n_class):
        super(BNHCEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.fc = nn.Linear(in_dim,node_dim)
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(BNHCLayer(self.node_dim, self.node_dim))
        else:
            self.convs.append(BNHCLayer(self.node_dim, self.node_dim))
            for _ in range(self.num_layers - 2):
                self.convs.append(BNHCLayer(self.node_dim, self.node_dim))
            self.convs.append(BNHCLayer(self.node_dim, self.node_dim))
        self.outLayer = nn.Linear(node_dim, n_class)

    def forward(self, x: Tensor, hyperedge_index: Tensor, weightMatrix, edgesWeights):
        x = F.relu(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(self.num_layers):
            z = self.convs[i](x, hyperedge_index, weightMatrix, edgesWeights)
            x = F.relu(x + z)
            x = F.dropout(x, self.dropout, training=self.training) 
        return x
    
class BNHCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha_init=0.5):
        super().__init__()
        self.lin_n2e = nn.Linear(in_dim, out_dim)
        self.lin_e2n = nn.Linear(out_dim, out_dim)
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = Parameter(torch.Tensor(out_dim))
        self.register_buffer('node_degree', None)
        self.register_buffer('hyper_degree', None)
        self.register_buffer('norm_degree', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x: Tensor, hyperedge_index: Tensor, weightMatrix, edgesWeights):
        row, col = hyperedge_index
        if self.norm_degree is None:
            self._compute_degrees(hyperedge_index, edgesWeights, weightMatrix, x.size(0), len(weightMatrix))
        norm_degree = self.norm_degree
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        weighted_x = x[row] * edgesWeights.unsqueeze(-1) * norm_degree.unsqueeze(-1)
        hyper_feat = scatter_add(weighted_x, col, dim=0)
        message_from_hyperedge = hyper_feat[col]
        beta_weight = weightMatrix[col].unsqueeze(-1)
        w_ve_weight = edgesWeights.unsqueeze(-1)
        weighted_hyper = message_from_hyperedge * beta_weight * w_ve_weight * norm_degree.unsqueeze(-1)
        node_out = scatter_add(weighted_hyper, row, dim=0)
        return node_out
    
    def _compute_degrees(self, hyper_index, hyper_weights, beta_fixed, num_nodes, num_hyperedges):
        node_degree = scatter_add(
            beta_fixed[hyper_index[1]], 
            hyper_index[0], 
            dim=0, 
            dim_size=num_nodes
        ) + 1e-8
        hyper_degree = scatter_add(
            hyper_weights,
            hyper_index[1], 
            dim=0, 
            dim_size=num_hyperedges
        ) + 1e-8
        node_degree_inv_sqrt = node_degree.pow(-0.5)
        node_degree_inv_sqrt[node_degree_inv_sqrt == float('inf')] = 0
        hyper_degree_inv_sqrt = hyper_degree.pow(-0.5)
        hyper_degree_inv_sqrt[hyper_degree_inv_sqrt == float('inf')] = 0
        norm_degree = hyper_degree_inv_sqrt[hyper_index[1]] * node_degree_inv_sqrt[hyper_index[0]]
        self.register_buffer('norm_degree', norm_degree)
        
class dualChannelArchitecture(nn.Module):
    def __init__(self, featureDim, dropout, nhid=256, nclass=2):
        super(dualChannelArchitecture, self).__init__()
        self.fc = nn.Linear(featureDim * 2, nhid)
        self.cls = nn.Linear(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x_1, x_2):
        """每次运行时都会执行的步骤，所有自定义的module都要重写这个函数"""
        x = torch.cat([x_1,x_2],dim = 1) 
        x = F.relu(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        #print(x.shape)
        res = self.cls(x)
        return F.log_softmax(res, dim=1) # 计算 sotmax + log 输出
    
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.clf = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.clf(x)
        return F.log_softmax(x, dim=1)
