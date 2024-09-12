import sys, argparse, os, copy, itertools, glob, datetime, math

import warnings

import torch_geometric
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

from torch_geometric.nn import DenseSAGEConv, GCNConv, GINConv, SAGEConv, dense_diff_pool,  Sequential,GENConv, DeepGCNLayer, GraphConv,GINConv, TopKPooling, MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch, degree

from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp




class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, cluster=False, conv=None):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)

        self.fc_k, self.fc_v = self.get_fc_kv(dim_K, dim_V, conv)
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.softmax_dim = 2
        if cluster == True:
            self.softmax_dim = 1

    def forward(self, Q, K, attention_mask=None, graph=None, return_attn=False):
        Q = self.fc_q(Q)

        # Adj: Exist (graph is not None), or Identity (else)
        if graph is not None:

            (x, edge_index, batch) = graph

            K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)

            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)

        else:

            K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
            A = torch.softmax(attention_mask + attention_score, self.softmax_dim)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), self.softmax_dim)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if return_attn:
            return O, A, attention_score
        else:
            return O
        
    def get_fc_kv(self, dim_K, dim_V, conv):

        if conv is None:
            fc_k = nn.Linear(dim_K, dim_V)
            fc_v = nn.Linear(dim_K, dim_V)


        else:

            fc_k = conv(dim_K, dim_V)
            fc_v = conv(dim_K, dim_V)

        return fc_k, fc_v


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, cluster=False, mab_conv=None):
        super(SAB, self).__init__()
        
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None):
        return self.mab(X, X, attention_mask, graph)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, cluster=False, mab_conv=None):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv=mab_conv)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, cluster=cluster, conv=mab_conv)

    def forward(self, X, attention_mask=None, graph=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attention_mask, graph)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, cluster=False, mab_conv=None):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        if mab_conv == None:
            self.mab = MAB(dim, dim, dim, num_heads, ln=ln, cluster=cluster, conv=None)
        else:
            self.mab = MAB(dim, dim, dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv)
        
    def forward(self, X, attention_mask=None, graph=None, return_attn=False):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask=attention_mask, graph=graph, return_attn=return_attn)


# https://github.com/JinheonBaek/GMT
# https://arxiv.org/abs/2102.11533
class graph_set_transformer(torch.nn.Module):
    def __init__(self,L, hidden_dim, dropout=0.0,Conv=GCNConv,layer_norm=False, nheads=4, mab_fc=False):
        super(graph_set_transformer,self).__init__()
        self.dropout_ratio = dropout
        self.mab_fc = mab_fc
        self.fc1 = Conv(L, hidden_dim)
        self.fc2= Conv(hidden_dim, hidden_dim)
        if mab_fc:
            self.pool = PMA(hidden_dim,nheads,1, ln=layer_norm, mab_conv=None)
        else:
            self.pool = PMA(hidden_dim,nheads,1, ln=layer_norm, mab_conv=Conv)
        self.classifier = nn.Linear(hidden_dim,1)

    def forward(self, data, return_attn=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.fc1(x, edge_index))
        x = F.relu(self.fc2(x,edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        batch_x, mask = to_dense_batch(x, batch)
        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9
        graph = (x, edge_index, batch)

        if self.mab_fc:
            h, A, A_raw = self.pool(batch_x, attention_mask=mask, return_attn=True)
        else:
            h, A, A_raw = self.pool(batch_x, graph=graph, attention_mask=mask, return_attn=True)
        
        y = self.classifier(h.squeeze(1))

        if return_attn:
            return y, A, A_raw
        else:
            return y

## same as graph set transformer model but attention module is replaced with simple averaging
class graph_set_mean(torch.nn.Module):
    def __init__(self,L, hidden_dim, dropout=0.0,Conv=GCNConv,layer_norm=False):
        super(graph_set_mean,self).__init__()
        self.dropout_ratio = dropout
        self.fc1 = Conv(L, hidden_dim)
        self.fc2= Conv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim,1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.fc1(x, edge_index))
        x = F.relu(self.fc2(x,edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        h = gap(x,batch)

        y = self.classifier(h)

        return y



