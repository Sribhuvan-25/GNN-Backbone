import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv, ResGatedGraphConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_sparse import spmm

import networkx as nx
import scipy.sparse as sp
from networkx.convert_matrix import from_numpy_matrix
import csv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, DummyExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
from torch_geometric.utils import to_networkx
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support



class simple_GCN_res(torch.nn.Module):
    def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel):
        super(simple_GCN_res, self).__init__()

        # Initialize GCN layers
        self.conv1 = GCNConv(input_channel, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)


        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)


        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = Linear(hidden_channels, n_classes)

    def forward(self, X, edge_index, batch):
        X_1 = F.relu(self.conv1(X, edge_index))
        X_1 = self.bn1(X_1)
        X_2 = F.relu(self.conv2(X_1, edge_index))
        X_2 = self.dropout(X_2)
        X_2 = self.bn2(X_2) + X_1  
        X_3 = F.relu(self.conv3(X_2, edge_index))
        X_3 = self.dropout(X_3)
        X_3 = self.bn3(X_3) + X_2  
        X_4 = F.relu(self.conv4(X_3, edge_index))
        X_4 = self.dropout(X_4)
        X_4 = self.bn4(X_4) + X_3  
        X_5 = F.relu(self.conv5(X_4, edge_index))
        X_5 = self.dropout(X_5)
        X_5 = self.bn5(X_5) + X_4  


        x = global_mean_pool(X_5, batch)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)

        return x
    

class simple_GCN_res_plus(torch.nn.Module):
    def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel):
        super(simple_GCN_res_plus, self).__init__()

        # Initialize GCN layers
        self.conv1 = GCNConv(input_channel, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)


        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)


        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = Linear(hidden_channels, n_classes)

    def forward(self, X, edge_index, batch):
        X_1 = F.relu(self.conv1(X, edge_index))
        X_1 = self.bn1(X_1)
        X_2 = F.relu(self.conv2(X_1, edge_index))
        X_2 = self.dropout(X_2)
        X_2 = self.bn2(X_2) + X_1  
        X_3 = F.relu(self.conv3(X_2, edge_index))
        X_3 = self.dropout(X_3)
        X_3 = self.bn3(X_3) + X_2  
        X_4 = F.relu(self.conv4(X_3, edge_index))
        X_4 = self.dropout(X_4)
        X_4 = self.bn4(X_4) + X_3  
        X_5 = F.relu(self.conv5(X_4, edge_index))
        X_5 = self.dropout(X_5)
        X_5 = self.bn5(X_5) + X_4  


        x = global_mean_pool(X_5, batch)
        feat = global_mean_pool(X_5, batch)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)

        return x, feat
    

class simple_RGGC(torch.nn.Module):
    def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel):
        super(simple_RGGC, self).__init__()

        self.conv1 = ResGatedGraphConv(input_channel, hidden_channels)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv4 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv5 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = Linear(hidden_channels, n_classes)

    def forward(self,  X, edge_index, batch):

        X_1 = F.relu(self.conv1(X, edge_index))
        X_1 = self.bn1(X_1)
        X_2 = F.relu(self.conv2(X_1, edge_index))
        X_2 = self.dropout(X_2)   
        X_2 = self.bn2(X_2)
        X_3 = F.relu(self.conv3(X_2, edge_index)) 
        X_3 = self.dropout(X_3)
        X_3 = self.bn3(X_3)
        X_4 = F.relu(self.conv4(X_3, edge_index))
        X_4 = self.dropout(X_4)
        X_4 = self.bn4(X_4)
        X_5 = F.relu(self.conv5(X_4, edge_index)) 
        X_5 = self.dropout(X_5)
        X_5 = self.bn5(X_5)

        x = global_mean_pool(X_5,batch)

        x = self.linear(x)
        
        x = F.softmax(x, dim=-1)

        return x
    
class simple_RGGC_plus(torch.nn.Module):
    def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel):
        super(simple_RGGC_plus, self).__init__()

        self.conv1 = ResGatedGraphConv(input_channel, hidden_channels)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv4 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv5 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = Linear(hidden_channels, n_classes)

    def forward(self,  X, edge_index, batch):

        X_1 = F.relu(self.conv1(X, edge_index))
        X_1 = self.bn1(X_1)
        X_2 = F.relu(self.conv2(X_1, edge_index))
        X_2 = self.dropout(X_2)   
        X_2 = self.bn2(X_2)
        X_3 = F.relu(self.conv3(X_2, edge_index)) 
        X_3 = self.dropout(X_3)
        X_3 = self.bn3(X_3)
        X_4 = F.relu(self.conv4(X_3, edge_index))
        X_4 = self.dropout(X_4)
        X_4 = self.bn4(X_4)
        X_5 = F.relu(self.conv5(X_4, edge_index)) 
        X_5 = self.dropout(X_5)
        X_5 = self.bn5(X_5)

        x = global_mean_pool(X_5,batch)
        feat = global_mean_pool(X_5, batch)

        x = self.linear(x)
        
        x = F.softmax(x, dim=-1)

        return x, feat
    
class simple_GAT(torch.nn.Module): 
    def __init__(self, hidden_channels, n_classes,dropout_prob, input_channel, num_heads=1):
        super(simple_GAT, self).__init__()

        self.conv1 = GATConv(input_channel, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv5 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = Linear(hidden_channels * num_heads, n_classes)

    def forward(self, X, edge_index, batch):

        X_1 = F.relu(self.conv1(X, edge_index))
        X_1 = self.bn1(X_1)
        X_2 = F.relu(self.conv2(X_1, edge_index))
        X_2 = self.dropout(X_2) 
        X_2 = self.bn2(X_2)
        X_3 = F.relu(self.conv3(X_2, edge_index))
        X_3 = self.dropout(X_3)
        X_3 = self.bn3(X_3) 
        X_4 = F.relu(self.conv4(X_3, edge_index))  
        X_4 = self.dropout(X_4) 
        X_4 = self.bn4(X_4) 
        X_5 = F.relu(self.conv5(X_4, edge_index))
        X_5 = self.dropout(X_5)  
        X_5 = self.bn5(X_5)
        
        x = global_mean_pool(X_5, batch)
        feat = global_mean_pool(X_5, batch)
        x = self.linear(x)
        
        x = F.softmax(x, dim=-1)

        return x, feat