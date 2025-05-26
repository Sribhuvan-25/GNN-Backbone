import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv, ResGatedGraphConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import from_networkx

import networkx as nx
import scipy.sparse as sp
import csv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, DummyExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
from torch_geometric.utils import to_networkx
from sklearn.model_selection import KFold


class RegressionHead(nn.Module):
    """
    Enhanced regression head that leverages GNN representations more effectively.
    Features:
    - Multi-level feature aggregation
    - Uncertainty estimation (optional)
    - Attention mechanism for feature importance
    - Multiple activation options for different regression tasks
    """
    def __init__(self, hidden_dim, output_dim=1, dropout_prob=0.2, 
                 estimate_uncertainty=False, activation='identity'):
        super(RegressionHead, self).__init__()
        
        self.estimate_uncertainty = estimate_uncertainty
        self.output_dim = output_dim
        
        # Feature transformation layers
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Main regression output
        self.regressor = nn.Linear(hidden_dim, output_dim)
        
        # Uncertainty estimation (optional)
        if estimate_uncertainty:
            self.uncertainty = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:  # 'identity' or any other value
            self.activation = nn.Identity()
    
    def forward(self, x):
        # Transform features
        x_transformed = self.feature_transform(x)
        
        # Apply attention weights
        attention_weights = torch.sigmoid(self.attention(x_transformed))
        x_weighted = x_transformed * attention_weights
        
        # Main regression output
        prediction = self.regressor(x_weighted)
        prediction = self.activation(prediction)
        
        if self.estimate_uncertainty:
            # Uncertainty estimation (log variance)
            uncertainty = self.uncertainty(x_weighted)
            uncertainty = F.softplus(uncertainty) + 1e-6  # ensure positive variance
            return prediction, uncertainty
        else:
            return prediction


class simple_GCN_res_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1, 
                 estimate_uncertainty=False, activation='identity'):
        super(simple_GCN_res_regression, self).__init__()

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
        
        # Enhanced regression head
        self.regression_head = RegressionHead(
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            estimate_uncertainty=estimate_uncertainty,
            activation=activation
        )

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

        # Multi-level pooling for better graph representation
        x_mean = global_mean_pool(X_5, batch)
        
        # Pass through regression head
        x = self.regression_head(x_mean)

        return x
    

class simple_GCN_res_plus_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1,
                 estimate_uncertainty=False, activation='identity'):
        super(simple_GCN_res_plus_regression, self).__init__()

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
        
        # Enhanced regression head
        self.regression_head = RegressionHead(
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            estimate_uncertainty=estimate_uncertainty,
            activation=activation
        )

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

        # Multi-level pooling for better graph representation
        x_mean = global_mean_pool(X_5, batch)
        
        # Store node embeddings for feature extraction
        feat = x_mean
        
        # Pass through regression head
        x = self.regression_head(x_mean)

        return x, feat
    

class simple_RGGC_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1,
                 estimate_uncertainty=False, activation='identity'):
        super(simple_RGGC_regression, self).__init__()

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
        
        # Enhanced regression head
        self.regression_head = RegressionHead(
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            estimate_uncertainty=estimate_uncertainty,
            activation=activation
        )

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

        x_mean = global_mean_pool(X_5, batch)
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x
    
class simple_RGGC_plus_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1,
                 estimate_uncertainty=False, activation='identity'):
        super(simple_RGGC_plus_regression, self).__init__()

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
        
        # Enhanced regression head
        self.regression_head = RegressionHead(
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            estimate_uncertainty=estimate_uncertainty,
            activation=activation
        )

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

        x_mean = global_mean_pool(X_5, batch)
        feat = x_mean
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x, feat
    
class simple_GAT_regression(torch.nn.Module): 
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1, 
                 num_heads=1, estimate_uncertainty=False, activation='identity'):
        super(simple_GAT_regression, self).__init__()

        self.conv1 = GATConv(input_channel, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv5 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)

        self.bn1 = BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = BatchNorm1d(hidden_channels * num_heads)
        self.bn3 = BatchNorm1d(hidden_channels * num_heads)
        self.bn4 = BatchNorm1d(hidden_channels * num_heads)
        self.bn5 = BatchNorm1d(hidden_channels * num_heads)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        # Enhanced regression head
        self.regression_head = RegressionHead(
            hidden_dim=hidden_channels * num_heads,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            estimate_uncertainty=estimate_uncertainty,
            activation=activation
        )

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
        
        x_mean = global_mean_pool(X_5, batch)
        feat = x_mean
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x, feat


# Example loss function for regression with uncertainty
class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for a Gaussian distribution with predicted mean and variance.
    Useful when the model predicts both the mean and uncertainty of regression targets.
    """
    def __init__(self, eps=1e-6, reduction='mean'):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_mean, pred_var, target):
        # Ensure variance is positive
        pred_var = pred_var.clamp(min=self.eps)
        
        # Negative log likelihood of Gaussian
        loss = 0.5 * (torch.log(pred_var) + (pred_mean - target)**2 / pred_var)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 