import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv, ResGatedGraphConv, GATConv, DynamicEdgeConv
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

class Enhanced_DGCNN_regression(nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1,
                 estimate_uncertainty=False, activation='identity', k=5, num_layers=5):
        super(Enhanced_DGCNN_regression, self).__init__()
        
        self.num_layers = num_layers
        self.k = k
        
        # Multi-scale k values for different layers
        k_values = [k, k//2 + 1, k//3 + 1, k//4 + 1, k//5 + 1] if num_layers == 5 else [k] * num_layers
        
        # Enhanced DynamicEdgeConv layers with varying k and deeper MLPs
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(DynamicEdgeConv(
            nn=Sequential(
                Linear(2 * input_channel, hidden_channels),
                ReLU(),
                BatchNorm1d(hidden_channels),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            ),
            k=k_values[0], aggr='max'
        ))
        self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Subsequent layers with residual connections and varying hidden dimensions
        for i in range(1, num_layers):
            # Gradually increase hidden dimensions
            curr_hidden = hidden_channels + (i * hidden_channels // 4)
            prev_hidden = hidden_channels + ((i-1) * hidden_channels // 4) if i > 1 else hidden_channels
            
            self.conv_layers.append(DynamicEdgeConv(
                nn=Sequential(
                    Linear(2 * prev_hidden, curr_hidden),
                    ReLU(), 
                    BatchNorm1d(curr_hidden),
                    Dropout(dropout_prob * 0.5),  # Lighter dropout in MLPs
                    Linear(curr_hidden, curr_hidden),
                    ReLU(),
                    Linear(curr_hidden, curr_hidden)
                ),
                k=k_values[i], aggr='max'
            ))
            self.bn_layers.append(BatchNorm1d(curr_hidden))
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_prob)
        self.light_dropout = nn.Dropout(dropout_prob * 0.3)
        
        # Multi-scale pooling
        final_hidden = hidden_channels + ((num_layers-1) * hidden_channels // 4)
        pooling_dim = final_hidden * 3  # mean + max + std pooling
        
        # Attention mechanism for pooling
        self.attention = nn.Sequential(
            Linear(final_hidden, final_hidden // 2),
            ReLU(),
            Linear(final_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        # Enhanced regression head with skip connections
        self.regression_head = Enhanced_RegressionHead(
            hidden_dim=pooling_dim,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            estimate_uncertainty=estimate_uncertainty,
            activation=activation
        )
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList()
        for i in range(num_layers):
            curr_hidden = hidden_channels + (i * hidden_channels // 4)
            self.skip_projections.append(Linear(curr_hidden, final_hidden))
    
    def forward(self, x, edge_index, batch):
        # Store skip connections
        skip_connections = []
        
        # Forward through layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x_new = F.relu(conv(x, batch=batch))
            x_new = bn(x_new)
            
            # Apply dropout (lighter for earlier layers)
            if i < 2:
                x_new = self.light_dropout(x_new)
            else:
                x_new = self.dropout(x_new)
            
            # Residual connection with projection if needed
            if i > 0:
                # Project previous output to match current dimensions
                x_projected = self.skip_projections[i-1](x)
                if x_projected.shape[1] == x_new.shape[1]:
                    x_new = x_new + x_projected
            
            skip_connections.append(x_new)
            x = x_new
        
        # Multi-scale pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Attention-weighted pooling
        attention_weights = self.attention(x)
        x_att = global_add_pool(x * attention_weights, batch)
        
        # Standard deviation pooling
        x_std = []
        unique_batches = torch.unique(batch)
        for b in unique_batches:
            mask = batch == b
            node_feats = x[mask]
            std_feat = torch.std(node_feats, dim=0, keepdim=True)
            x_std.append(std_feat)
        x_std = torch.cat(x_std, dim=0)
        
        # Combine all pooling strategies
        graph_emb = torch.cat([x_mean, x_max, x_std], dim=1)
        feat = graph_emb
        
        # Regression prediction
        out = self.regression_head(graph_emb)
        return out, feat


class Enhanced_RegressionHead(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, dropout_prob=0.2, 
                 estimate_uncertainty=False, activation='identity'):
        super(Enhanced_RegressionHead, self).__init__()
        
        self.estimate_uncertainty = estimate_uncertainty
        
        # Enhanced multi-layer head with skip connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                ReLU(),
                nn.Dropout(dropout_prob)
            ),
            nn.Sequential(
                Linear(hidden_dim // 2, hidden_dim // 4), 
                nn.LayerNorm(hidden_dim // 4),
                ReLU(),
                nn.Dropout(dropout_prob * 0.5)
            ),
            nn.Sequential(
                Linear(hidden_dim // 4, hidden_dim // 8),
                nn.LayerNorm(hidden_dim // 8), 
                ReLU(),
                nn.Dropout(dropout_prob * 0.3)
            )
        ])
        
        # Skip connection projections
        self.skip_proj1 = Linear(hidden_dim, hidden_dim // 4)
        self.skip_proj2 = Linear(hidden_dim, hidden_dim // 8)
        
        if estimate_uncertainty:
            # Separate heads for mean and variance
            self.mean_head = Linear(hidden_dim // 8, output_dim)
            self.var_head = Linear(hidden_dim // 8, output_dim)
        else:
            self.output_layer = Linear(hidden_dim // 8, output_dim)
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x_orig = x
        
        # Forward through layers with skip connections
        x = self.layers[0](x)
        
        x = self.layers[1](x)
        # Skip connection from input to layer 2
        x = x + self.skip_proj1(x_orig)
        
        x = self.layers[2](x)
        # Skip connection from input to final layer
        x = x + self.skip_proj2(x_orig)
        
        if self.estimate_uncertainty:
            mean = self.mean_head(x)
            var = F.softplus(self.var_head(x)) + 1e-6
            return mean, var
        else:
            output = self.output_layer(x)
            return self.activation(output)


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