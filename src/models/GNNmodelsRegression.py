import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d
from torch_geometric.nn import EdgeConv, GCNConv, GraphConv, ResGatedGraphConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax, scatter
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
    def __init__(self, hidden_dim, output_dim=1, dropout_prob=0.2):
        super(RegressionHead, self).__init__()
        
        self.output_dim = output_dim  # Number of regression targets (always 1 since we train separate models per target)
        
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer for regularization (prevents overfitting)
        self.regressor = nn.Linear(hidden_dim, output_dim)  # Single linear layer for regression prediction
        
        nn.init.xavier_normal_(self.regressor.weight)  # Xavier normal initialization for weight matrix
        nn.init.zeros_(self.regressor.bias)  # Zero initialization for bias (neutral starting point)
    
    def forward(self, x):
        x = self.dropout(x)  
        
        prediction = self.regressor(x)  
        
        return prediction  


class simple_GCN_res_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1, activation='identity'):
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
            dropout_prob=dropout_prob
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

        return x, x_mean  # FIXED: Standardized return signature
    

class simple_GCN_res_plus_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1, activation='identity'):
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
            dropout_prob=dropout_prob
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
                 activation='identity'):
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
            dropout_prob=dropout_prob
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

        x_mean = global_mean_pool(X_5, batch)
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x, x_mean  # FIXED: Standardized return signature
    
class simple_RGGC_plus_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1, activation='identity'):
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
            dropout_prob=dropout_prob
        )

    def forward(self, X, edge_index, batch):
        X_1 = F.relu(self.conv1(X, edge_index))
        X_1 = self.bn1(X_1)
        X_2 = F.relu(self.conv2(X_1, edge_index))
        X_2 = self.dropout(X_2)   
        X_2 = self.bn2(X_2) + X_1  # FIXED: Added residual connection
        X_3 = F.relu(self.conv3(X_2, edge_index)) 
        X_3 = self.dropout(X_3)
        X_3 = self.bn3(X_3) + X_2  # FIXED: Added residual connection
        X_4 = F.relu(self.conv4(X_3, edge_index))
        X_4 = self.dropout(X_4)
        X_4 = self.bn4(X_4) + X_3  # FIXED: Added residual connection
        X_5 = F.relu(self.conv5(X_4, edge_index)) 
        X_5 = self.dropout(X_5)
        X_5 = self.bn5(X_5) + X_4  # FIXED: Added residual connection

        x_mean = global_mean_pool(X_5, batch)
        feat = x_mean
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x, feat
    
class simple_GAT_regression(torch.nn.Module): 
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1, 
                 num_heads=1, activation='identity'):
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
            dropout_prob=dropout_prob
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


class simple_GraphTransformer_regression(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.5, input_channel=1,
                 num_heads=8, num_layers=4, activation='identity'):
        super(simple_GraphTransformer_regression, self).__init__()

        # Graph Transformer layers - handle multi-head attention correctly
        self.conv1 = TransformerConv(input_channel, hidden_channels, heads=num_heads, dropout=dropout_prob)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
        self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
        self.conv4 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
        self.conv5 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
        
        # Batch normalization layers - account for multi-head attention output dimensions
        self.bn1 = BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = BatchNorm1d(hidden_channels * num_heads)
        self.bn3 = BatchNorm1d(hidden_channels * num_heads)
        self.bn4 = BatchNorm1d(hidden_channels * num_heads)
        self.bn5 = BatchNorm1d(hidden_channels * num_heads)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        # Enhanced regression head - account for multi-head attention output
        self.regression_head = RegressionHead(
            hidden_dim=hidden_channels * num_heads,
            output_dim=output_dim,
            dropout_prob=dropout_prob
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


class KnowledgeGuidedGraphTransformer(nn.Module):
    def __init__(self, hidden_channels=256, output_dim=1, dropout_prob=0.1, 
                 input_channel=1, num_heads=8, num_layers=4, 
                 estimate_uncertainty=True, use_edge_features=True):
        super(KnowledgeGuidedGraphTransformer, self).__init__()
        
        # Store configuration
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.use_edge_features = use_edge_features
        self.estimate_uncertainty = estimate_uncertainty
        
        # Input validation
        assert hidden_channels > 0, f"hidden_channels must be positive, got {hidden_channels}"
        assert output_dim > 0, f"output_dim must be positive, got {output_dim}"
        assert input_channel > 0, f"input_channel must be positive, got {input_channel}"
        assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"
        assert 0 <= dropout_prob <= 1, f"dropout_prob must be in [0,1], got {dropout_prob}"
        assert hidden_channels % num_heads == 0, f"hidden_channels must be divisible by num_heads"
        
        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_channel, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        
        # Graph Transformer layers (key innovation vs RGGC)
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_channels, 
                num_heads=num_heads, 
                dropout_prob=dropout_prob,
                use_edge_features=use_edge_features
            )
            for _ in range(num_layers)
        ])
        
        # Enhanced regression head
        self.regression_head = EnhancedRegressionHead(
            hidden_channels, 
            output_dim, 
            dropout_prob, 
            estimate_uncertainty
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization for transformers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _validate_inputs(self, x, edge_index, batch, edge_attr=None):
        """Validate all inputs to prevent runtime errors"""
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {x.dim()}D")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index shape [2, num_edges], got {edge_index.shape}")
        if batch is None:
            raise ValueError("Batch tensor cannot be None")
        if batch.dim() != 1:
            raise ValueError(f"Expected batch to be 1D, got {batch.dim()}D")
        if x.size(0) != batch.size(0):
            raise ValueError(f"x and batch size mismatch: {x.size(0)} vs {batch.size(0)}")
        if edge_attr is not None:
            if edge_attr.size(0) != edge_index.size(1):
                raise ValueError(f"edge_attr size {edge_attr.size(0)} != num_edges {edge_index.size(1)}")
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Input validation
        self._validate_inputs(x, edge_index, batch, edge_attr)
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply transformer layers with proper residual connections
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, edge_index, edge_attr)
        
        # Global pooling for graph-level representation
        x_mean = global_mean_pool(X_5, batch)
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x, x_mean  


# Factory function for creating Graph Transformer models
def create_knowledge_guided_graph_transformer(hidden_channels=256, output_dim=1, 
                                            dropout_prob=0.1, input_channel=1,
                                            num_heads=8, num_layers=4,
                                            use_edge_features=True):
    """
    Factory function to create clean Graph Transformer model.
    
    Args:
        hidden_channels: Hidden dimension size
        output_dim: Output dimension (number of targets)  
        dropout_prob: Dropout probability
        input_channel: Input feature dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers (currently fixed at 5 for consistency with RGGC)
        use_edge_features: Whether to use edge features (kept for compatibility)
        
    Returns:
        simple_GraphTransformer_regression model
    """
    return simple_GraphTransformer_regression(
        hidden_channels=hidden_channels,
        output_dim=output_dim, 
        dropout_prob=dropout_prob,
        input_channel=input_channel,
        num_heads=num_heads,
    )


class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for a Gaussian distribution with predicted mean and variance.
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