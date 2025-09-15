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

# Import enhanced Graph Transformer
try:
    from .enhanced_graph_transformer import create_enhanced_graph_transformer
    HAS_ENHANCED_GT = True
except ImportError:
    try:
        from enhanced_graph_transformer import create_enhanced_graph_transformer
        HAS_ENHANCED_GT = True
    except ImportError:
        HAS_ENHANCED_GT = False
        print("Enhanced Graph Transformer not available")
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

        # Global pooling for graph-level representation
        x_mean = global_mean_pool(X_5, batch)
        
        # Pass through regression head
        x = self.regression_head(x_mean)
        
        return x, x_mean


# Note: The KnowledgeGuidedGraphTransformer was removed due to missing dependencies
# Use simple_GraphTransformer_regression instead for Graph Transformer functionality


# Factory function for creating Graph Transformer models
def create_simple_graph_transformer(hidden_channels=256, output_dim=1, 
                                   dropout_prob=0.1, input_channel=1,
                                   num_heads=8, num_layers=4):
    """
    Factory function to create simple Graph Transformer model.
    
    Args:
        hidden_channels: Hidden dimension size
        output_dim: Output dimension (number of targets)  
        dropout_prob: Dropout probability
        input_channel: Input feature dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers (currently fixed at 5 for consistency)
        
    Returns:
        simple_GraphTransformer_regression model
    """
    return simple_GraphTransformer_regression(
        hidden_channels=hidden_channels,
        output_dim=output_dim, 
        dropout_prob=dropout_prob,
        input_channel=input_channel,
        num_heads=num_heads,
        num_layers=num_layers
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


class enhanced_GraphTransformer_regression(torch.nn.Module):
    """
    Enhanced Graph Transformer for regression with proper architecture
    
    Key improvements over simple_GraphTransformer_regression:
    1. Proper positional encoding (Laplacian-based for biological graphs)
    2. LayerNorm instead of BatchNorm (standard for Transformers) 
    3. Residual connections and proper normalization
    4. Attention visualization capabilities
    5. Flexible pooling strategies
    
    This model should be used instead of simple_GraphTransformer_regression
    for research publication purposes.
    """
    
    def __init__(self, hidden_channels, output_dim=1, dropout_prob=0.1, input_channel=1,
                 num_heads=8, num_layers=4, activation='relu'):
        super(enhanced_GraphTransformer_regression, self).__init__()
        
        # Store parameters
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.input_channel = input_channel
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if HAS_ENHANCED_GT:
            # Use the enhanced implementation
            self.enhanced_gt = create_enhanced_graph_transformer(
                input_dim=input_channel,
                hidden_dim=hidden_channels,
                output_dim=output_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout_prob,
                positional_encoding='laplacian',  # Best for biological networks
                pooling='mean',
                activation=activation,
                use_skip_connections=True
            )
        else:
            # Fallback to simple implementation with LayerNorm fixes
            print("Using fallback Graph Transformer implementation")
            self.conv1 = TransformerConv(input_channel, hidden_channels, heads=num_heads, dropout=dropout_prob)
            self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
            self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
            self.conv4 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout_prob)
            
            # Use LayerNorm instead of BatchNorm (proper for Transformers)
            self.ln1 = torch.nn.LayerNorm(hidden_channels * num_heads)
            self.ln2 = torch.nn.LayerNorm(hidden_channels * num_heads)  
            self.ln3 = torch.nn.LayerNorm(hidden_channels * num_heads)
            self.ln4 = torch.nn.LayerNorm(hidden_channels * num_heads)
            
            self.dropout = torch.nn.Dropout(dropout_prob)
            
            # Regression head
            self.regression_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * num_heads, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_prob),
                torch.nn.Linear(hidden_channels, output_dim)
            )
            
    def forward(self, X, edge_index, batch, return_attention=False):
        """
        Forward pass of Enhanced Graph Transformer
        
        Args:
            X: Node features [num_nodes, input_channel]
            edge_index: Graph connectivity [2, num_edges] 
            batch: Batch assignment [num_nodes]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Predictions [batch_size, output_dim]
            graph_repr: Graph representations [batch_size, hidden_channels] (optional)
            attention_weights: Attention weights (optional)
        """
        
        if HAS_ENHANCED_GT:
            # Use enhanced implementation
            if return_attention:
                output, graph_repr, attention_weights = self.enhanced_gt(
                    X, edge_index, batch, return_attention=True
                )
                return output, graph_repr, attention_weights
            else:
                output, graph_repr = self.enhanced_gt(
                    X, edge_index, batch, return_attention=False
                )
                return output
        else:
            # Fallback implementation with LayerNorm fixes
            # Layer 1
            X_1 = F.relu(self.conv1(X, edge_index))
            X_1 = self.ln1(X_1)
            X_1 = self.dropout(X_1)
            
            # Layer 2 with residual connection
            X_2 = F.relu(self.conv2(X_1, edge_index))
            X_2 = self.ln2(X_2 + X_1)  # Residual connection
            X_2 = self.dropout(X_2)
            
            # Layer 3 with residual connection  
            X_3 = F.relu(self.conv3(X_2, edge_index))
            X_3 = self.ln3(X_3 + X_2)  # Residual connection
            X_3 = self.dropout(X_3)
            
            # Layer 4 with residual connection
            X_4 = F.relu(self.conv4(X_3, edge_index))
            X_4 = self.ln4(X_4 + X_3)  # Residual connection
            X_4 = self.dropout(X_4)
            
            # Global pooling
            graph_repr = global_mean_pool(X_4, batch)
            
            # Regression prediction
            output = self.regression_head(graph_repr)
            
            if return_attention:
                # For fallback, no attention weights available
                return output, graph_repr, None
            else:
                return output 