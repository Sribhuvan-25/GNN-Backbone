# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, GraphSAGE, ResGatedGraphConv
# from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.explain import Explainer, GNNExplainer
# from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d


# class MethaneGNN(nn.Module):
#     """Base GNN model for methane prediction with model explainability"""
    
#     def __init__(self, 
#                  num_node_features=1, 
#                  hidden_dim=64, 
#                  num_targets=2, 
#                  dropout_rate=0.3,
#                  num_layers=4, 
#                  model_type='gat'):
#         super(MethaneGNN, self).__init__()
        
#         self.num_layers = num_layers
#         self.model_type = model_type
#         self.hidden_dim = hidden_dim
        
#         # Choose GNN layer type
#         if model_type == 'gcn':
#             self.conv_layers = nn.ModuleList()
#             # First layer
#             self.conv_layers.append(GCNConv(num_node_features, hidden_dim))
#             # Middle layers
#             for i in range(num_layers - 1):
#                 self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
#         elif model_type == 'gat':
#             self.conv_layers = nn.ModuleList()
#             # First layer with 8 attention heads
#             self.conv_layers.append(GATConv(num_node_features, hidden_dim // 8, heads=8, dropout=dropout_rate))
#             # Middle layers
#             for i in range(num_layers - 2):
#                 self.conv_layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout_rate))
#             # Final GAT layer with 1 attention head for concatenation
#             self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate))
            
#         elif model_type == 'gatv2':
#             self.conv_layers = nn.ModuleList()
#             # First layer with 8 attention heads
#             self.conv_layers.append(GATv2Conv(num_node_features, hidden_dim // 8, heads=8, dropout=dropout_rate))
#             # Middle layers
#             for i in range(num_layers - 2):
#                 self.conv_layers.append(GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout_rate))
#             # Final GATv2 layer
#             self.conv_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate))
        
#         elif model_type == 'gin':
#             self.conv_layers = nn.ModuleList()
#             # For GIN, we need to define MLPs for each layer
#             mlp1 = nn.Sequential(
#                 nn.Linear(num_node_features, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
#             self.conv_layers.append(GINConv(mlp1))
            
#             # Middle GIN layers
#             for i in range(num_layers - 1):
#                 mlp = nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     nn.BatchNorm1d(hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(hidden_dim, hidden_dim)
#                 )
#                 self.conv_layers.append(GINConv(mlp))
        
#         # Batch normalization layers
#         self.batch_norms = nn.ModuleList()
#         for i in range(num_layers):
#             self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
#         # Attention pooling layer
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim // 2, 1)
#         )
        
#         # Output MLP - replacing with enhanced regression head
#         # Enhanced regression head with feature interaction, residual connections and self-attention
#         self.feature_interaction = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.LayerNorm(hidden_dim * 2),
#             nn.GELU(),  # GELU activation often works better for regression
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#         # Self-attention for feature refinement
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=4,
#             dropout=dropout_rate,
#             batch_first=True
#         )
        
#         # Feature compression with residual connection
#         self.compress = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )
        
#         # Cross-feature bilinear interaction layer
#         self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim // 2)
        
#         # Final prediction layers with residual connections
#         self.final_layers = nn.Sequential(
#             nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout_rate // 2),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.GELU(),
#             nn.Linear(hidden_dim // 4, num_targets)
#         )
        
#         # Optional: Uncertainty estimation (aleatoric uncertainty)
#         self.estimate_uncertainty = True
#         if self.estimate_uncertainty:
#             self.uncertainty_layer = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 2),
#                 nn.LayerNorm(hidden_dim // 2),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim // 2, num_targets)
#             )
        
#         # Initialize weights
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         """Initialize weights using Xavier initialization"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
    
#     def attention_pooling(self, x, batch):
#         """Attention pooling to weight node importance"""
#         # Compute attention weights
#         attn_weights = self.attention(x).squeeze(-1)
        
#         # Apply softmax over nodes in same graph (per batch)
#         output = torch.zeros(batch.max().item() + 1, x.size(-1), device=x.device)
        
#         for i in range(batch.max().item() + 1):
#             mask = (batch == i)
#             # Get nodes for this graph
#             graph_x = x[mask]
#             # Get attention weights for this graph
#             graph_weights = attn_weights[mask]
#             # Apply softmax to get normalized weights
#             graph_weights = F.softmax(graph_weights, dim=0)
#             # Apply attention pooling
#             graph_output = torch.sum(graph_x * graph_weights.unsqueeze(-1), dim=0)
#             # Store result
#             output[i] = graph_output
            
#         return output
    
#     def forward(self, data):
#         """Forward pass through the network"""
#         x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
#         # Handle batch being None for single graph
#         if batch is None:
#             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
#         # Initial features
#         last_x = x
        
#         # Apply GNN layers with residual connections and batch normalization
#         for i, conv in enumerate(self.conv_layers):
#             if self.model_type == 'gcn':
#                 # For GCN, we can pass edge weights
#                 x = conv(x, edge_index, edge_weight=edge_weight)
#             else:
#                 # For other models like GAT, GIN
#                 x = conv(x, edge_index)
            
#             x = self.batch_norms[i](x)
#             x = F.relu(x)
            
#             # Apply residual connection for all but first layer
#             if i > 0 and x.size() == last_x.size():
#                 x = x + last_x
            
#             # Apply dropout except last layer
#             if i < len(self.conv_layers) - 1:
#                 x = F.dropout(x, p=0.2, training=self.training)
            
#             last_x = x
        
#         # Multiple pooling methods combined
#         x_mean = global_mean_pool(x, batch)
#         x_add = global_add_pool(x, batch)
#         # Also use attention pooling
#         x_attn = self.attention_pooling(x, batch)
        
#         # Combine different pooling methods
#         x_combined = x_mean + 0.5 * x_attn + 0.1 * x_add
        
#         # Apply enhanced regression head
#         # 1. Feature interaction and expansion
#         x_interact = self.feature_interaction(x_combined)
        
#         # 2. Self-attention for contextual feature refinement
#         # Reshape for attention (add sequence length dimension)
#         x_attn_input = x_interact.unsqueeze(1)
#         x_attn_out, _ = self.self_attn(x_attn_input, x_attn_input, x_attn_input)
#         x_attn_out = x_attn_out.squeeze(1)
        
#         # 3. Apply residual connection
#         x_refined = x_interact + x_attn_out
        
#         # 4. Feature compression with another residual connection
#         x_compressed = self.compress(x_refined) + x_refined
        
#         # 5. Bilinear feature interactions to capture pairwise interactions
#         x_bilinear = self.bilinear(x_compressed, x_compressed)
        
#         # 6. Concatenate main features with interaction features
#         x_combined_features = torch.cat([x_compressed, x_bilinear], dim=1)
        
#         # 7. Final prediction
#         predictions = self.final_layers(x_combined_features)
        
#         # 8. Optional: Uncertainty estimation
#         if self.estimate_uncertainty and self.training:
#             # Log variance for numerical stability
#             log_var = self.uncertainty_layer(x_compressed)
#             return predictions, log_var
        
#         return predictions
    
#     def get_embeddings(self, data):
#         """Get node embeddings from the model"""
#         x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
#         # Handle batch being None for single graph
#         if batch is None:
#             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
#         # Initial features
#         last_x = x
        
#         # Apply GNN layers
#         for i, conv in enumerate(self.conv_layers):
#             if self.model_type == 'gcn':
#                 x = conv(x, edge_index, edge_weight=edge_weight)
#             else:
#                 x = conv(x, edge_index)
            
#             x = self.batch_norms[i](x)
#             x = F.relu(x)
            
#             if i > 0 and x.size() == last_x.size():
#                 x = x + last_x
            
#             if i < len(self.conv_layers) - 1:
#                 x = F.dropout(x, p=0.2, training=self.training)
            
#             last_x = x
        
#         # Return node embeddings and graph embedding
#         graph_embed = global_mean_pool(x, batch)
        
#         return x, graph_embed


# class MethaneGNNExplainer:
#     """Wrapper for GNNExplainer to explain predictions of the methane GNN model"""
    
#     def __init__(self, model, device):
#         self.model = model
#         self.device = device
        
#         # Create a proper model wrapper class
#         class ModelWrapper(torch.nn.Module):
#             def __init__(self, model):
#                 super().__init__()
#                 self.model = model
#                 self.training = model.training
                
#             def forward(self, x, edge_index, **kwargs):
#                 batch = kwargs.get('batch', None)
#                 # Create a dummy data object with the expected attributes
#                 from torch_geometric.data import Data
#                 data = Data(x=x, edge_index=edge_index, batch=batch)
#                 if 'edge_weight' in kwargs:
#                     data.edge_weight = kwargs['edge_weight']
                
#                 # Run the model with the data object
#                 return self.model(data)
        
#         # Initialize GNNExplainer with wrapper class
#         self.wrapped_model = ModelWrapper(model)
#         self.explainer = Explainer(
#             model=self.wrapped_model,
#             algorithm=GNNExplainer(epochs=300),
#             explanation_type='model',
#             node_mask_type=None,
#             edge_mask_type='object',
#             model_config=dict(
#                 mode='regression',
#                 task_level='graph',
#                 return_type='raw',
#             ),
#             threshold_config=dict(
#                 threshold_type='topk',
#                 value=20
#             )
#         )
    
#     def explain_graph(self, graph, node_names=None, save_path=None, target_idx=0):
#         """
#         Explain prediction for a single graph
        
#         Args:
#             graph: Graph to explain
#             node_names: List of node names
#             save_path: Path to save the edge importance matrix
#             target_idx: Index of the target to explain (0 for ACE-km, 1 for H2-km)
            
#         Returns:
#             edge_importance_matrix: Matrix of edge importance scores
#             explanation: Raw explanation object
#         """
#         graph = graph.to(self.device)
#         self.model.eval()
        
#         # Get model prediction
#         with torch.no_grad():
#             prediction = self.model(graph)
#             predicted_value = prediction[0, target_idx].item()
#             actual_value = graph.y[0, target_idx].item()
        
#         print(f"Target: {target_idx}, Actual value: {actual_value:.4f}, Predicted value: {predicted_value:.4f}")
        
#         # Create zeros tensor for batch
#         batch = torch.zeros(len(graph.x), dtype=torch.long, device=self.device)
        
#         # Pass edge_weight if available
#         kwargs = {'batch': batch}
#         if hasattr(graph, 'edge_weight'):
#             kwargs['edge_weight'] = graph.edge_weight
        
#         # Get explanation
#         explanation = self.explainer(
#             graph.x, 
#             graph.edge_index,
#             **kwargs
#         )
        
#         # Convert to edge importance matrix
#         num_nodes = graph.x.size(0)
#         edge_importance_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
#         for i, (u, v) in enumerate(zip(graph.edge_index[0], graph.edge_index[1])):
#             edge_importance_matrix[u, v] = explanation.edge_mask[i]
        
#         # Make matrix symmetric for undirected graph
#         edge_importance_matrix = (edge_importance_matrix + edge_importance_matrix.t()) / 2
        
#         # Save edge importance matrix if requested
#         if save_path is not None:
#             import numpy as np
#             np.savetxt(save_path, edge_importance_matrix.cpu().numpy(), delimiter=',')
#             print(f"Saved edge importance matrix to {save_path}")
        
#         return edge_importance_matrix, explanation 

# class simple_GCN_res(torch.nn.Module):
#     def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel):
#         super(simple_GCN_res, self).__init__()

#         # Initialize GCN layers
#         self.conv1 = GCNConv(input_channel, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
#         self.conv4 = GCNConv(hidden_channels, hidden_channels)
#         self.conv5 = GCNConv(hidden_channels, hidden_channels)

#         self.bn1 = BatchNorm1d(hidden_channels)
#         self.bn2 = BatchNorm1d(hidden_channels)
#         self.bn3 = BatchNorm1d(hidden_channels)
#         self.bn4 = BatchNorm1d(hidden_channels)
#         self.bn5 = BatchNorm1d(hidden_channels)

#         self.dropout = torch.nn.Dropout(dropout_prob)
#         self.linear = Linear(hidden_channels, n_classes)

#     def forward(self, data):
#         X, edge_index = data.x, data.edge_index
#         batch = data.batch if hasattr(data, 'batch') else torch.zeros(X.size(0), dtype=torch.long, device=X.device)
        
#         X_1 = F.relu(self.conv1(X, edge_index))
#         X_1 = self.bn1(X_1)
#         X_2 = F.relu(self.conv2(X_1, edge_index))
#         X_2 = self.dropout(X_2)
#         X_2 = self.bn2(X_2) + X_1  
#         X_3 = F.relu(self.conv3(X_2, edge_index))
#         X_3 = self.dropout(X_3)
#         X_3 = self.bn3(X_3) + X_2  
#         X_4 = F.relu(self.conv4(X_3, edge_index))
#         X_4 = self.dropout(X_4)
#         X_4 = self.bn4(X_4) + X_3  
#         X_5 = F.relu(self.conv5(X_4, edge_index))
#         X_5 = self.dropout(X_5)
#         X_5 = self.bn5(X_5) + X_4  

#         x = global_mean_pool(X_5, batch)
#         x = self.linear(x)
        
#         return x

# class simple_RGGC(torch.nn.Module):
#     def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel):
#         super(simple_RGGC, self).__init__()

#         self.conv1 = ResGatedGraphConv(input_channel, hidden_channels)
#         self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels)
#         self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels)
#         self.conv4 = ResGatedGraphConv(hidden_channels, hidden_channels)
#         self.conv5 = ResGatedGraphConv(hidden_channels, hidden_channels)
#         self.bn1 = BatchNorm1d(hidden_channels)
#         self.bn2 = BatchNorm1d(hidden_channels)
#         self.bn3 = BatchNorm1d(hidden_channels)
#         self.bn4 = BatchNorm1d(hidden_channels)
#         self.bn5 = BatchNorm1d(hidden_channels)
#         self.dropout = torch.nn.Dropout(dropout_prob)
#         self.linear = Linear(hidden_channels, n_classes)

#     def forward(self, data):
#         X, edge_index = data.x, data.edge_index
#         batch = data.batch if hasattr(data, 'batch') else torch.zeros(X.size(0), dtype=torch.long, device=X.device)

#         X_1 = F.relu(self.conv1(X, edge_index))
#         X_1 = self.bn1(X_1)
#         X_2 = F.relu(self.conv2(X_1, edge_index))
#         X_2 = self.dropout(X_2)   
#         X_2 = self.bn2(X_2)
#         X_3 = F.relu(self.conv3(X_2, edge_index)) 
#         X_3 = self.dropout(X_3)
#         X_3 = self.bn3(X_3)
#         X_4 = F.relu(self.conv4(X_3, edge_index))
#         X_4 = self.dropout(X_4)
#         X_4 = self.bn4(X_4)
#         X_5 = F.relu(self.conv5(X_4, edge_index)) 
#         X_5 = self.dropout(X_5)
#         X_5 = self.bn5(X_5)

#         x = global_mean_pool(X_5, batch)
#         x = self.linear(x)
        
#         return x

# class simple_GAT(torch.nn.Module): 
#     def __init__(self, hidden_channels, n_classes, dropout_prob, input_channel, num_heads=1):
#         super(simple_GAT, self).__init__()

#         self.conv1 = GATConv(input_channel, hidden_channels, heads=num_heads)
#         self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
#         self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
#         self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
#         self.conv5 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)

#         self.bn1 = BatchNorm1d(hidden_channels * num_heads)
#         self.bn2 = BatchNorm1d(hidden_channels * num_heads)
#         self.bn3 = BatchNorm1d(hidden_channels * num_heads)
#         self.bn4 = BatchNorm1d(hidden_channels * num_heads)
#         self.bn5 = BatchNorm1d(hidden_channels * num_heads)
#         self.dropout = torch.nn.Dropout(dropout_prob)
#         self.linear = Linear(hidden_channels * num_heads, n_classes)

#     def forward(self, data):
#         X, edge_index = data.x, data.edge_index
#         batch = data.batch if hasattr(data, 'batch') else torch.zeros(X.size(0), dtype=torch.long, device=X.device)

#         X_1 = F.relu(self.conv1(X, edge_index))
#         X_1 = self.bn1(X_1)
#         X_2 = F.relu(self.conv2(X_1, edge_index))
#         X_2 = self.dropout(X_2) 
#         X_2 = self.bn2(X_2)
#         X_3 = F.relu(self.conv3(X_2, edge_index))
#         X_3 = self.dropout(X_3)
#         X_3 = self.bn3(X_3) 
#         X_4 = F.relu(self.conv4(X_3, edge_index))  
#         X_4 = self.dropout(X_4) 
#         X_4 = self.bn4(X_4) 
#         X_5 = F.relu(self.conv5(X_4, edge_index))
#         X_5 = self.dropout(X_5)  
#         X_5 = self.bn5(X_5)
        
#         x = global_mean_pool(X_5, batch)
#         x = self.linear(x)
        
#         return x 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GATv2Conv, GINConv,
    ResGatedGraphConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch_geometric.utils import softmax
from torch_geometric.explain import Explainer, GNNExplainer
from torch.nn import BatchNorm1d, Linear


class MethaneGNN(nn.Module):
    """
    Enhanced GNN model for methane parameter regression with:
     - Choice of GCN/GAT/GATv2/GIN
     - Residual graph conv layers + BatchNorm + ReLU
     - Attention-based graph pooling
     - Deep MLP regression head
     - Optional aleatoric uncertainty estimation
    """
    def __init__(self,
                 num_node_features=1,
                 hidden_dim=64,
                 num_targets=2,
                 dropout_rate=0.3,
                 num_layers=4,
                 model_type='gat',
                 estimate_uncertainty=False):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.estimate_uncertainty = estimate_uncertainty

        # Build graph conv layers
        self.conv_layers = nn.ModuleList()
        if model_type == 'gcn':
            self.conv_layers.append(GCNConv(num_node_features, hidden_dim))
            for _ in range(num_layers - 1):
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        elif model_type == 'gat':
            self.conv_layers.append(
                GATConv(num_node_features, hidden_dim // 8, heads=8, dropout=dropout_rate)
            )
            for _ in range(num_layers - 2):
                self.conv_layers.append(
                    GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout_rate)
                )
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
            )

        elif model_type == 'gatv2':
            self.conv_layers.append(
                GATv2Conv(num_node_features, hidden_dim // 8, heads=8, dropout=dropout_rate)
            )
            for _ in range(num_layers - 2):
                self.conv_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout_rate)
                )
            self.conv_layers.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
            )

        elif model_type == 'gin':
            mlp = nn.Sequential(
                Linear(num_node_features, hidden_dim),
                BatchNorm1d(hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.conv_layers.append(GINConv(mlp))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    Linear(hidden_dim, hidden_dim)
                )
                self.conv_layers.append(GINConv(mlp))

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # BatchNorm after each conv
        self.batch_norms = nn.ModuleList([
            BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Attention pooling network
        self.att_pool = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            Linear(hidden_dim // 2, 1)
        )

        # Deep MLP regression head
        self.reg_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            Linear(hidden_dim // 2, num_targets)
        )

        # Aleatoric uncertainty head
        if self.estimate_uncertainty:
            self.uncertainty_head = nn.Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                Linear(hidden_dim // 2, num_targets)
            )

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        """Forward pass through the network"""
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # 1) Graph conv layers with residuals
        last_x = x
        for i, conv in enumerate(self.conv_layers):
            if isinstance(conv, GCNConv):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            if i > 0 and x.shape == last_x.shape:
                x = x + last_x
            if i < len(self.conv_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)
            last_x = x

        # 2) Attention pooling - Fix softmax issue
        scores = self.att_pool(x).squeeze(-1)      # [num_nodes]
        
        # Handle the case where batch might not be properly specified
        if batch is None or torch.all(batch == 0):
            # If all nodes belong to the same graph or batch is None, use standard softmax
            weights = F.softmax(scores, dim=0)
            x = x * weights.view(-1, 1)
        else:
            # Otherwise use PyG's softmax with proper index
            try:
                alpha = softmax(scores, index=batch)  # [num_nodes]
                x = x * alpha.view(-1, 1)            # weight node feats
            except Exception as e:
                # Fallback to standard softmax per graph in case of errors
                weights = torch.zeros_like(scores)
                for b in torch.unique(batch):
                    batch_mask = (batch == b)
                    batch_scores = scores[batch_mask]
                    batch_weights = F.softmax(batch_scores, dim=0)
                    weights[batch_mask] = batch_weights
                x = x * weights.view(-1, 1)
        
        graph_emb = global_add_pool(x, batch)     # [batch_size, hidden_dim]

        # 3) Regression head
        preds = self.reg_head(graph_emb)          # [batch_size, num_targets]

        # 4) Handle return values based on mode
        if self.training:
            # During training
            if self.estimate_uncertainty:
                # Return 3-tuple with (predictions, log_variance, embeddings)
                log_var = self.uncertainty_head(graph_emb)
                return preds, log_var, graph_emb
            else:
                # Return 2-tuple with (predictions, embeddings)
                return preds, graph_emb
        else:
            # During evaluation/inference, just return predictions for compatibility
            return preds

    def get_embeddings(self, data):
        """Return node embeddings and graph embedding."""
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        last_x = x
        for i, conv in enumerate(self.conv_layers):
            if isinstance(conv, GCNConv):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            if i > 0 and x.shape == last_x.shape:
                x = x + last_x
            last_x = x

        graph_emb = global_add_pool(x, batch)
        return x, graph_emb


class MethaneGNNExplainer:
    """Wrapper for GNNExplainer to explain predictions of the methane GNN model"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Create a proper model wrapper class
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.training = model.training
                
            def forward(self, x, edge_index, **kwargs):
                batch = kwargs.get('batch', None)
                # Create a dummy data object with the expected attributes
                from torch_geometric.data import Data
                data = Data(x=x, edge_index=edge_index, batch=batch)
                if 'edge_weight' in kwargs:
                    data.edge_weight = kwargs['edge_weight']
                
                # Run the model with the data object
                result = self.model(data)
                
                # Handle tuple returns (predictions, embeddings) from models like simple_GCN_res
                if isinstance(result, tuple):
                    return result[0]  # Return only the predictions
                
                return result
        
        # Initialize GNNExplainer with wrapper class
        self.wrapped_model = ModelWrapper(model)
        self.explainer = Explainer(
            model=self.wrapped_model,
            algorithm=GNNExplainer(epochs=300),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='graph',
                return_type='raw',
            ),
            threshold_config=dict(
                threshold_type='topk',
                value=20
            )
        )
    
    def explain_graph(self, graph, node_names=None, save_path=None, target_idx=0):
        """
        Explain prediction for a single graph
        
        Args:
            graph: Graph to explain
            node_names: List of node names
            save_path: Path to save the edge importance matrix
            target_idx: Index of the target to explain (0 for ACE-km, 1 for H2-km)
            
        Returns:
            edge_importance_matrix: Matrix of edge importance scores
            explanation: Raw explanation object
        """
        graph = graph.to(self.device)
        self.model.eval()
        
        # Get model prediction
        with torch.no_grad():
            # First, make sure we're in eval mode to avoid uncertainty output
            if hasattr(self.model, 'estimate_uncertainty'):
                self.model.estimate_uncertainty = False
            
            prediction = self.model(graph)
            
            # Handle tuple output (pred, embedding) from models like simple_GCN_res
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            
            # Handle 1D tensor outputs (models that return squeezed predictions)
            if len(prediction.shape) == 1:
                # For single target models that return [batch_size] tensor
                predicted_value = prediction[0].item()
                
                # If target_idx is not 0 but we have a 1D tensor, we need to handle this case
                if target_idx > 0:
                    print(f"Warning: Model returns single value but target_idx={target_idx} was requested.")
                    print(f"Using available prediction at index 0 instead.")
            else:
                # For multi-target models that return [batch_size, num_targets]
                predicted_value = prediction[0, target_idx].item()
            
            # Extract the actual value - handle different target formats
            if len(graph.y.shape) == 1:  # [batch_size] format
                actual_value = graph.y[0].item()
            elif graph.y.shape[1] == 1:  # [batch_size, 1] format
                actual_value = graph.y[0, 0].item()
            else:  # [batch_size, num_targets] format
                actual_value = graph.y[0, target_idx].item()
        
        print(f"Target: {target_idx}, Actual value: {actual_value:.4f}, Predicted value: {predicted_value:.4f}")
        
        # Create zeros tensor for batch
        batch = torch.zeros(len(graph.x), dtype=torch.long, device=self.device)
        
        # Pass edge_weight if available
        kwargs = {'batch': batch}
        if hasattr(graph, 'edge_weight'):
            kwargs['edge_weight'] = graph.edge_weight
        
        # Get explanation
        try:
            explanation = self.explainer(
                graph.x, 
                graph.edge_index,
                **kwargs
            )
        except NotImplementedError as e:
            # For models using softmax with index parameter,
            # we need to wrap them differently
            print("Adapting explainer for models using softmax with index parameter...")
            
            # Create a simpler wrapper that avoids the softmax issue
            class SimpleModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x, edge_index, **kwargs):
                    # Extract batch tensor
                    batch = kwargs.get('batch', None)
                    
                    # Create simplified view of the model that avoids the softmax issue
                    class SimplifiedModel(torch.nn.Module):
                        def __init__(self, original_model):
                            super().__init__()
                            self.original_model = original_model
                            # Copy necessary components
                            self.conv_layers = original_model.conv_layers if hasattr(original_model, 'conv_layers') else None
                            self.batch_norms = original_model.batch_norms if hasattr(original_model, 'batch_norms') else None
                            self.reg_head = original_model.reg_head if hasattr(original_model, 'reg_head') else None
                        
                        def forward(self, x, edge_index, batch=None):
                            # If this is a model like simple_GCN_res that doesn't have these attributes
                            if self.conv_layers is None:
                                # Just call the original model but handle tuple returns
                                from torch_geometric.data import Data
                                data = Data(x=x, edge_index=edge_index, batch=batch)
                                result = self.original_model(data)
                                # If the model returns a tuple, just return the first item (prediction)
                                if isinstance(result, tuple):
                                    return result[0]
                                return result
                            
                            # Original implementation for MethaneGNN model
                            if batch is None:
                                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                            
                            # Apply convolutions
                            last_x = x
                            for i, conv in enumerate(self.conv_layers):
                                if isinstance(conv, GCNConv):
                                    x = conv(x, edge_index)  # Skip edge_weight for simplicity
                                else:
                                    x = conv(x, edge_index)
                                x = self.batch_norms[i](x)
                                x = F.relu(x)
                                if i > 0 and x.shape == last_x.shape:
                                    x = x + last_x
                                last_x = x
                            
                            # Simple global pooling instead of attention
                            g = global_mean_pool(x, batch)
                            
                            # Apply regression head
                            return self.reg_head(g)
                    
                    # Create and use the simplified model
                    simplified = SimplifiedModel(self.model)
                    
                    # Create data object
                    from torch_geometric.data import Data
                    data = Data(x=x, edge_index=edge_index, batch=batch)
                    
                    # Run simplified model
                    return simplified(x, edge_index, batch)
            
            # Recreate explainer with the simpler wrapper
            simplified_wrapper = SimpleModelWrapper(self.model)
            simplified_explainer = Explainer(
                model=simplified_wrapper,
                algorithm=GNNExplainer(epochs=300),
                explanation_type='model',
                node_mask_type=None,
                edge_mask_type='object',
                model_config=dict(
                    mode='regression',
                    task_level='graph',
                    return_type='raw',
                ),
                threshold_config=dict(
                    threshold_type='topk',
                    value=20
                )
            )
            
            # Get explanation with simplified model
            explanation = simplified_explainer(
                graph.x, 
                graph.edge_index,
                **kwargs
            )
        
        # Convert to edge importance matrix
        num_nodes = graph.x.size(0)
        edge_importance_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        for i, (u, v) in enumerate(zip(graph.edge_index[0], graph.edge_index[1])):
            edge_importance_matrix[u, v] = explanation.edge_mask[i]
        
        # Make matrix symmetric for undirected graph
        edge_importance_matrix = (edge_importance_matrix + edge_importance_matrix.t()) / 2
        
        # Save edge importance matrix if requested
        if save_path is not None:
            import numpy as np
            np.savetxt(save_path, edge_importance_matrix.cpu().numpy(), delimiter=',')
            print(f"Saved edge importance matrix to {save_path}")
        
        return edge_importance_matrix, explanation

class simple_GCN_res(nn.Module):
    def __init__(self, hidden_channels, n_targets, dropout_prob, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout_prob)
        
        # 2 Layer MLP head for regression
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x1 = F.relu(self.conv1(x, edge_index)); x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index)); x2 = self.dropout(x2); x2 = self.bn2(x2) + x1
        x3 = F.relu(self.conv3(x2, edge_index)); x3 = self.dropout(x3); x3 = self.bn3(x3) + x2
        x4 = F.relu(self.conv4(x3, edge_index)); x4 = self.dropout(x4); x4 = self.bn4(x4) + x3
        x5 = F.relu(self.conv5(x4, edge_index)); x5 = self.dropout(x5); x5 = self.bn5(x5) + x4

        g = global_mean_pool(x5, batch)
        
        # regression output: a single real value per graph
        out = self.reg_head(g).squeeze(-1)
        
        # In training mode, return both prediction and embedding
        # In eval mode, return just prediction for compatibility with explainers
        if self.training:
            return out, g
        else:
            return out


class simple_RGGC(nn.Module):
    def __init__(self, hidden_channels, n_targets, dropout_prob, in_channels):
        super().__init__()
        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels)
        self.conv2 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv3 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv4 = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.conv5 = ResGatedGraphConv(hidden_channels, hidden_channels)

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout_prob)
        
        # 2 Layer MLP head for regression
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x1 = F.relu(self.conv1(x, edge_index)); x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index)); x2 = self.dropout(x2); x2 = self.bn2(x2) + x1
        x3 = F.relu(self.conv3(x2, edge_index)); x3 = self.dropout(x3); x3 = self.bn3(x3) + x2
        x4 = F.relu(self.conv4(x3, edge_index)); x4 = self.dropout(x4); x4 = self.bn4(x4) + x3
        x5 = F.relu(self.conv5(x4, edge_index)); x5 = self.dropout(x5); x5 = self.bn5(x5) + x4

        g = global_mean_pool(x5, batch)
        
        # regression output: a single real value per graph
        out = self.reg_head(g).squeeze(-1)
        
        # In training mode, return both prediction and embedding
        # In eval mode, return just prediction for compatibility with explainers
        if self.training:
            return out, g
        else:
            return out


class simple_GAT(nn.Module):
    def __init__(self, hidden_channels, n_targets, dropout_prob, in_channels, num_heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv5 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)

        self.bn1 = BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = BatchNorm1d(hidden_channels * num_heads)
        self.bn3 = BatchNorm1d(hidden_channels * num_heads)
        self.bn4 = BatchNorm1d(hidden_channels * num_heads)
        self.bn5 = BatchNorm1d(hidden_channels * num_heads)

        self.dropout = nn.Dropout(dropout_prob)
        
        # Linear projection for residual connections (needed because of dimension mismatch with GAT)
        self.proj2 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.proj3 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.proj4 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.proj5 = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        
        # 2 Layer MLP head for regression
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels // 3, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x1 = F.relu(self.conv1(x, edge_index)); x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index)); x2 = self.dropout(x2); x2 = self.bn2(x2) + self.proj2(x1)
        x3 = F.relu(self.conv3(x2, edge_index)); x3 = self.dropout(x3); x3 = self.bn3(x3) + self.proj3(x2)
        x4 = F.relu(self.conv4(x3, edge_index)); x4 = self.dropout(x4); x4 = self.bn4(x4) + self.proj4(x3)
        x5 = F.relu(self.conv5(x4, edge_index)); x5 = self.dropout(x5); x5 = self.bn5(x5) + self.proj5(x4)

        g = global_mean_pool(x5, batch)
        
        # regression output: a single real value per graph
        out = self.reg_head(g).squeeze(-1)
        
        # In training mode, return both prediction and embedding
        # In eval mode, return just prediction for compatibility with explainers
        if self.training:
            return out, g
        else:
            return out
