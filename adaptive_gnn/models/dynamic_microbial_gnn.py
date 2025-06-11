import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dense_to_sparse, to_dense_adj, softmax
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class SimpleDynamicMicrobialGNN(nn.Module):
    """
    Simplified Dynamic GNN for Microbial Networks - Optimized for Small Datasets
    
    Key simplifications:
    - Uses existing graph structure instead of creating new one
    - Much smaller parameter count
    - Simple dynamic edge weighting
    - Maintains biological interpretability
    - Supports variable complexity for different targets
    """
    
    def __init__(self, 
                 num_nodes, 
                 input_dim=1,
                 hidden_dim=32,  # Variable now
                 output_dim=1,
                 dropout=0.2,
                 num_gnn_layers=2):  # Variable now
        super(SimpleDynamicMicrobialGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        
        # Adaptive node embedding based on complexity
        if hidden_dim > 32:
            # More sophisticated embedding for challenging targets
            self.node_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # Simple embedding for easy targets
            self.node_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Enhanced edge weight predictor for challenging targets
        if hidden_dim > 32:
            self.edge_weight_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Less dropout for edge weights
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            # Simple edge weight predictor
            self.edge_weight_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Adaptive GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization for deeper networks
        if num_gnn_layers > 2:
            self.layer_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_gnn_layers)
            ])
        else:
            self.layer_norms = None
        
        # Enhanced prediction head for challenging targets
        if hidden_dim > 32:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 4, output_dim)
            )
        else:
            # Simple prediction head
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass - uses provided graph structure with dynamic edge weighting
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed nodes
        if x.dim() == 1:
            x = x.view(-1, 1)
            
        # Handle batch normalization in embedding
        if self.hidden_dim > 32 and x.size(0) > 1:
            h = self.node_embedding(x)
        else:
            # Skip batch norm for small batches
            h = self.node_embedding[0](x)  # Just the linear layer
            for i in range(2, len(self.node_embedding), 2):  # Skip batch norm layers
                if i < len(self.node_embedding):
                    h = self.node_embedding[i](h)
                if i + 1 < len(self.node_embedding):
                    h = self.node_embedding[i + 1](h)
        
        # Learn dynamic edge weights if edges exist
        if edge_index.size(1) > 0:
            # Get edge features
            row, col = edge_index
            edge_features = torch.cat([h[row], h[col]], dim=1)
            
            # Predict dynamic edge weights with proper batch handling
            if edge_features.size(0) > 1 and self.hidden_dim > 32:
                edge_weights = self.edge_weight_predictor(edge_features).squeeze()
            else:
                # Skip batch norm for edge weights with small batches
                edge_weights = edge_features
                for i, layer in enumerate(self.edge_weight_predictor):
                    if not isinstance(layer, nn.BatchNorm1d):
                        edge_weights = layer(edge_weights)
                edge_weights = edge_weights.squeeze()
            
            # Ensure edge_weights is 1D and matches number of edges
            if edge_weights.dim() == 0:
                edge_weights = edge_weights.unsqueeze(0)
            
            # Apply GNN layers with dynamic weights and residual connections
            for i, gnn_layer in enumerate(self.gnn_layers):
                h_old = h
                h_new = gnn_layer(h, edge_index, edge_weight=edge_weights)
                
                # Apply batch normalization if available and batch size > 1
                if self.layer_norms is not None and h.size(0) > 1:
                    h_new = self.layer_norms[i](h_new)
                
                # Residual connection + activation
                h = F.relu(h_new + h_old) if self.num_gnn_layers > 2 else F.relu(h_new + h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            # No edges - just apply node transformations with residuals
            for i in range(self.num_gnn_layers):
                h_old = h
                h_new = F.relu(h)
                h = h_new + h_old if self.num_gnn_layers > 2 else h_new
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        batch_size = batch.max().item() + 1
        graph_embeddings = []
        
        for sample_idx in range(batch_size):
            sample_mask = (batch == sample_idx)
            sample_h = h[sample_mask]
            
            # Enhanced pooling for challenging targets
            if self.hidden_dim > 32:
                # Use both mean and max pooling
                mean_pool = sample_h.mean(dim=0, keepdim=True)
                max_pool = sample_h.max(dim=0, keepdim=True)[0]
                graph_emb = (mean_pool + max_pool) / 2
            else:
                # Simple mean pooling
                graph_emb = sample_h.mean(dim=0, keepdim=True)
            
            graph_embeddings.append(graph_emb)
        
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        
        # Prediction with proper batch handling
        if graph_embeddings.size(0) > 1 and self.hidden_dim > 32:
            output = self.predictor(graph_embeddings)
        else:
            # Skip batch norm layers for small batches
            output = graph_embeddings
            for layer in self.predictor:
                if not isinstance(layer, nn.BatchNorm1d):
                    output = layer(output)
        
        return output
    
    def get_embeddings(self, x, edge_index, batch=None):
        """Extract embeddings for downstream ML models"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            # Embed nodes
            h = self.node_embedding(x)
            
            # Apply GNN processing
            if edge_index.size(1) > 0:
                row, col = edge_index
                edge_features = torch.cat([h[row], h[col]], dim=1)
                edge_weights = self.edge_weight_predictor(edge_features).squeeze()
                
                # Ensure edge_weights is 1D and matches number of edges
                if edge_weights.dim() == 0:
                    edge_weights = edge_weights.unsqueeze(0)
                
                for gnn_layer in self.gnn_layers:
                    h_new = gnn_layer(h, edge_index, edge_weight=edge_weights)
                    h = F.relu(h_new) + h
            
            # Graph-level embeddings
            batch_size = batch.max().item() + 1
            graph_embeddings = []
            
            for sample_idx in range(batch_size):
                sample_mask = (batch == sample_idx)
                sample_h = h[sample_mask]
                graph_emb = sample_h.mean(dim=0, keepdim=True)
                graph_embeddings.append(graph_emb)
            
            graph_embeddings = torch.cat(graph_embeddings, dim=0)
        
        return {
            'node_embeddings': h,
            'graph_embeddings': graph_embeddings
        }


class SimpleHeterogeneousMicrobialGNN(nn.Module):
    """
    Simplified Heterogeneous GNN - Uses provided graph with relation learning
    
    Key simplifications:
    - Uses existing graph structure
    - Simple relation type prediction
    - Much smaller parameter count
    """
    
    def __init__(self, num_families, hidden_dim=32, num_relations=3):  # Reduced relations
        super(SimpleHeterogeneousMicrobialGNN, self).__init__()
        self.num_families = num_families
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        
        # Simple node embedding
        self.node_embedding = nn.Linear(1, hidden_dim)
        
        # Relation type predictor (much simpler)
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_relations),
            nn.Softmax(dim=-1)
        )
        
        # Simple RGCN layers
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(2)  # Reduced from 3
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(2)
        ])
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass - uses provided graph structure
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Embed nodes
        h = self.node_embedding(x)
        
        # Predict edge types if edges exist
        if edge_index.size(1) > 0:
            row, col = edge_index
            edge_features = torch.cat([h[row], h[col]], dim=1)
            edge_type_probs = self.relation_predictor(edge_features)
            
            # Use most likely edge type
            edge_types = edge_type_probs.argmax(dim=1)
            
            # Apply RGCN layers
            for rgcn, norm in zip(self.rgcn_layers, self.layer_norms):
                h_new = rgcn(h, edge_index, edge_types)
                h_new = norm(h_new)
                h = F.relu(h_new) + h  # Residual connection
        else:
            # No edges - just apply linear layers
            for norm in self.layer_norms:
                h_new = norm(F.relu(self.node_embedding(h)))
                h = h_new + h
        
        return h
    
    def get_embeddings(self, x, edge_index, batch=None):
        """Extract embeddings for downstream ML models"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            h = self.forward(x, edge_index, batch)
            
            # Graph-level embeddings
            batch_size = batch.max().item() + 1
            graph_embeddings = []
            
            for sample_idx in range(batch_size):
                sample_mask = (batch == sample_idx)
                sample_h = h[sample_mask]
                graph_emb = sample_h.mean(dim=0, keepdim=True)
                graph_embeddings.append(graph_emb)
            
            graph_embeddings = torch.cat(graph_embeddings, dim=0)
        
        return {
            'node_embeddings': h,
            'graph_embeddings': graph_embeddings
        }


# Keep the old complex models for reference but add aliases for the simple ones
DynamicMicrobialGNN = SimpleDynamicMicrobialGNN
HeterogeneousMicrobialGNN = SimpleHeterogeneousMicrobialGNN


# Original complex models (renamed for backup)
class ComplexDynamicMicrobialGNN(nn.Module):
    """
    Complex Dynamic GNN - TOO COMPLEX FOR SMALL DATASETS
    """
    pass  # Removed complex implementation


class ComplexHeterogeneousMicrobialGNN(nn.Module):
    """
    Complex Heterogeneous GNN - TOO COMPLEX FOR SMALL DATASETS  
    """
    pass  # Removed complex implementation


class BiologicalConstraintLoss(nn.Module):
    """
    Simplified biological constraint loss
    """
    
    def __init__(self, alpha=0.01, beta=0.01):  # Reduced regularization
        super(BiologicalConstraintLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, edge_weights, interaction_matrices=None, microbial_metadata=None):
        """
        Simple sparsity loss
        """
        if edge_weights is not None and len(edge_weights) > 0:
            sparsity_loss = torch.mean(edge_weights)
        else:
            sparsity_loss = torch.tensor(0.0, device=edge_weights.device if edge_weights is not None else 'cpu')
        
        total_loss = self.alpha * sparsity_loss
        
        return total_loss, {
            'sparsity_loss': sparsity_loss.item()
        } 