import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import numpy as np


class AdaptiveGraphLearner(nn.Module):
    """
    Learn adaptive graph structure from microbial abundance patterns
    
    This module learns which microbes should be connected based on:
    - Abundance correlation patterns
    - Biological interaction potential
    - Sample-specific conditions
    """
    
    def __init__(self, num_nodes, hidden_dim=64, sparsity_factor=0.1):
        super(AdaptiveGraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.sparsity_factor = sparsity_factor
        
        # Node embedding layers
        self.node_embedding = nn.Linear(1, hidden_dim)  # Input is abundance (1D)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Graph learning layers
        self.graph_learner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temperature parameter for Gumbel softmax (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, batch):
        """
        Learn adaptive graph structure for each sample
        
        Args:
            x: Node features [num_nodes, 1] (microbial abundances)
            batch: Batch assignment for nodes
            
        Returns:
            edge_index: Learned edge indices
            edge_weights: Learned edge weights
            adjacency_matrix: Dense adjacency matrix for interpretability
        """
        batch_size = batch.max().item() + 1
        
        # Embed node features
        node_emb = F.relu(self.node_embedding(x))  # [num_nodes, hidden_dim]
        node_emb = self.node_projection(node_emb)   # [num_nodes, hidden_dim]
        
        # Learn graph structure for each sample in batch
        all_edge_indices = []
        all_edge_weights = []
        all_adj_matrices = []
        
        for sample_idx in range(batch_size):
            # Get nodes for this sample
            sample_mask = (batch == sample_idx)
            sample_nodes = node_emb[sample_mask]  # [num_sample_nodes, hidden_dim]
            num_sample_nodes = sample_nodes.size(0)
            
            # Create pairwise combinations
            node_i = sample_nodes.unsqueeze(1).repeat(1, num_sample_nodes, 1)  # [N, N, hidden_dim]
            node_j = sample_nodes.unsqueeze(0).repeat(num_sample_nodes, 1, 1)  # [N, N, hidden_dim]
            
            # Concatenate node pairs
            node_pairs = torch.cat([node_i, node_j], dim=-1)  # [N, N, hidden_dim*2]
            
            # Learn edge probabilities
            edge_probs = self.graph_learner(node_pairs).squeeze(-1)  # [N, N]
            
            # Make symmetric (undirected graph)
            edge_probs = (edge_probs + edge_probs.T) / 2
            
            # Remove self-loops
            eye = torch.eye(num_sample_nodes, device=edge_probs.device)
            edge_probs = edge_probs * (1 - eye)
            
            # Apply sparsity: keep top-k edges
            num_edges_to_keep = int(self.sparsity_factor * num_sample_nodes * (num_sample_nodes - 1))
            
            # Get top-k edges
            edge_probs_flat = edge_probs.view(-1)
            _, top_indices = torch.topk(edge_probs_flat, k=min(num_edges_to_keep, edge_probs_flat.size(0)))
            
            # Create sparse adjacency matrix
            sparse_adj = torch.zeros_like(edge_probs_flat)
            sparse_adj[top_indices] = edge_probs_flat[top_indices]
            sparse_adj = sparse_adj.view(num_sample_nodes, num_sample_nodes)
            
            # Convert to edge_index and edge_weights
            sample_edge_index, sample_edge_weights = dense_to_sparse(sparse_adj)
            
            # Adjust indices for global indexing
            node_offset = sample_idx * self.num_nodes
            sample_edge_index = sample_edge_index + node_offset
            
            all_edge_indices.append(sample_edge_index)
            all_edge_weights.append(sample_edge_weights)
            all_adj_matrices.append(sparse_adj)
        
        # Concatenate all edges
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
            edge_weights = torch.cat(all_edge_weights, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_weights = torch.empty((0,), dtype=torch.float, device=x.device)
        
        return edge_index, edge_weights, all_adj_matrices


class DynamicEdgeAttention(nn.Module):
    """
    Compute dynamic edge weights based on microbial abundance patterns
    """
    
    def __init__(self, hidden_dim, num_heads=4):
        super(DynamicEdgeAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # Attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge attention
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index):
        """
        Compute dynamic edge weights
        
        Args:
            node_features: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            
        Returns:
            edge_weights: [num_edges]
        """
        row, col = edge_index
        
        # Get source and target node features
        source_features = node_features[row]  # [num_edges, hidden_dim]
        target_features = node_features[col]  # [num_edges, hidden_dim]
        
        # Concatenate for edge attention
        edge_features = torch.cat([source_features, target_features], dim=-1)  # [num_edges, hidden_dim*2]
        
        # Compute edge weights
        edge_weights = self.edge_attention(edge_features).squeeze(-1)  # [num_edges]
        
        return edge_weights


class AdaptiveMicrobialGNN(nn.Module):
    """
    Non-temporal Dynamic GNN for microbial interaction learning
    
    Key Features:
    - Learns graph structure from microbial abundance data
    - Sample-specific edge weights
    - Attention-based interaction discovery
    - Biologically interpretable connections
    """
    
    def __init__(self, 
                 num_nodes, 
                 input_dim=1, 
                 hidden_dim=64, 
                 output_dim=1,
                 num_heads=4,
                 dropout=0.2,
                 sparsity_factor=0.1,
                 num_gnn_layers=3):
        super(AdaptiveMicrobialGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Adaptive graph learner
        self.graph_learner = AdaptiveGraphLearner(
            num_nodes=num_nodes, 
            hidden_dim=hidden_dim,
            sparsity_factor=sparsity_factor
        )
        
        # Dynamic edge attention
        self.edge_attention = DynamicEdgeAttention(hidden_dim, num_heads)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers (GAT-based for attention mechanisms)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True if i < num_gnn_layers - 1 else False
                )
            )
        
        # Final layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers for regression
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # For embedding extraction (like plus models)
        self.embedding_dim = hidden_dim
        
    def forward(self, x, edge_index=None, batch=None, return_dynamics=False):
        """
        Forward pass with adaptive graph learning
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Optional initial edge index (will be replaced by learned)
            batch: Batch assignment
            return_dynamics: Whether to return learned dynamics
            
        Returns:
            If return_dynamics=False: (predictions, embeddings)
            If return_dynamics=True: (predictions, dynamics_dict)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Project input features
        h = self.input_projection(x)  # [num_nodes, hidden_dim]
        
        # Learn adaptive graph structure
        learned_edge_index, learned_edge_weights, adj_matrices = self.graph_learner(x, batch)
        
        # Apply dynamic edge attention
        if learned_edge_index.size(1) > 0:  # Check if edges exist
            dynamic_edge_weights = self.edge_attention(h, learned_edge_index)
            # Combine learned and dynamic weights
            final_edge_weights = learned_edge_weights * dynamic_edge_weights
        else:
            final_edge_weights = learned_edge_weights
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            if learned_edge_index.size(1) > 0:
                h = gnn_layer(h, learned_edge_index, edge_attr=final_edge_weights)
            else:
                h = gnn_layer(h, learned_edge_index)
            
            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
                h = self.dropout_layer(h)
        
        # Normalize
        h = self.norm(h)
        
        # Global pooling for graph-level predictions
        graph_embeddings = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
        
        # Make predictions
        predictions = self.regressor(graph_embeddings)  # [batch_size, output_dim]
        
        if return_dynamics:
            dynamics = {
                'learned_adj_matrices': adj_matrices,
                'learned_edge_index': learned_edge_index,
                'learned_edge_weights': learned_edge_weights,
                'dynamic_edge_weights': dynamic_edge_weights if learned_edge_index.size(1) > 0 else None,
                'final_edge_weights': final_edge_weights,
                'node_embeddings': h,
                'graph_embeddings': graph_embeddings
            }
            return predictions, dynamics
        else:
            # Return in format compatible with mixed_embedding_pipeline
            return predictions, graph_embeddings  # (predictions, embeddings)
    
    def get_learned_graph(self, x, batch=None):
        """
        Extract learned graph structure for analysis
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            edge_index, edge_weights, adj_matrices = self.graph_learner(x, batch)
        
        return {
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'adjacency_matrices': adj_matrices
        }


class BiologicalConstraints:
    """
    Helper class for biological constraints and interpretability
    """
    
    @staticmethod
    def apply_biological_priors(edge_probs, microbial_families, known_interactions=None):
        """
        Apply biological knowledge to guide graph learning
        
        Args:
            edge_probs: Learned edge probabilities
            microbial_families: List of microbial family names
            known_interactions: Known biological interactions (optional)
        """
        # This can be extended with biological knowledge
        # For now, we just ensure reasonable sparsity
        return edge_probs
    
    @staticmethod
    def compute_biological_metrics(adj_matrix, microbial_families):
        """
        Compute biological interpretability metrics
        """
        # Degree distribution
        degrees = torch.sum(adj_matrix > 0, dim=1)
        
        # Clustering coefficient (simplified)
        clustering = torch.zeros_like(degrees, dtype=torch.float)
        
        # Community structure (can be extended)
        metrics = {
            'degree_distribution': degrees,
            'clustering_coefficients': clustering,
            'edge_density': torch.sum(adj_matrix > 0) / (adj_matrix.size(0) ** 2),
            'max_degree': torch.max(degrees),
            'mean_degree': torch.mean(degrees.float())
        }
        
        return metrics 