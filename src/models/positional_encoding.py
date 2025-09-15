#!/usr/bin/env python3
"""
Positional Encoding for Graph Neural Networks

This module provides various positional encoding methods for Graph Neural Networks,
specifically designed for Graph Transformer architectures.

Mathematical Foundation:
======================
Graph positional encoding methods:

1. Laplacian Positional Encoding (LPE):
   Uses eigenvalues and eigenvectors of the graph Laplacian matrix
   L = D - A where D is degree matrix, A is adjacency matrix
   
2. Sinusoidal Positional Encoding:
   Adapted from Transformer architecture for sequences
   PE(pos,2i) = sin(pos/10000^(2i/d_model))
   PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
   
3. Learned Positional Encoding:
   Trainable embeddings for each node position

Authors: Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple
import warnings


class PositionalEncoding(nn.Module):
    """
    Universal positional encoding for Graph Neural Networks
    
    Supports multiple encoding types:
    - 'laplacian': Uses graph Laplacian eigenvectors (most suitable for graphs)
    - 'sinusoidal': Adapted Transformer-style positional encoding  
    - 'learned': Trainable position embeddings
    - 'random_walk': Random walk statistics-based encoding
    """
    
    def __init__(self, 
                 d_model: int, 
                 encoding_type: str = 'laplacian',
                 max_nodes: int = 1000,
                 dropout: float = 0.0):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimensionality
            encoding_type: Type of positional encoding ('laplacian', 'sinusoidal', 'learned', 'random_walk')
            max_nodes: Maximum number of nodes (for learned/sinusoidal encoding)
            dropout: Dropout probability for positional encodings
        """
        super().__init__()
        
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.max_nodes = max_nodes
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        if encoding_type == 'sinusoidal':
            self.pe = self._create_sinusoidal_encoding(max_nodes, d_model)
        elif encoding_type == 'learned':
            self.pe = nn.Embedding(max_nodes, d_model)
            nn.init.normal_(self.pe.weight, 0, 0.1)
        elif encoding_type == 'laplacian':
            # Laplacian PE computed dynamically based on graph
            self.register_buffer('pe', None)
            self.k_eigenvectors = min(16, d_model)  # Number of eigenvectors to use
        elif encoding_type == 'random_walk':
            # Random walk PE computed dynamically
            self.register_buffer('pe', None)
            self.rw_steps = 16  # Number of random walk steps
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """
        Create sinusoidal positional encoding
        
        Mathematical Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create frequency terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (handle odd d_model)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _compute_laplacian_encoding(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute Laplacian positional encoding using graph structure
        
        Mathematical Foundation:
        1. Compute graph Laplacian: L = D - A
        2. Eigendecomposition: L = V Λ V^T  
        3. Use k smallest non-zero eigenvalues and corresponding eigenvectors
        4. Project to desired dimension if needed
        
        Args:
            edge_index: Graph connectivity in COO format [2, num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            pe: Positional encoding [num_nodes, d_model]
        """
        try:
            from torch_geometric.utils import get_laplacian, to_dense_adj
        except ImportError:
            raise ImportError("torch_geometric required for Laplacian positional encoding")
        
        device = edge_index.device
        
        # Compute normalized Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, num_nodes=num_nodes, normalization='sym'
        )
        
        # Convert to dense matrix for eigendecomposition
        L = torch.sparse_coo_tensor(
            edge_index_lap, edge_weight_lap, (num_nodes, num_nodes), device=device
        ).to_dense()
        
        # Handle disconnected components by adding small diagonal
        L = L + 1e-8 * torch.eye(num_nodes, device=device)
        
        # Eigendecomposition
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(L)
        except RuntimeError:
            # Fallback for numerical issues
            eigenvals, eigenvecs = torch.linalg.eig(L)
            eigenvals = eigenvals.real
            eigenvecs = eigenvecs.real
        
        # Sort eigenvalues and eigenvectors
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Use k smallest non-zero eigenvalues (skip the first one which is 0)
        start_idx = 1 if eigenvals[0] < 1e-6 else 0  # Skip zero eigenvalue
        end_idx = min(start_idx + self.k_eigenvectors, num_nodes)
        
        pe = eigenvecs[:, start_idx:end_idx]  # Shape: [num_nodes, k]
        
        # Pad or project to desired dimension
        if pe.size(1) < self.d_model:
            # Pad with zeros if we have fewer eigenvectors than needed
            padding = torch.zeros(num_nodes, self.d_model - pe.size(1), device=device)
            pe = torch.cat([pe, padding], dim=1)
        elif pe.size(1) > self.d_model:
            # Use linear projection if we have more eigenvectors than needed
            if not hasattr(self, 'lap_projection'):
                self.lap_projection = nn.Linear(pe.size(1), self.d_model).to(device)
            pe = self.lap_projection(pe)
        
        return pe
    
    def _compute_random_walk_encoding(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute random walk-based positional encoding
        
        Uses random walk return probabilities as positional features
        """
        try:
            from torch_geometric.utils import to_dense_adj
        except ImportError:
            raise ImportError("torch_geometric required for random walk encoding")
        
        device = edge_index.device
        
        # Convert to adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        
        # Normalize to get transition matrix
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        P = adj / degree
        
        # Compute powers of transition matrix for multi-step random walks
        pe_list = []
        P_k = torch.eye(num_nodes, device=device)  # P^0 = I
        
        for k in range(self.rw_steps):
            # Diagonal elements represent return probabilities
            pe_list.append(P_k.diag().unsqueeze(1))
            P_k = P_k @ P  # P^{k+1}
        
        # Concatenate all return probabilities
        pe = torch.cat(pe_list, dim=1)  # Shape: [num_nodes, rw_steps]
        
        # Project to desired dimension
        if pe.size(1) != self.d_model:
            if not hasattr(self, 'rw_projection'):
                self.rw_projection = nn.Linear(pe.size(1), self.d_model).to(device)
            pe = self.rw_projection(pe)
        
        return pe
    
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply positional encoding to node features
        
        Args:
            x: Node features [num_nodes, d_input]
            edge_index: Graph connectivity [2, num_edges] (required for graph-based encodings)
            
        Returns:
            x_pe: Node features with positional encoding [num_nodes, d_model]
        """
        num_nodes = x.size(0)
        device = x.device
        
        if self.encoding_type == 'sinusoidal':
            if num_nodes > self.max_nodes:
                warnings.warn(f"Number of nodes ({num_nodes}) exceeds max_nodes ({self.max_nodes})")
                pe = self.pe[:, :num_nodes, :].to(device)
            else:
                pe = self.pe[:, :num_nodes, :].to(device)
            pe = pe.squeeze(0)  # Remove batch dimension
            
        elif self.encoding_type == 'learned':
            if num_nodes > self.max_nodes:
                raise ValueError(f"Number of nodes ({num_nodes}) exceeds max_nodes ({self.max_nodes})")
            positions = torch.arange(num_nodes, device=device)
            pe = self.pe(positions)
            
        elif self.encoding_type == 'laplacian':
            if edge_index is None:
                raise ValueError("edge_index required for Laplacian positional encoding")
            pe = self._compute_laplacian_encoding(edge_index, num_nodes)
            
        elif self.encoding_type == 'random_walk':
            if edge_index is None:
                raise ValueError("edge_index required for random walk positional encoding")
            pe = self._compute_random_walk_encoding(edge_index, num_nodes)
        
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Add positional encoding to input features
        # If input dimension doesn't match PE dimension, project input first
        if x.size(1) != pe.size(1):
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(x.size(1), pe.size(1)).to(device)
            x = self.input_projection(x)
        
        # Add positional encoding
        x_pe = x + pe
        
        # Apply dropout if specified
        if self.dropout is not None:
            x_pe = self.dropout(x_pe)
        
        return x_pe


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for Graph Transformers
    
    Implements scaled dot-product attention with multiple heads:
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store attention weights for visualization
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head self-attention
        
        Args:
            x: Input features [num_nodes, d_model]
            mask: Attention mask [num_nodes, num_nodes] (optional)
            
        Returns:
            output: Attended features [num_nodes, d_model]
            attention_weights: Attention weights [num_heads, num_nodes, num_nodes]
        """
        batch_size, seq_len = x.size(0), x.size(0)  # For graphs, batch_size = num_nodes
        
        # Linear projections
        Q = self.W_q(x)  # [num_nodes, d_model]
        K = self.W_k(x)  # [num_nodes, d_model]
        V = self.W_v(x)  # [num_nodes, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # [num_heads, num_nodes, d_k]
        K = K.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # [num_heads, num_nodes, d_k]
        V = V.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)  # [num_heads, num_nodes, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [num_heads, num_nodes, num_nodes]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)  # [num_heads, num_nodes, num_nodes]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [num_heads, num_nodes, d_k]
        
        # Concatenate heads
        context = context.transpose(0, 1).contiguous().view(seq_len, self.d_model)  # [num_nodes, d_model]
        
        # Final linear projection
        output = self.W_o(context)
        
        # Store attention weights for potential visualization
        self.attention_weights = attention_weights.detach()
        
        return output, attention_weights.mean(dim=0)  # Return average attention across heads


# Example usage and testing
def main():
    """
    Test positional encoding implementations
    """
    # Test parameters
    num_nodes = 50
    d_model = 64
    
    # Create sample graph
    edge_index = torch.randint(0, num_nodes, (2, 200))  # Random edges
    node_features = torch.randn(num_nodes, 32)  # Random node features
    
    print("Testing Positional Encoding implementations:")
    print("="*60)
    
    # Test different encoding types
    encoding_types = ['sinusoidal', 'learned', 'laplacian', 'random_walk']
    
    for enc_type in encoding_types:
        try:
            print(f"\nTesting {enc_type} encoding:")
            
            pe = PositionalEncoding(d_model, encoding_type=enc_type, max_nodes=100)
            
            if enc_type in ['laplacian', 'random_walk']:
                encoded_features = pe(node_features, edge_index)
            else:
                encoded_features = pe(node_features)
            
            print(f"  Input shape: {node_features.shape}")
            print(f"  Output shape: {encoded_features.shape}")
            print(f"  ✅ {enc_type} encoding successful!")
            
        except Exception as e:
            print(f"  ❌ {enc_type} encoding failed: {e}")
    
    # Test multi-head self-attention
    print(f"\nTesting Multi-Head Self-Attention:")
    try:
        mha = MultiHeadSelfAttention(d_model, num_heads=8)
        input_tensor = torch.randn(num_nodes, d_model)
        output, attention_weights = mha(input_tensor)
        
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention weights shape: {attention_weights.shape}")
        print(f"  ✅ Multi-Head Self-Attention successful!")
        
    except Exception as e:
        print(f"  ❌ Multi-Head Self-Attention failed: {e}")


if __name__ == "__main__":
    main()