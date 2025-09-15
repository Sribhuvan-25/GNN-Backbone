#!/usr/bin/env python3
"""
Enhanced Graph Transformer for Biological Network Analysis

This module provides an improved Graph Transformer architecture with:
- Proper positional encoding for graphs
- LayerNorm instead of BatchNorm  
- Attention visualization capabilities
- Residual connections and proper normalization
- Biological domain-specific optimizations

Mathematical Foundation:
======================
Graph Transformer Architecture:

1. Input Processing:
   X' = PositionalEncoding(X) + InputProjection(X)
   
2. Transformer Blocks (L layers):
   For each layer l:
   H_l^1 = LayerNorm(H_{l-1} + MultiHeadAttention(H_{l-1}))
   H_l^2 = LayerNorm(H_l^1 + FeedForward(H_l^1))
   
3. Graph Pooling:
   z = GlobalMeanPool(H_L)
   
4. Output Projection:
   y = OutputProjection(z)

Authors: Research Team  
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import LayerNorm as PyGLayerNorm
import math
from typing import Optional, Tuple, List, Dict
import warnings

try:
    from .positional_encoding import PositionalEncoding
except ImportError:
    from positional_encoding import PositionalEncoding


class GraphTransformerBlock(nn.Module):
    """
    Single Graph Transformer block with proper normalization and residual connections
    
    Architecture:
    1. Multi-head graph attention (TransformerConv)
    2. Add & LayerNorm
    3. Feed-forward network
    4. Add & LayerNorm
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int = 8, 
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * hidden_dim  # Standard Transformer ratio
        
        # Multi-head graph attention
        self.attention = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,  # Average heads instead of concatenating
            dropout=dropout,
            edge_dim=None,  # No edge features for now
            bias=True
        )
        
        # Layer normalization (proper for Transformers)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, self.ff_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Graph Transformer block
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Updated node features [num_nodes, hidden_dim]
            attention_weights: Attention weights if requested [num_edges, num_heads]
        """
        # Multi-head self-attention with residual connection
        residual = x
        
        if return_attention:
            attn_output, (edge_index_att, attention_weights) = self.attention(
                x, edge_index, return_attention_weights=True
            )
        else:
            attn_output = self.attention(x, edge_index)
            attention_weights = None
        
        # Add & Norm
        x = self.ln1(residual + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        residual = x
        ff_output = self.ffn(x)
        x = self.ln2(residual + self.dropout(ff_output))
        
        return x, attention_weights


class EnhancedGraphTransformer(nn.Module):
    """
    Enhanced Graph Transformer for biological network analysis
    
    Key improvements over standard Graph Transformer:
    1. Proper positional encoding for graphs
    2. LayerNorm instead of BatchNorm
    3. Flexible pooling strategies
    4. Attention visualization
    5. Biological domain optimizations
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128, 
                 output_dim: int = 1,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 positional_encoding: str = 'laplacian',
                 pooling: str = 'mean',
                 activation: str = 'relu',
                 use_skip_connections: bool = True):
        """
        Initialize Enhanced Graph Transformer
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for transformer
            output_dim: Output dimension 
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (default: 4 * hidden_dim)
            dropout: Dropout probability
            positional_encoding: Type of positional encoding ('laplacian', 'sinusoidal', 'learned', 'none')
            pooling: Graph pooling method ('mean', 'max', 'attention')
            activation: Activation function
            use_skip_connections: Whether to use skip connections between distant layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pooling = pooling
        self.use_skip_connections = use_skip_connections
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if positional_encoding != 'none':
            self.pos_encoding = PositionalEncoding(
                d_model=hidden_dim,
                encoding_type=positional_encoding,
                dropout=dropout
            )
        else:
            self.pos_encoding = None
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Skip connection projections (for layers that are far apart)
        if use_skip_connections and num_layers > 2:
            self.skip_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_layers // 2)
            ])
        else:
            self.skip_projections = None
        
        # Graph pooling
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Enhanced Graph Transformer
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Prediction [batch_size, output_dim]
            graph_repr: Graph-level representation [batch_size, hidden_dim]
            attention_weights: List of attention weights for each layer (if requested)
        """
        # Input projection
        h = self.input_projection(x)  # [num_nodes, hidden_dim]
        
        # Add positional encoding
        if self.pos_encoding is not None:
            h = self.pos_encoding(h, edge_index)
        
        # Store intermediate representations for skip connections
        layer_outputs = [h]
        attention_weights = []
        
        # Pass through transformer blocks
        for layer_idx, transformer_block in enumerate(self.transformer_blocks):
            # Regular forward pass
            h, attn_weights = transformer_block(
                h, edge_index, return_attention=return_attention
            )
            
            # Store attention weights
            if return_attention and attn_weights is not None:
                attention_weights.append(attn_weights)
            
            # Skip connections for deeper networks
            if (self.use_skip_connections and 
                self.skip_projections is not None and 
                layer_idx >= 2 and 
                (layer_idx - 2) < len(self.skip_projections)):
                
                # Connect to layer that is 2 steps back
                skip_idx = layer_idx - 2
                skip_connection = self.skip_projections[skip_idx // 2](layer_outputs[skip_idx])
                h = h + skip_connection
            
            layer_outputs.append(h)
        
        # Graph-level pooling
        if self.pooling == 'mean':
            graph_repr = global_mean_pool(h, batch)
        elif self.pooling == 'max':
            graph_repr = global_max_pool(h, batch)
        elif self.pooling == 'attention':
            # Attention-based pooling
            attention_scores = self.attention_pool(h)  # [num_nodes, 1]
            attention_weights_pool = torch.softmax(attention_scores, dim=0)
            
            # Weighted sum of node representations
            weighted_h = h * attention_weights_pool  # [num_nodes, hidden_dim]
            graph_repr = global_mean_pool(weighted_h, batch)  # This sums within each graph
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Final prediction
        output = self.output_projection(graph_repr)
        
        # Store attention weights for visualization
        if return_attention:
            self.attention_weights = attention_weights
            return output, graph_repr, attention_weights
        else:
            return output, graph_repr
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """
        Get attention maps from the last forward pass
        
        Returns:
            List of attention weight tensors for each layer
        """
        return self.attention_weights
    
    def visualize_attention(self, 
                           node_names: Optional[List[str]] = None,
                           layer_idx: int = 0,
                           head_idx: Optional[int] = None,
                           save_path: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Visualize attention patterns
        
        Args:
            node_names: Names of nodes for labeling
            layer_idx: Which transformer layer to visualize
            head_idx: Which attention head to visualize (None for average)
            save_path: Path to save visualization
            
        Returns:
            attention_matrix: Attention weights matrix
        """
        if not self.attention_weights:
            warnings.warn("No attention weights available. Run forward pass with return_attention=True first.")
            return None
        
        if layer_idx >= len(self.attention_weights):
            warnings.warn(f"Layer {layer_idx} not available. Only {len(self.attention_weights)} layers.")
            return None
        
        # Get attention weights for specified layer
        attention = self.attention_weights[layer_idx]  # [num_edges, num_heads]
        
        if head_idx is not None:
            if head_idx >= attention.size(1):
                warnings.warn(f"Head {head_idx} not available. Only {attention.size(1)} heads.")
                return None
            attention = attention[:, head_idx]  # [num_edges]
        else:
            attention = attention.mean(dim=1)  # Average across heads
        
        # Convert edge-wise attention to node-wise attention matrix
        # This requires additional information about the graph structure
        # For now, return the edge-wise attention
        
        return attention.detach().cpu()
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters
        
        Returns:
            Dictionary with parameter counts for different components
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        param_counts = {
            'input_projection': count_params(self.input_projection),
            'positional_encoding': count_params(self.pos_encoding) if self.pos_encoding else 0,
            'transformer_blocks': sum(count_params(block) for block in self.transformer_blocks),
            'skip_projections': sum(count_params(proj) for proj in self.skip_projections) if self.skip_projections else 0,
            'attention_pool': count_params(self.attention_pool) if hasattr(self, 'attention_pool') else 0,
            'output_projection': count_params(self.output_projection),
        }
        
        param_counts['total'] = sum(param_counts.values())
        
        return param_counts


def create_enhanced_graph_transformer(input_dim: int,
                                    hidden_dim: int = 128,
                                    output_dim: int = 1,
                                    num_layers: int = 4,
                                    num_heads: int = 8,
                                    dropout: float = 0.1,
                                    **kwargs) -> EnhancedGraphTransformer:
    """
    Factory function to create Enhanced Graph Transformer with sensible defaults
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension  
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        **kwargs: Additional arguments for EnhancedGraphTransformer
        
    Returns:
        Configured EnhancedGraphTransformer model
    """
    # Set defaults for parameters not provided
    defaults = {
        'positional_encoding': 'laplacian',  # Best for biological networks
        'pooling': 'mean',  # Standard graph pooling
        'activation': 'relu',
        'use_skip_connections': True
    }
    
    # Override defaults with kwargs, avoiding duplicates
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    
    return EnhancedGraphTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        **kwargs
    )


# Example usage and testing
def main():
    """
    Test Enhanced Graph Transformer implementation
    """
    print("Testing Enhanced Graph Transformer:")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    num_nodes_per_graph = [25, 30]  # Variable graph sizes
    input_dim = 16
    hidden_dim = 64
    output_dim = 1
    
    # Create test data
    x_list = []
    edge_index_list = []
    batch = []
    
    node_offset = 0
    for i, num_nodes in enumerate(num_nodes_per_graph):
        # Node features
        x_graph = torch.randn(num_nodes, input_dim)
        x_list.append(x_graph)
        
        # Edges (random graph)
        edge_index_graph = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_index_graph += node_offset  # Adjust for batch
        edge_index_list.append(edge_index_graph)
        
        # Batch assignment
        batch.extend([i] * num_nodes)
        
        node_offset += num_nodes
    
    # Concatenate batch data
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    batch = torch.tensor(batch)
    
    print(f"Test data shapes:")
    print(f"  x: {x.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  batch: {batch.shape}")
    
    # Create model
    try:
        model = create_enhanced_graph_transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        )
        
        print(f"\n✅ Model created successfully!")
        
        # Count parameters
        param_counts = model.count_parameters()
        print(f"Parameter counts:")
        for component, count in param_counts.items():
            print(f"  {component}: {count:,}")
        
        # Test forward pass
        print(f"\nTesting forward pass:")
        
        # Without attention weights
        output, graph_repr = model(x, edge_index, batch, return_attention=False)
        print(f"  Output shape: {output.shape}")
        print(f"  Graph representation shape: {graph_repr.shape}")
        
        # With attention weights
        output, graph_repr, attention_weights = model(x, edge_index, batch, return_attention=True)
        print(f"  Number of attention weight tensors: {len(attention_weights)}")
        
        if attention_weights:
            print(f"  First layer attention shape: {attention_weights[0].shape}")
        
        print(f"  ✅ Forward pass successful!")
        
        # Test different configurations
        print(f"\nTesting different configurations:")
        
        configs = [
            {'positional_encoding': 'sinusoidal'},
            {'positional_encoding': 'learned'}, 
            {'pooling': 'max'},
            {'pooling': 'attention'},
            {'activation': 'gelu'},
            {'use_skip_connections': False}
        ]
        
        for i, config in enumerate(configs):
            try:
                test_model = create_enhanced_graph_transformer(
                    input_dim=input_dim,
                    hidden_dim=32,  # Smaller for testing
                    output_dim=output_dim,
                    num_layers=2,
                    **config
                )
                
                test_output, _ = test_model(x, edge_index, batch)
                print(f"  Config {i+1} ({config}): ✅")
                
            except Exception as e:
                print(f"  Config {i+1} ({config}): ❌ - {e}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()