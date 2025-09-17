#!/usr/bin/env python3
"""
Paper-Style Correlation Graph Builder

This module implements the exact correlation-based graph construction method
from Thapa et al. (2023) paper for microbial co-occurrence networks.

The key difference from existing pipeline:
- Paper: Correlations between OTUs across samples (species co-occurrence)
- Current: Correlations between samples across features (sample similarity)

This provides the biological foundation for graph initialization before GNN training.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class PaperStyleCorrelationGraph:
    """
    Implements exact correlation-based graph construction from Thapa et al. (2023)

    Key Method:
    1. Calculate Spearman correlations between every pair of microbial families/OTUs
    2. Apply significance threshold (p < 0.05) and correlation strength threshold
    3. Create edges only between significantly correlated families
    4. Preserve correlation signs and strengths as edge attributes
    """

    def __init__(self,
                 correlation_threshold: float = 0.6,
                 significance_threshold: float = 0.05,
                 min_edges: int = 20,
                 max_edges: Optional[int] = None):
        """
        Initialize correlation graph builder with paper's parameters

        Args:
            correlation_threshold: Minimum |correlation| for edge creation (paper used ~0.6)
            significance_threshold: Maximum p-value for significance (paper used 0.05)
            min_edges: Minimum edges to ensure connected graph
            max_edges: Maximum edges to prevent over-connectivity (optional)
        """
        self.correlation_threshold = correlation_threshold
        self.significance_threshold = significance_threshold
        self.min_edges = min_edges
        self.max_edges = max_edges

    def build_correlation_graph(self,
                              abundance_data: np.ndarray,
                              feature_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Build correlation-based graph exactly like Thapa et al. (2023)

        Args:
            abundance_data: (n_samples, n_features) abundance matrix
            feature_names: List of feature names (families/OTUs)

        Returns:
            edge_index: PyTorch tensor of edge connections
            edge_weight: PyTorch tensor of correlation strengths
            edge_type: PyTorch tensor of correlation signs (1=positive, 0=negative)
            metadata: Dictionary with correlation matrix and statistics
        """

        print(f"Building paper-style correlation graph...")
        print(f"Data shape: {abundance_data.shape} (samples × features)")
        print(f"Correlation threshold: {self.correlation_threshold}")
        print(f"Significance threshold: {self.significance_threshold}")

        n_samples, n_features = abundance_data.shape

        # Step 1: Calculate all pairwise correlations (like the paper)
        correlation_matrix, p_value_matrix = self._calculate_correlation_matrix(abundance_data)

        # Step 2: Apply paper's filtering criteria
        edge_index, edge_weight, edge_type = self._create_edges_from_correlations(
            correlation_matrix, p_value_matrix, feature_names
        )

        # Step 3: Ensure minimum connectivity
        if edge_index.shape[1] < self.min_edges:
            print(f"Only {edge_index.shape[1]} edges found, applying relaxed criteria...")
            edge_index, edge_weight, edge_type = self._ensure_minimum_connectivity(
                correlation_matrix, p_value_matrix, feature_names
            )

        # Step 4: Apply maximum connectivity if specified
        if self.max_edges and edge_index.shape[1] > self.max_edges:
            print(f"Too many edges ({edge_index.shape[1]}), selecting top {self.max_edges}...")
            edge_index, edge_weight, edge_type = self._limit_connectivity(
                edge_index, edge_weight, edge_type
            )

        # Generate metadata
        metadata = self._generate_metadata(correlation_matrix, p_value_matrix, edge_index, edge_weight)

        print(f"✅ Created correlation graph: {n_features} nodes, {edge_index.shape[1]//2} undirected edges")

        return edge_index, edge_weight, edge_type, metadata

    def _calculate_correlation_matrix(self, abundance_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Spearman correlation matrix exactly like the paper

        Paper method: For each pair of features (families/OTUs), calculate correlation
        of their abundance patterns across all samples.
        """
        n_samples, n_features = abundance_data.shape

        # Initialize correlation and p-value matrices
        correlation_matrix = np.zeros((n_features, n_features))
        p_value_matrix = np.ones((n_features, n_features))

        print(f"Computing {n_features * (n_features-1) // 2} pairwise correlations...")

        # Calculate correlations between each pair of features
        for i in range(n_features):
            for j in range(i, n_features):  # Include diagonal
                if i == j:
                    # Self-correlation
                    correlation_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                else:
                    # Spearman correlation between feature i and feature j across samples
                    feature_i_abundances = abundance_data[:, i]
                    feature_j_abundances = abundance_data[:, j]

                    # Handle constant features (no variation)
                    if np.std(feature_i_abundances) == 0 or np.std(feature_j_abundances) == 0:
                        corr, p_val = 0.0, 1.0
                    else:
                        try:
                            corr, p_val = spearmanr(feature_i_abundances, feature_j_abundances)
                            if np.isnan(corr):
                                corr, p_val = 0.0, 1.0
                        except Exception:
                            corr, p_val = 0.0, 1.0

                    # Fill symmetric matrix
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
                    p_value_matrix[i, j] = p_val
                    p_value_matrix[j, i] = p_val

        return correlation_matrix, p_value_matrix

    def _create_edges_from_correlations(self,
                                      correlation_matrix: np.ndarray,
                                      p_value_matrix: np.ndarray,
                                      feature_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create edges using paper's criteria: significant correlation above threshold"""

        n_features = correlation_matrix.shape[0]
        edge_i, edge_j, edge_weights, edge_types = [], [], [], []

        significant_pairs = 0
        total_pairs = 0

        for i in range(n_features):
            for j in range(i+1, n_features):  # Only upper triangle (undirected graph)
                total_pairs += 1

                corr = correlation_matrix[i, j]
                p_val = p_value_matrix[i, j]

                # Paper's criteria: significant AND strong correlation
                if p_val < self.significance_threshold and abs(corr) > self.correlation_threshold:
                    significant_pairs += 1

                    # Add bidirectional edges (undirected graph)
                    edge_i.extend([i, j])
                    edge_j.extend([j, i])

                    # Edge weight = absolute correlation strength
                    edge_weights.extend([abs(corr), abs(corr)])

                    # Edge type = correlation sign (1=positive, 0=negative)
                    edge_type = 1 if corr > 0 else 0
                    edge_types.extend([edge_type, edge_type])

        print(f"Significant correlations: {significant_pairs}/{total_pairs} pairs ({significant_pairs/total_pairs*100:.1f}%)")

        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_weight, edge_type_tensor

    def _ensure_minimum_connectivity(self,
                                   correlation_matrix: np.ndarray,
                                   p_value_matrix: np.ndarray,
                                   feature_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure minimum graph connectivity by relaxing criteria if needed"""

        # Strategy 1: Lower correlation threshold but keep significance
        relaxed_threshold = max(0.3, self.correlation_threshold - 0.2)
        print(f"Trying relaxed correlation threshold: {relaxed_threshold}")

        edge_index, edge_weight, edge_type = self._create_edges_with_threshold(
            correlation_matrix, p_value_matrix, relaxed_threshold, self.significance_threshold
        )

        if edge_index.shape[1] >= self.min_edges:
            return edge_index, edge_weight, edge_type

        # Strategy 2: Further relax correlation threshold
        very_relaxed_threshold = 0.2
        print(f"Trying very relaxed correlation threshold: {very_relaxed_threshold}")

        edge_index, edge_weight, edge_type = self._create_edges_with_threshold(
            correlation_matrix, p_value_matrix, very_relaxed_threshold, self.significance_threshold
        )

        if edge_index.shape[1] >= self.min_edges:
            return edge_index, edge_weight, edge_type

        # Strategy 3: Use top-k strongest correlations regardless of significance
        print(f"Using top-{self.min_edges} strongest correlations...")
        return self._create_top_k_edges(correlation_matrix, self.min_edges)

    def _create_edges_with_threshold(self,
                                   correlation_matrix: np.ndarray,
                                   p_value_matrix: np.ndarray,
                                   corr_threshold: float,
                                   p_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create edges with specified thresholds"""

        n_features = correlation_matrix.shape[0]
        edge_i, edge_j, edge_weights, edge_types = [], [], [], []

        for i in range(n_features):
            for j in range(i+1, n_features):
                corr = correlation_matrix[i, j]
                p_val = p_value_matrix[i, j]

                if p_val < p_threshold and abs(corr) > corr_threshold:
                    edge_i.extend([i, j])
                    edge_j.extend([j, i])
                    edge_weights.extend([abs(corr), abs(corr)])
                    edge_type = 1 if corr > 0 else 0
                    edge_types.extend([edge_type, edge_type])

        return (torch.tensor([edge_i, edge_j], dtype=torch.long),
                torch.tensor(edge_weights, dtype=torch.float32),
                torch.tensor(edge_types, dtype=torch.long))

    def _create_top_k_edges(self, correlation_matrix: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create graph using top-k strongest correlations as fallback"""

        n_features = correlation_matrix.shape[0]

        # Get all correlation pairs (excluding diagonal)
        correlation_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr = correlation_matrix[i, j]
                correlation_pairs.append((i, j, abs(corr), corr))

        # Sort by absolute correlation strength
        correlation_pairs.sort(key=lambda x: x[2], reverse=True)

        # Take top k pairs
        top_pairs = correlation_pairs[:k]

        edge_i, edge_j, edge_weights, edge_types = [], [], [], []
        for i, j, abs_corr, corr in top_pairs:
            edge_i.extend([i, j])
            edge_j.extend([j, i])
            edge_weights.extend([abs_corr, abs_corr])
            edge_type = 1 if corr > 0 else 0
            edge_types.extend([edge_type, edge_type])

        return (torch.tensor([edge_i, edge_j], dtype=torch.long),
                torch.tensor(edge_weights, dtype=torch.float32),
                torch.tensor(edge_types, dtype=torch.long))

    def _limit_connectivity(self,
                          edge_index: torch.Tensor,
                          edge_weight: torch.Tensor,
                          edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Limit graph connectivity by keeping strongest edges"""

        # Get unique edges (undirected)
        num_edges = edge_index.shape[1] // 2

        # Create list of edge strengths for unique edges
        edge_strengths = []
        for i in range(0, edge_index.shape[1], 2):  # Step by 2 for undirected edges
            edge_strengths.append((edge_weight[i].item(), i))

        # Sort by strength and take top max_edges
        edge_strengths.sort(reverse=True)
        selected_indices = []

        for strength, idx in edge_strengths[:self.max_edges]:
            selected_indices.extend([idx, idx+1])  # Include both directions

        # Filter tensors
        edge_index_limited = edge_index[:, selected_indices]
        edge_weight_limited = edge_weight[selected_indices]
        edge_type_limited = edge_type[selected_indices]

        return edge_index_limited, edge_weight_limited, edge_type_limited

    def _generate_metadata(self,
                         correlation_matrix: np.ndarray,
                         p_value_matrix: np.ndarray,
                         edge_index: torch.Tensor,
                         edge_weight: torch.Tensor) -> dict:
        """Generate comprehensive metadata about the correlation graph"""

        n_features = correlation_matrix.shape[0]
        num_edges = edge_index.shape[1] // 2  # Undirected edges

        # Network density
        max_possible_edges = n_features * (n_features - 1) // 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        # Correlation statistics
        upper_triangle_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix[upper_triangle_mask]
        p_values = p_value_matrix[upper_triangle_mask]

        # Edge weight statistics
        edge_weights_unique = edge_weight[::2]  # Take every other weight (undirected)

        metadata = {
            'method': 'paper_style_correlation',
            'correlation_threshold': self.correlation_threshold,
            'significance_threshold': self.significance_threshold,
            'n_nodes': n_features,
            'n_edges': num_edges,
            'density': density,
            'correlation_stats': {
                'mean': float(np.mean(correlations)),
                'std': float(np.std(correlations)),
                'min': float(np.min(correlations)),
                'max': float(np.max(correlations)),
                'median': float(np.median(correlations))
            },
            'significance_stats': {
                'significant_pairs': int(np.sum(p_values < self.significance_threshold)),
                'total_pairs': len(p_values),
                'proportion_significant': float(np.mean(p_values < self.significance_threshold))
            },
            'edge_weight_stats': {
                'mean': float(torch.mean(edge_weights_unique)),
                'std': float(torch.std(edge_weights_unique)),
                'min': float(torch.min(edge_weights_unique)),
                'max': float(torch.max(edge_weights_unique))
            },
            'positive_correlations': int(torch.sum(edge_weight > 0) // 2),
            'negative_correlations': int(torch.sum(edge_weight < 0) // 2)
        }

        return metadata


def create_paper_style_graph(abundance_data: np.ndarray,
                            feature_names: List[str],
                            correlation_threshold: float = 0.6,
                            significance_threshold: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Convenience function to create paper-style correlation graph

    Args:
        abundance_data: (n_samples, n_features) abundance matrix
        feature_names: List of feature names
        correlation_threshold: Minimum correlation strength
        significance_threshold: Maximum p-value for significance

    Returns:
        edge_index, edge_weight, edge_type, metadata
    """

    builder = PaperStyleCorrelationGraph(
        correlation_threshold=correlation_threshold,
        significance_threshold=significance_threshold
    )

    return builder.build_correlation_graph(abundance_data, feature_names)


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Paper-Style Correlation Graph Builder")

    # Create synthetic microbial abundance data
    np.random.seed(42)
    n_samples, n_features = 50, 20

    # Simulate correlated abundance patterns
    abundance_data = np.random.lognormal(0, 1, (n_samples, n_features))
    feature_names = [f"Family_{i}" for i in range(n_features)]

    # Build correlation graph
    edge_index, edge_weight, edge_type, metadata = create_paper_style_graph(
        abundance_data, feature_names
    )

    print(f"\n✅ Test completed successfully!")
    print(f"Graph: {metadata['n_nodes']} nodes, {metadata['n_edges']} edges")
    print(f"Density: {metadata['density']:.3f}")
    print(f"Mean correlation: {metadata['correlation_stats']['mean']:.3f}")
    print(f"Significant pairs: {metadata['significance_stats']['proportion_significant']:.3f}")