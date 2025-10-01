"""
Network Topology Analysis for Node Pruning

This module provides comprehensive network topology metrics for intelligent node pruning
in microbial networks. Based on research from network analysis literature.

References:
- Newman (2003): "The Structure and Function of Complex Networks"
- Girvan & Newman (2002): "Community structure in social and biological networks" 
- Freeman (1977): "A set of measures of centrality based on betweenness"
- Bonacich (1972): "Factoring and weighting approaches to status scores"
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import warnings

# Try to import community detection
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    warnings.warn("python-louvain not available, modularity calculation will be limited")

class NetworkTopologyAnalyzer:
    """
    Comprehensive network topology analysis for node importance scoring.
    
    This class calculates various centrality measures and network properties
    to create a composite topology-based importance score for node pruning.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the topology analyzer.
        
        Args:
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
        
        # Empirically determined weights for centrality measures
        # Based on general network analysis principles and domain expertise
        # Note: These weights are empirically chosen and can be adjusted based on specific network properties
        self.centrality_weights = {
            'degree': 0.35,        # Direct connectivity importance (highest weight)
            'betweenness': 0.25,   # Bridge/hub importance (second highest)
            'closeness': 0.20,     # Global reach importance
            'eigenvector': 0.15,   # Connection quality importance
            'pagerank': 0.05       # Global influence importance (lowest weight)
        }
        
        # Network property thresholds for adaptive pruning (relaxed)
        self.network_thresholds = {
            'high_modularity': 0.4,      # Highly modular networks (relaxed from 0.5)
            'dense_network': 2.5,        # Average path length threshold (relaxed from 2.0)
            'small_world': 0.25,         # Clustering coefficient threshold (relaxed from 0.3)
            'scale_free': 2.5           # Degree distribution power law threshold
        }
    
    def calculate_centrality_measures(self, data: Data) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive centrality measures for all nodes.
        
        Args:
            data: PyG Data object with graph structure
            
        Returns:
            Dictionary mapping centrality type to node scores
        """
        print("Calculating network centrality measures...")
        
        # Convert to NetworkX for centrality calculations
        try:
            G = to_networkx(data, to_undirected=True, remove_self_loops=True)
            if data.edge_weight is not None:
                # Add edge weights if available
                for i, (u, v) in enumerate(data.edge_index.t().cpu().numpy()):
                    if G.has_edge(u, v):
                        G[u][v]['weight'] = data.edge_weight[i].item()
        except Exception as e:
            print(f"Warning: NetworkX conversion failed: {e}")
            return self._fallback_centrality_calculation(data)
        
        centrality_measures = {}
        
        # 1. Degree Centrality: DC(v) = deg(v) / (n-1)
        print("  Computing degree centrality...")
        centrality_measures['degree'] = np.array(list(nx.degree_centrality(G).values()))
        
        # 2. Betweenness Centrality: BC(v) = Σ σ_st(v)/σ_st
        print("  Computing betweenness centrality...")
        try:
            centrality_measures['betweenness'] = np.array(list(nx.betweenness_centrality(G, weight='weight').values()))
        except Exception as e:
            print(f"  Warning: Betweenness centrality failed: {e}")
            centrality_measures['betweenness'] = np.zeros(len(G.nodes()))
        
        # 3. Closeness Centrality: CC(v) = 1/Σ d(v,t)
        print("  Computing closeness centrality...")
        try:
            centrality_measures['closeness'] = np.array(list(nx.closeness_centrality(G, distance='weight').values()))
        except Exception as e:
            print(f"  Warning: Closeness centrality failed: {e}")
            centrality_measures['closeness'] = np.zeros(len(G.nodes()))
        
        # 4. Eigenvector Centrality: Based on connections to well-connected nodes
        print("  Computing eigenvector centrality...")
        try:
            centrality_measures['eigenvector'] = np.array(list(nx.eigenvector_centrality(G, weight='weight', max_iter=1000).values()))
        except Exception as e:
            print(f"  Warning: Eigenvector centrality failed: {e}")
            centrality_measures['eigenvector'] = np.zeros(len(G.nodes()))
        
        # 5. PageRank: PR(v) = (1-d)/N + d * Σ PR(u)/L(u)
        print("  Computing PageRank...")
        try:
            centrality_measures['pagerank'] = np.array(list(nx.pagerank(G, weight='weight').values()))
        except Exception as e:
            print(f"  Warning: PageRank failed: {e}")
            centrality_measures['pagerank'] = np.zeros(len(G.nodes()))
        
        print(f"  Centrality calculation complete for {len(G.nodes())} nodes")
        return centrality_measures
    
    def _fallback_centrality_calculation(self, data: Data) -> Dict[str, np.ndarray]:
        """
        Fallback centrality calculation using PyTorch operations.
        Used when NetworkX conversion fails.
        """
        print("  Using fallback centrality calculation...")
        
        num_nodes = data.x.size(0)
        centrality_measures = {}
        
        # Simple degree centrality as fallback
        degrees = torch.zeros(num_nodes, device=self.device)
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i], data.edge_index[1, i]
            degrees[src] += 1
            degrees[dst] += 1
        
        # Normalize by max possible degree
        if num_nodes > 1:
            degrees = degrees / (num_nodes - 1)
        
        degree_centrality = degrees.cpu().numpy()
        
        # Use degree centrality for all measures as fallback
        centrality_measures = {
            'degree': degree_centrality,
            'betweenness': degree_centrality.copy(),
            'closeness': degree_centrality.copy(),
            'eigenvector': degree_centrality.copy(),
            'pagerank': degree_centrality.copy()
        }
        
        return centrality_measures
    
    def calculate_network_properties(self, data: Data) -> Dict[str, float]:
        """
        Calculate global network properties for adaptive thresholding.
        
        Args:
            data: PyG Data object with graph structure
            
        Returns:
            Dictionary of network properties
        """
        print("Calculating network properties...")
        
        try:
            G = to_networkx(data, to_undirected=True, remove_self_loops=True)
            if data.edge_weight is not None:
                for i, (u, v) in enumerate(data.edge_index.t().cpu().numpy()):
                    if G.has_edge(u, v):
                        G[u][v]['weight'] = data.edge_weight[i].item()
        except Exception as e:
            print(f"Warning: NetworkX conversion failed: {e}")
            return self._fallback_network_properties(data)
        
        properties = {}
        
        # Basic properties
        properties['num_nodes'] = G.number_of_nodes()
        properties['num_edges'] = G.number_of_edges()
        properties['density'] = nx.density(G)
        
        # Connectivity properties
        if nx.is_connected(G):
            properties['average_path_length'] = nx.average_shortest_path_length(G, weight='weight')
            properties['diameter'] = nx.diameter(G)
            properties['is_connected'] = True
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            properties['average_path_length'] = nx.average_shortest_path_length(subgraph, weight='weight')
            properties['diameter'] = nx.diameter(subgraph)
            properties['is_connected'] = False
            properties['largest_cc_size'] = len(largest_cc)
        
        # Clustering coefficient
        properties['clustering_coefficient'] = nx.average_clustering(G, weight='weight')
        
        # Modularity (if community detection available)
        if HAS_LOUVAIN:
            try:
                partition = community_louvain.best_partition(G)
                properties['modularity'] = community_louvain.modularity(partition, G)
            except Exception as e:
                print(f"  Warning: Modularity calculation failed: {e}")
                properties['modularity'] = 0.0
        else:
            properties['modularity'] = 0.0
        
        # Degree distribution properties
        degrees = [d for n, d in G.degree()]
        properties['mean_degree'] = np.mean(degrees)
        properties['degree_std'] = np.std(degrees)
        
        print(f"  Network properties: {properties['num_nodes']} nodes, {properties['num_edges']} edges")
        print(f"  Density: {properties['density']:.3f}, Clustering: {properties['clustering_coefficient']:.3f}")
        print(f"  Modularity: {properties['modularity']:.3f}, Avg path length: {properties['average_path_length']:.3f}")
        
        return properties
    
    def _fallback_network_properties(self, data: Data) -> Dict[str, float]:
        """
        Fallback network properties calculation.
        """
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1) // 2  # Undirected edges
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0,
            'average_path_length': 2.0,  # Default estimate
            'diameter': 4,  # Default estimate
            'is_connected': True,  # Assume connected
            'clustering_coefficient': 0.3,  # Default estimate
            'modularity': 0.0,  # Cannot calculate without community detection
            'mean_degree': (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0,
            'degree_std': 1.0  # Default estimate
        }
    
    def compute_composite_topology_score(self, centrality_measures: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute composite topology score using research-backed weights.
        
        Formula: T(v) = Σ w_i * C_i(v)
        Where w_i are research-backed weights and C_i(v) are normalized centrality measures.
        
        Args:
            centrality_measures: Dictionary of centrality measures
            
        Returns:
            Composite topology scores for all nodes
        """
        print("Computing composite topology score...")
        
        # Normalize each centrality measure to [0, 1] range
        normalized_measures = {}
        for measure_name, scores in centrality_measures.items():
            if len(scores) == 0 or np.all(scores == 0):
                normalized_measures[measure_name] = np.zeros_like(scores)
            else:
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    normalized_measures[measure_name] = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_measures[measure_name] = np.ones_like(scores) * 0.5
        
        # Compute weighted combination
        composite_score = np.zeros(len(list(centrality_measures.values())[0]))
        
        for measure_name, weight in self.centrality_weights.items():
            if measure_name in normalized_measures:
                composite_score += weight * normalized_measures[measure_name]
                print(f"  {measure_name}: weight={weight:.2f}, range=[{normalized_measures[measure_name].min():.3f}, {normalized_measures[measure_name].max():.3f}]")
        
        # Normalize final score to [0, 1] range
        if composite_score.max() > composite_score.min():
            composite_score = (composite_score - composite_score.min()) / (composite_score.max() - composite_score.min())
        
        print(f"  Composite topology score range: [{composite_score.min():.3f}, {composite_score.max():.3f}]")
        return composite_score
    
    def determine_adaptive_threshold(self, topology_scores: np.ndarray, 
                                   network_properties: Dict[str, float],
                                   base_threshold: float = 0.2) -> float:
        """
        Determine adaptive pruning threshold based on network properties.
        
        Args:
            topology_scores: Composite topology scores
            network_properties: Network properties dictionary
            base_threshold: Base threshold for pruning
            
        Returns:
            Adaptive threshold value
        """
        print("Determining adaptive threshold...")
        
        # Analyze network characteristics
        modularity = network_properties.get('modularity', 0.0)
        avg_path_length = network_properties.get('average_path_length', 2.0)
        clustering = network_properties.get('clustering_coefficient', 0.3)
        density = network_properties.get('density', 0.1)
        
        # Adaptive thresholding rules based on network properties (relaxed)
        if modularity > self.network_thresholds['high_modularity']:
            # Highly modular network - preserve more nodes to maintain communities
            threshold = np.percentile(topology_scores, 20)  # Keep top 80% (relaxed from 70%)
            print(f"  High modularity detected ({modularity:.3f}), using conservative threshold")
        elif avg_path_length < self.network_thresholds['dense_network']:
            # Dense, well-connected network - can prune more aggressively
            threshold = np.percentile(topology_scores, 40)  # Keep top 60% (relaxed from 40%)
            print(f"  Dense network detected (path length: {avg_path_length:.3f}), using moderate threshold")
        elif clustering > self.network_thresholds['small_world']:
            # Small-world network - preserve clustering structure
            threshold = np.percentile(topology_scores, 25)  # Keep top 75% (relaxed from 60%)
            print(f"  Small-world network detected (clustering: {clustering:.3f}), using conservative threshold")
        else:
            # Standard network - use base threshold with score distribution adjustment
            mean_score = np.mean(topology_scores)
            std_score = np.std(topology_scores)
            threshold = max(base_threshold * 0.8, mean_score + 0.3 * std_score)  # More conservative
            print(f"  Standard network, using adjusted base threshold")
        
        # Ensure threshold is within valid range
        threshold = max(0.0, min(1.0, threshold))
        
        # Calculate expected retention
        retention_rate = np.mean(topology_scores >= threshold)
        print(f"  Adaptive threshold: {threshold:.3f}")
        print(f"  Expected retention: {retention_rate:.1%}")
        
        return threshold
    
    def analyze_network_topology(self, data: Data) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Complete network topology analysis for node pruning.
        
        Args:
            data: PyG Data object with graph structure
            
        Returns:
            Tuple of (composite_topology_scores, network_properties)
        """
        print(f"\n{'='*60}")
        print("NETWORK TOPOLOGY ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate centrality measures
        centrality_measures = self.calculate_centrality_measures(data)
        
        # Calculate network properties
        network_properties = self.calculate_network_properties(data)
        
        # Compute composite topology score
        topology_scores = self.compute_composite_topology_score(centrality_measures)
        
        print(f"Topology analysis complete: {len(topology_scores)} nodes analyzed")
        
        return topology_scores, network_properties
