#!/usr/bin/env python3
"""
Comprehensive Baseline Comparator for Node Pruning Research

This module provides comprehensive baseline comparison methods for evaluating
node importance and pruning techniques against established state-of-the-art methods.

Implemented Baselines:
====================
1. Integrated Gradients - More robust gradient-based importance
2. GNNExplainer - Standard graph explainability method  
3. PageRank - Graph centrality-based importance
4. Degree Centrality - Simple connectivity-based importance
5. Betweenness Centrality - Path-based importance
6. Random Baseline - Control method for statistical comparison

Mathematical Foundation:
======================
Each baseline uses different principles:

1. Integrated Gradients: IG(x) = (x - x') ⊙ ∫₀¹ ∇f(x' + α(x-x'))dα
2. PageRank: PR(v) = (1-d)/N + d × Σ_{u∈M(v)} PR(u)/C(u)
3. Degree Centrality: DC(v) = deg(v) / (n-1)
4. Betweenness: BC(v) = Σ_{s≠v≠t} σ_st(v)/σ_st

Authors: Research Team
Date: 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path

# Optional imports with fallbacks
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not available - centrality baselines disabled")

try:
    from torch_geometric.utils import to_networkx, from_networkx
    from torch_geometric.explain import Explainer, GNNExplainer
    HAS_PYGEOMETRIC_EXPLAIN = True
except ImportError:
    HAS_PYGEOMETRIC_EXPLAIN = False
    warnings.warn("PyG explainer not available - GNNExplainer baseline disabled")


class BaselineComparator:
    """
    Comprehensive baseline comparison framework for node importance methods
    
    Provides implementations of state-of-the-art node importance baselines
    and statistical comparison with the proposed unified pruning method.
    """
    
    def __init__(self, device: torch.device = None, random_state: int = 42):
        """
        Initialize baseline comparator
        
        Args:
            device: Computing device (CPU/GPU)
            random_state: Random seed for reproducibility
        """
        self.device = device or torch.device('cpu')
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Available baseline methods
        self.baseline_methods = {
            'integrated_gradients': self._integrated_gradients_importance,
            'vanilla_gradients': self._vanilla_gradients_importance,
            'pagerank': self._pagerank_importance,
            'degree_centrality': self._degree_centrality_importance,
            'betweenness_centrality': self._betweenness_centrality_importance,
            'closeness_centrality': self._closeness_centrality_importance,
            'eigenvector_centrality': self._eigenvector_centrality_importance,
            'random_baseline': self._random_importance
        }
        
        # Add GNNExplainer if available
        if HAS_PYGEOMETRIC_EXPLAIN:
            self.baseline_methods['gnn_explainer'] = self._gnn_explainer_importance
    
    def _integrated_gradients_importance(self, model, data, steps: int = 50) -> np.ndarray:
        """
        Integrated Gradients for node importance
        
        Mathematical Formula:
        IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂f(x' + α(x - x'))/∂x_i dα
        
        Where:
        - x = input features
        - x' = baseline (typically zeros)
        - f = model output function
        - α = interpolation coefficient
        
        More robust than vanilla gradients by considering the integration path.
        
        Args:
            model: Trained GNN model
            data: Graph data object
            steps: Number of integration steps
            
        Returns:
            Node importance scores
        """
        model.eval()
        model.zero_grad()
        
        # Create baseline (zero features)
        baseline = torch.zeros_like(data.x, device=self.device)
        
        # Generate interpolated inputs along the path
        alphas = torch.linspace(0, 1, steps, device=self.device)
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (data.x - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            if hasattr(data, 'batch') and data.batch is not None:
                batch = data.batch
            else:
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            
            # Handle different model output formats
            output = model(interpolated, data.edge_index, batch)
            if isinstance(output, tuple):
                output = output[0]  # Take first element if tuple
            
            # Compute gradients
            grad = torch.autograd.grad(
                outputs=output.sum(), 
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_grads = (data.x - baseline) * avg_gradients
        
        # Sum across features for node-level importance
        node_importance = integrated_grads.abs().sum(dim=1)
        
        return node_importance.detach().cpu().numpy()
    
    def _vanilla_gradients_importance(self, model, data) -> np.ndarray:
        """
        Vanilla gradient-based node importance
        
        Mathematical Formula:
        VG_i(x) = |∂f(x)/∂x_i|
        
        Simple but can suffer from saturation issues.
        """
        model.eval()
        model.zero_grad()
        
        x = data.x.clone().requires_grad_(True)
        
        # Forward pass
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        
        output = model(x, data.edge_index, batch)
        if isinstance(output, tuple):
            output = output[0]
        
        # Compute gradients
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Sum across features for node-level importance
        node_importance = grad.abs().sum(dim=1)
        
        return node_importance.detach().cpu().numpy()
    
    def _gnn_explainer_importance(self, model, data) -> np.ndarray:
        """
        GNNExplainer-based node importance
        
        Uses the standard GNNExplainer method to generate node importance scores
        based on edge mask optimization.
        """
        if not HAS_PYGEOMETRIC_EXPLAIN:
            raise RuntimeError("PyTorch Geometric explainer not available")
        
        try:
            # Create explainer
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=100, lr=0.01),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='regression',
                    task_level='graph',
                    return_type='raw'
                )
            )
            
            # Generate explanation
            if hasattr(data, 'batch') and data.batch is not None:
                batch = data.batch
            else:
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            
            explanation = explainer(data.x, data.edge_index, batch=batch)
            
            # Extract node importance from node mask
            if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
                node_importance = explanation.node_mask.sum(dim=1)
                return node_importance.detach().cpu().numpy()
            else:
                # Fallback to edge mask aggregation
                edge_mask = explanation.edge_mask
                node_importance = torch.zeros(data.x.size(0), device=self.device)
                
                # Aggregate edge importance to nodes
                for i in range(data.edge_index.size(1)):
                    src, dst = data.edge_index[0, i], data.edge_index[1, i]
                    edge_imp = edge_mask[i]
                    node_importance[src] += edge_imp
                    node_importance[dst] += edge_imp
                
                return node_importance.detach().cpu().numpy()
        
        except Exception as e:
            warnings.warn(f"GNNExplainer failed: {e}, using gradient fallback")
            return self._vanilla_gradients_importance(model, data)
    
    def _pagerank_importance(self, data, damping: float = 0.85) -> np.ndarray:
        """
        PageRank centrality as node importance
        
        Mathematical Formula:
        PR(v) = (1-d)/N + d × Σ_{u∈M(v)} PR(u)/C(u)
        
        Where:
        - d = damping factor
        - N = number of nodes
        - M(v) = nodes linking to v
        - C(u) = out-degree of u
        
        Args:
            data: Graph data
            damping: PageRank damping factor
            
        Returns:
            PageRank scores as node importance
        """
        if not HAS_NETWORKX:
            warnings.warn("NetworkX not available, using degree centrality fallback")
            return self._degree_centrality_importance(data)
        
        try:
            # Convert to NetworkX graph
            G = to_networkx(data, to_undirected=True)
            
            # Compute PageRank
            pagerank = nx.pagerank(G, alpha=damping, max_iter=1000, tol=1e-6)
            
            # Convert to array (ensure proper ordering)
            node_importance = np.array([pagerank.get(i, 0) for i in range(len(pagerank))])
            
            return node_importance
            
        except Exception as e:
            warnings.warn(f"PageRank computation failed: {e}, using degree fallback")
            return self._degree_centrality_importance(data)
    
    def _degree_centrality_importance(self, data) -> np.ndarray:
        """
        Degree centrality as node importance
        
        Mathematical Formula:
        DC(v) = deg(v) / (n-1)
        
        Simple connectivity-based importance measure.
        """
        # Count node degrees
        num_nodes = data.x.size(0)
        degrees = torch.zeros(num_nodes, device=self.device)
        
        # Count edges for each node
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i], data.edge_index[1, i]
            degrees[src] += 1
            degrees[dst] += 1
        
        # Normalize by max possible degree
        if num_nodes > 1:
            degrees = degrees / (num_nodes - 1)
        
        return degrees.cpu().numpy()
    
    def _betweenness_centrality_importance(self, data) -> np.ndarray:
        """
        Betweenness centrality as node importance
        
        Mathematical Formula:
        BC(v) = Σ_{s≠v≠t} σ_st(v)/σ_st
        
        Where σ_st is the number of shortest paths from s to t,
        and σ_st(v) is the number passing through v.
        """
        if not HAS_NETWORKX:
            warnings.warn("NetworkX not available, using degree centrality fallback")
            return self._degree_centrality_importance(data)
        
        try:
            # Convert to NetworkX graph
            G = to_networkx(data, to_undirected=True)
            
            # Compute betweenness centrality
            betweenness = nx.betweenness_centrality(G, normalized=True)
            
            # Convert to array
            node_importance = np.array([betweenness.get(i, 0) for i in range(len(betweenness))])
            
            return node_importance
            
        except Exception as e:
            warnings.warn(f"Betweenness centrality failed: {e}, using degree fallback")
            return self._degree_centrality_importance(data)
    
    def _closeness_centrality_importance(self, data) -> np.ndarray:
        """
        Closeness centrality as node importance
        
        Mathematical Formula:
        CC(v) = (n-1) / Σ_{u≠v} d(v,u)
        
        Where d(v,u) is the shortest path distance.
        """
        if not HAS_NETWORKX:
            warnings.warn("NetworkX not available, using degree centrality fallback")
            return self._degree_centrality_importance(data)
        
        try:
            G = to_networkx(data, to_undirected=True)
            closeness = nx.closeness_centrality(G, normalized=True)
            node_importance = np.array([closeness.get(i, 0) for i in range(len(closeness))])
            return node_importance
            
        except Exception as e:
            warnings.warn(f"Closeness centrality failed: {e}, using degree fallback")
            return self._degree_centrality_importance(data)
    
    def _eigenvector_centrality_importance(self, data) -> np.ndarray:
        """
        Eigenvector centrality as node importance
        
        Mathematical Formula:
        EC(v) = (1/λ) × Σ_{u∈M(v)} EC(u)
        
        Where λ is the largest eigenvalue of the adjacency matrix.
        """
        if not HAS_NETWORKX:
            warnings.warn("NetworkX not available, using degree centrality fallback")
            return self._degree_centrality_importance(data)
        
        try:
            G = to_networkx(data, to_undirected=True)
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
            node_importance = np.array([eigenvector.get(i, 0) for i in range(len(eigenvector))])
            return node_importance
            
        except Exception as e:
            warnings.warn(f"Eigenvector centrality failed: {e}, using degree fallback")
            return self._degree_centrality_importance(data)
    
    def _random_importance(self, data) -> np.ndarray:
        """
        Random node importance (control baseline)
        
        Assigns random importance scores for statistical comparison.
        """
        num_nodes = data.x.size(0)
        np.random.seed(self.random_state)
        return np.random.random(num_nodes)
    
    def compare_all_baselines(self, 
                            model, 
                            data, 
                            unified_scores: np.ndarray,
                            node_names: Optional[List[str]] = None) -> Dict:
        """
        Compare unified pruning method against all baseline methods
        
        Args:
            model: Trained GNN model
            data: Graph data
            unified_scores: Importance scores from unified method
            node_names: Node names for interpretability
            
        Returns:
            Comprehensive comparison results
        """
        print("Running comprehensive baseline comparison...")
        
        baseline_results = {}
        comparison_stats = {}
        
        # Run each baseline method
        for method_name, method_func in self.baseline_methods.items():
            print(f"  Running {method_name}...")
            
            try:
                if method_name in ['integrated_gradients', 'vanilla_gradients', 'gnn_explainer']:
                    # Model-dependent methods
                    importance_scores = method_func(model, data)
                else:
                    # Graph structure-dependent methods
                    importance_scores = method_func(data)
                
                # Store results
                baseline_results[method_name] = {
                    'importance_scores': importance_scores,
                    'ranking': np.argsort(importance_scores)[::-1],  # Descending order
                    'success': True
                }
                
                print(f"    ✅ {method_name} completed")
                
            except Exception as e:
                print(f"    ❌ {method_name} failed: {e}")
                baseline_results[method_name] = {
                    'importance_scores': None,
                    'ranking': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Compare with unified method
        unified_ranking = np.argsort(unified_scores)[::-1]
        
        print("Computing ranking correlations...")
        
        # Compute ranking correlations
        for method_name, results in baseline_results.items():
            if results['success']:
                baseline_ranking = results['ranking']
                
                # Spearman rank correlation
                spear_corr, spear_p = stats.spearmanr(unified_ranking, baseline_ranking)
                
                # Kendall's tau
                kendall_corr, kendall_p = stats.kendalltau(unified_ranking, baseline_ranking)
                
                # Top-k overlap analysis
                top_k_overlaps = {}
                for k in [5, 10, min(20, len(unified_ranking))]:
                    if k <= len(unified_ranking):
                        unified_top_k = set(unified_ranking[:k])
                        baseline_top_k = set(baseline_ranking[:k])
                        overlap = len(unified_top_k.intersection(baseline_top_k))
                        overlap_ratio = overlap / k
                        top_k_overlaps[f'top_{k}'] = {
                            'overlap_count': overlap,
                            'overlap_ratio': overlap_ratio,
                            'jaccard_similarity': overlap / len(unified_top_k.union(baseline_top_k))
                        }
                
                comparison_stats[method_name] = {
                    'spearman_correlation': spear_corr,
                    'spearman_p_value': spear_p,
                    'kendall_correlation': kendall_corr,
                    'kendall_p_value': kendall_p,
                    'top_k_overlaps': top_k_overlaps,
                    'ranking_similarity': 'High' if abs(spear_corr) > 0.7 
                                        else 'Medium' if abs(spear_corr) > 0.4 
                                        else 'Low'
                }
        
        # Identify best correlating baselines
        successful_methods = {k: v for k, v in comparison_stats.items() 
                            if not np.isnan(v['spearman_correlation'])}
        
        if successful_methods:
            best_correlation = max(successful_methods.items(), 
                                 key=lambda x: abs(x[1]['spearman_correlation']))
            worst_correlation = min(successful_methods.items(), 
                                  key=lambda x: abs(x[1]['spearman_correlation']))
        else:
            best_correlation = worst_correlation = (None, None)
        
        return {
            'baseline_results': baseline_results,
            'comparison_statistics': comparison_stats,
            'summary': {
                'total_baselines_tested': len(self.baseline_methods),
                'successful_baselines': sum(1 for r in baseline_results.values() if r['success']),
                'best_correlating_method': best_correlation[0] if best_correlation[0] else None,
                'best_correlation_value': best_correlation[1]['spearman_correlation'] if best_correlation[1] else None,
                'worst_correlating_method': worst_correlation[0] if worst_correlation[0] else None,
                'worst_correlation_value': worst_correlation[1]['spearman_correlation'] if worst_correlation[1] else None,
                'average_correlation': np.mean([stats['spearman_correlation'] 
                                              for stats in comparison_stats.values() 
                                              if not np.isnan(stats['spearman_correlation'])]) if comparison_stats else 0
            }
        }
    
    def generate_baseline_comparison_report(self, 
                                          comparison_results: Dict,
                                          save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive baseline comparison report
        
        Args:
            comparison_results: Results from compare_all_baselines()
            save_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE BASELINE COMPARISON REPORT")
        report_lines.append("="*80)
        
        summary = comparison_results['summary']
        
        # Summary statistics
        report_lines.append(f"\nSUMMARY:")
        report_lines.append(f"Total baselines tested: {summary['total_baselines_tested']}")
        report_lines.append(f"Successful baselines: {summary['successful_baselines']}")
        report_lines.append(f"Average correlation with unified method: {summary['average_correlation']:.4f}")
        
        # Best and worst performing baselines
        if summary['best_correlating_method']:
            report_lines.append(f"\nBest correlating method: {summary['best_correlating_method']}")
            report_lines.append(f"  Correlation: {summary['best_correlation_value']:.4f}")
        
        if summary['worst_correlating_method']:
            report_lines.append(f"Worst correlating method: {summary['worst_correlating_method']}")
            report_lines.append(f"  Correlation: {summary['worst_correlation_value']:.4f}")
        
        # Detailed comparison statistics
        report_lines.append(f"\nDETAILED COMPARISON STATISTICS:")
        report_lines.append(f"{'Method':<20} {'Spearman ρ':<12} {'p-value':<10} {'Kendall τ':<12} {'p-value':<10} {'Similarity':<10}")
        report_lines.append("-" * 80)
        
        comparison_stats = comparison_results['comparison_statistics']
        for method_name, stats in comparison_stats.items():
            spear_corr = stats['spearman_correlation']
            spear_p = stats['spearman_p_value']
            kendall_corr = stats['kendall_correlation']
            kendall_p = stats['kendall_p_value']
            similarity = stats['ranking_similarity']
            
            report_lines.append(f"{method_name:<20} {spear_corr:>11.4f} {spear_p:>9.3e} "
                              f"{kendall_corr:>11.4f} {kendall_p:>9.3e} {similarity:<10}")
        
        # Top-k overlap analysis
        report_lines.append(f"\nTOP-K OVERLAP ANALYSIS:")
        for method_name, stats in comparison_stats.items():
            if 'top_k_overlaps' in stats:
                report_lines.append(f"\n{method_name}:")
                for k_name, overlap_stats in stats['top_k_overlaps'].items():
                    overlap_ratio = overlap_stats['overlap_ratio']
                    jaccard_sim = overlap_stats['jaccard_similarity']
                    report_lines.append(f"  {k_name}: {overlap_ratio:.2%} overlap, "
                                      f"Jaccard similarity: {jaccard_sim:.4f}")
        
        # Method success/failure summary
        baseline_results = comparison_results['baseline_results']
        report_lines.append(f"\nMETHOD STATUS:")
        for method_name, results in baseline_results.items():
            status = "✅ SUCCESS" if results['success'] else "❌ FAILED"
            report_lines.append(f"{method_name:<20}: {status}")
            if not results['success'] and 'error' in results:
                report_lines.append(f"  Error: {results['error']}")
        
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Baseline comparison report saved to: {save_path}")
        
        return report


# Example usage and testing
def main():
    """
    Test baseline comparison framework
    """
    print("Testing Baseline Comparison Framework")
    print("="*60)
    
    # Create sample data
    num_nodes = 30
    num_features = 8
    
    # Simple graph structure
    from torch_geometric.data import Data
    
    x = torch.randn(num_nodes, num_features)
    edge_list = []
    
    # Ring + random connections
    for i in range(num_nodes):
        edge_list.append([i, (i+1) % num_nodes])
        edge_list.append([(i+1) % num_nodes, i])
        
        # Add random connections
        if np.random.random() > 0.7:
            j = np.random.randint(0, num_nodes)
            if i != j:
                edge_list.append([i, j])
                edge_list.append([j, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates
    
    data = Data(x=x, edge_index=edge_index)
    
    print(f"Test data: {num_nodes} nodes, {edge_index.shape[1]} edges")
    
    # Create dummy model (for testing gradient-based methods)
    class DummyModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, 1)
            
        def forward(self, x, edge_index, batch=None):
            h = torch.relu(self.linear(x))
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long)
            # Simple pooling
            graph_repr = torch.zeros(batch.max().item() + 1, h.size(1))
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                if mask.sum() > 0:
                    graph_repr[i] = h[mask].mean(dim=0)
            return self.output(graph_repr)
    
    model = DummyModel(num_features)
    model.eval()
    
    # Create comparator
    comparator = BaselineComparator()
    
    # Generate unified importance scores (dummy)
    unified_scores = np.random.random(num_nodes)
    
    # Run comparison
    try:
        results = comparator.compare_all_baselines(
            model=model,
            data=data,
            unified_scores=unified_scores,
            node_names=[f"Node_{i}" for i in range(num_nodes)]
        )
        
        # Generate report
        report = comparator.generate_baseline_comparison_report(results)
        print(report)
        
        print(f"\n✅ Baseline comparison framework test completed successfully!")
        
    except Exception as e:
        print(f"❌ Baseline comparison test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()