#!/usr/bin/env python3
"""
Test Correlation-Based Graph Initialization

This script tests the different graph construction methods:
1. Original (current pipeline method)
2. Paper-style correlation (Thapa et al. 2023)
3. Hybrid (paper correlation + k-NN)

Compares graph properties and validates biological relevance.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Import dataset class
from datasets.dataset_regression import MicrobialGNNDataset

def test_graph_construction_methods():
    """Test and compare different graph construction methods"""

    print("="*80)
    print("GRAPH CONSTRUCTION METHOD COMPARISON")
    print("="*80)

    # Data path
    data_path = '../Data/New_Data.csv'

    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return

    print(f"Testing with data: {data_path}")

    # Test parameters
    methods_to_test = [
        ('original', 'Original Pipeline Method'),
        ('paper_correlation', 'Paper-Style Correlation (Thapa et al. 2023)'),
        ('hybrid', 'Hybrid: Paper Correlation + k-NN')
    ]

    results = {}

    for method_key, method_name in methods_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING: {method_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Create dataset with specific graph construction method
            dataset = MicrobialGNNDataset(
                data_path=data_path,
                k_neighbors=10,
                graph_mode='family',
                family_filter_mode='strict',
                graph_construction_method=method_key
            )

            construction_time = time.time() - start_time

            # Collect statistics
            stats = analyze_graph_properties(dataset, method_name, construction_time)
            results[method_key] = stats

            print(f"✅ {method_name} completed successfully!")

        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[method_key] = None

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    compare_methods(results)

    # Create visualizations
    create_comparison_plots(results)

    return results

def analyze_graph_properties(dataset, method_name, construction_time):
    """Analyze properties of the constructed graph"""

    # Basic graph statistics
    n_nodes = len(dataset.node_feature_names)
    n_edges = dataset.edge_index.shape[1] // 2  # Undirected edges

    # Network density
    max_edges = n_nodes * (n_nodes - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0

    # Edge weight statistics
    edge_weights = dataset.edge_weight[::2]  # Every other weight (undirected)

    # Node degree statistics
    node_degrees = torch.zeros(n_nodes)
    for i in range(dataset.edge_index.shape[1]):
        node_degrees[dataset.edge_index[0, i]] += 1

    stats = {
        'method_name': method_name,
        'construction_time': construction_time,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'edge_weight_stats': {
            'mean': float(torch.mean(edge_weights)),
            'std': float(torch.std(edge_weights)),
            'min': float(torch.min(edge_weights)),
            'max': float(torch.max(edge_weights))
        },
        'degree_stats': {
            'mean': float(torch.mean(node_degrees)),
            'std': float(torch.std(node_degrees)),
            'min': float(torch.min(node_degrees)),
            'max': float(torch.max(node_degrees))
        },
        'node_names': dataset.node_feature_names,
        'edge_index': dataset.edge_index,
        'edge_weight': dataset.edge_weight,
        'edge_type': dataset.edge_type
    }

    # Add method-specific metadata if available
    if hasattr(dataset, 'graph_metadata'):
        stats['graph_metadata'] = dataset.graph_metadata

    print(f"\nGraph Statistics for {method_name}:")
    print(f"  Construction time: {construction_time:.2f} seconds")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Density: {density:.4f}")
    print(f"  Mean edge weight: {stats['edge_weight_stats']['mean']:.4f}")
    print(f"  Mean node degree: {stats['degree_stats']['mean']:.2f}")

    return stats

def compare_methods(results):
    """Create comparison table of different methods"""

    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v is not None}

    if not successful_results:
        print("❌ No successful results to compare")
        return

    # Create comparison DataFrame
    comparison_data = []
    for method, stats in successful_results.items():
        comparison_data.append({
            'Method': stats['method_name'],
            'Time (s)': f"{stats['construction_time']:.2f}",
            'Nodes': stats['n_nodes'],
            'Edges': stats['n_edges'],
            'Density': f"{stats['density']:.4f}",
            'Mean Edge Weight': f"{stats['edge_weight_stats']['mean']:.4f}",
            'Mean Degree': f"{stats['degree_stats']['mean']:.2f}",
            'Max Degree': f"{stats['degree_stats']['max']:.0f}"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\nMethod Comparison:")
    print(df_comparison.to_string(index=False))

    # Biological relevance analysis
    print(f"\n{'='*50}")
    print("BIOLOGICAL RELEVANCE ANALYSIS")
    print(f"{'='*50}")

    analyze_biological_relevance(successful_results)

def analyze_biological_relevance(results):
    """Analyze biological relevance of different graph construction methods"""

    # Key microbial families for anaerobic digestion
    important_families = {
        'methanogens': ['Methanobacteriaceae', 'Methanosaetaceae', 'Methanosarcinaceae', 'Methanomicrobiaceae'],
        'fermenters': ['Bacteroidaceae', 'Prevotellaceae', 'Ruminococcaceae', 'Clostridiaceae'],
        'syntrophs': ['Syntrophomonadaceae', 'Syntrophobacteraceae', 'Pelotomaculaceae']
    }

    for method, stats in results.items():
        if stats is None:
            continue

        print(f"\n{stats['method_name']}:")

        node_names = stats['node_names']
        edge_index = stats['edge_index']

        # Calculate node degrees
        node_degrees = torch.zeros(len(node_names))
        for i in range(edge_index.shape[1]):
            node_degrees[edge_index[0, i]] += 1

        # Analyze important families
        for group_name, families in important_families.items():
            found_families = []
            total_degree = 0

            for family in families:
                matches = [i for i, name in enumerate(node_names) if family.lower() in name.lower()]
                for match_idx in matches:
                    found_families.append((node_names[match_idx], node_degrees[match_idx].item()))
                    total_degree += node_degrees[match_idx].item()

            if found_families:
                avg_degree = total_degree / len(found_families)
                print(f"  {group_name.capitalize()}: {len(found_families)} families, avg degree: {avg_degree:.1f}")
                for name, degree in sorted(found_families, key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    - {name}: degree {degree}")
            else:
                print(f"  {group_name.capitalize()}: No families found")

def create_comparison_plots(results):
    """Create visualization comparing different methods"""

    successful_results = {k: v for k, v in results.items() if v is not None}

    if len(successful_results) < 2:
        print("Not enough successful results for comparison plots")
        return

    try:
        # Create output directory
        output_dir = Path('correlation_comparison_plots')
        output_dir.mkdir(exist_ok=True)

        # 1. Graph statistics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        methods = list(successful_results.keys())
        method_names = [successful_results[m]['method_name'] for m in methods]

        # Nodes and edges
        nodes = [successful_results[m]['n_nodes'] for m in methods]
        edges = [successful_results[m]['n_edges'] for m in methods]

        axes[0, 0].bar(method_names, nodes, alpha=0.7)
        axes[0, 0].set_title('Number of Nodes')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(method_names, edges, alpha=0.7)
        axes[0, 1].set_title('Number of Edges')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Density and mean degree
        densities = [successful_results[m]['density'] for m in methods]
        mean_degrees = [successful_results[m]['degree_stats']['mean'] for m in methods]

        axes[1, 0].bar(method_names, densities, alpha=0.7)
        axes[1, 0].set_title('Network Density')
        axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].bar(method_names, mean_degrees, alpha=0.7)
        axes[1, 1].set_title('Mean Node Degree')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'graph_statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Edge weight distributions
        fig, axes = plt.subplots(1, len(successful_results), figsize=(5*len(successful_results), 4))
        if len(successful_results) == 1:
            axes = [axes]

        for i, (method, stats) in enumerate(successful_results.items()):
            edge_weights = stats['edge_weight'][::2].numpy()  # Undirected edges
            axes[i].hist(edge_weights, bins=20, alpha=0.7)
            axes[i].set_title(f"{stats['method_name']}\nEdge Weights")
            axes[i].set_xlabel('Edge Weight')
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(output_dir / 'edge_weight_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Comparison plots saved to: {output_dir}")

    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

def main():
    """Main test function"""
    print("Testing Correlation-Based Graph Initialization")
    print("="*50)

    try:
        results = test_graph_construction_methods()

        print(f"\n🎉 Testing completed!")

        # Summary
        successful = sum(1 for r in results.values() if r is not None)
        total = len(results)

        print(f"Results: {successful}/{total} methods completed successfully")

        if successful >= 2:
            print("✅ Multiple methods available for comparison")
        elif successful == 1:
            print("⚠️ Only one method succeeded - limited comparison possible")
        else:
            print("❌ No methods succeeded - check implementation")

        return results

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()