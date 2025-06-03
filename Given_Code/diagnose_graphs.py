#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path to import modules
sys.path.append('.')

from dataset_regression import MicrobialGNNDataset

def diagnose_graph_construction():
    """Diagnose graph construction issues with strict vs relaxed filtering"""
    
    # Assuming the data file is in the parent directory
    data_path = "../Updated_ASD_data_bacteria_analysis_results_all_samples_analysis.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        # Try alternative path
        data_path = "Updated_ASD_data_bacteria_analysis_results_all_samples_analysis.csv"
        if not os.path.exists(data_path):
            print("Data file not found. Please check the path.")
            return
    
    print("="*80)
    print("GRAPH CONSTRUCTION DIAGNOSIS")
    print("="*80)
    
    for filter_mode in ['strict', 'relaxed']:
        print(f"\n{'-'*60}")
        print(f"Testing family_filter_mode: {filter_mode}")
        print(f"{'-'*60}")
        
        try:
            # Create dataset with current filter mode
            dataset = MicrobialGNNDataset(
                data_path=data_path,
                k_neighbors=5,
                mantel_threshold=0.05,
                use_fast_correlation=True,
                graph_mode='family',
                family_filter_mode=filter_mode
            )
            
            print(f"\nDataset created successfully!")
            print(f"Number of family nodes: {len(dataset.node_feature_names)}")
            print(f"Number of samples: {len(dataset.data_list)}")
            print(f"Target variables: {dataset.target_names}")
            
            # Check edge statistics
            if hasattr(dataset, 'full_edge_index') and dataset.full_edge_index is not None:
                full_edges = dataset.full_edge_index.shape[1] // 2  # Undirected edges
                print(f"Full graph edges: {full_edges}")
            else:
                print("Full graph edges: None or 0")
            
            if hasattr(dataset, 'edge_index') and dataset.edge_index is not None:
                knn_edges = dataset.edge_index.shape[1] // 2  # Undirected edges
                print(f"KNN graph edges: {knn_edges}")
            else:
                print("KNN graph edges: None or 0")
            
            # Check correlation matrix properties
            if hasattr(dataset, 'feature_matrix') and dataset.feature_matrix is not None:
                correlation_matrix = np.corrcoef(dataset.feature_matrix)
                correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
                
                # Remove diagonal (self-correlations)
                np.fill_diagonal(correlation_matrix, 0.0)
                
                print(f"\nCorrelation matrix stats:")
                print(f"  Shape: {correlation_matrix.shape}")
                print(f"  Min correlation: {correlation_matrix.min():.4f}")
                print(f"  Max correlation: {correlation_matrix.max():.4f}")
                print(f"  Mean correlation: {correlation_matrix.mean():.4f}")
                print(f"  Std correlation: {correlation_matrix.std():.4f}")
                
                # Check how many correlations exceed different thresholds
                thresholds = [0.3, 0.5, 0.7]
                for thresh in thresholds:
                    count = np.sum(np.abs(correlation_matrix) > thresh) // 2  # Divide by 2 for undirected
                    print(f"  Correlations > {thresh}: {count}")
            
            # Create a small visualization of the graph
            dataset.visualize_graphs(save_dir=f'diagnosis_{filter_mode}')
            print(f"Graph visualization saved to diagnosis_{filter_mode}/")
            
        except Exception as e:
            print(f"Error creating dataset with {filter_mode} filtering: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    diagnose_graph_construction() 