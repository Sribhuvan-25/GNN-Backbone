import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.data import Data
from explainer_regression import GNNExplainerRegression

# Set device to CPU
device = torch.device('cpu')

def create_explainer_sparsified_graph(pipeline, model, target_idx=0, importance_threshold=0.3):
    """
    Create a sparsified graph based on GNNExplainer results
    
    Args:
        pipeline: RegressionPipeline instance
        model: Trained GNN model
        target_idx: Index of the target variable to explain
        importance_threshold: Threshold for edge importance
        
    Returns:
        List of sparsified graph data objects
    """
    print(f"\nCreating explainer-based sparsified graph...")
    print(f"Graph mode: {pipeline.graph_mode}")
    print(f"Number of nodes: {len(pipeline.dataset.node_feature_names)}")
    print(f"Original importance threshold: {importance_threshold}")
    
    # Initialize explainer
    explainer = GNNExplainerRegression(model, device)
    
    # Create a combined edge importance matrix from multiple samples
    num_explain = min(10, len(pipeline.dataset.data_list))  # Use up to 10 samples
    combined_edge_importance = torch.zeros((len(pipeline.dataset.node_feature_names), len(pipeline.dataset.node_feature_names)), device=device)
    
    importance_matrices = []
    
    for i in range(num_explain):
        # Get sample data and ensure it's on the correct device
        data = pipeline.dataset.data_list[i].to(device)
        
        # Generate explanation
        edge_importance_matrix, _ = explainer.explain_graph(
            data, 
            node_names=pipeline.dataset.node_feature_names,
            target_idx=target_idx
        )
        
        importance_matrices.append(edge_importance_matrix)
        # Add to combined matrix
        combined_edge_importance += edge_importance_matrix
    
    # Average the importance (could also use median for robustness)
    combined_edge_importance /= num_explain
    
    # Add diagnostics
    print(f"Edge importance statistics:")
    print(f"  Min: {combined_edge_importance.min():.6f}")
    print(f"  Max: {combined_edge_importance.max():.6f}")
    print(f"  Mean: {combined_edge_importance.mean():.6f}")
    print(f"  Std: {combined_edge_importance.std():.6f}")
    
    # Count non-zero importance values
    non_zero_importance = combined_edge_importance[combined_edge_importance > 0]
    print(f"  Non-zero values: {len(non_zero_importance)}")
    if len(non_zero_importance) > 0:
        print(f"  Non-zero min: {non_zero_importance.min():.6f}")
        print(f"  Non-zero max: {non_zero_importance.max():.6f}")
        print(f"  Non-zero mean: {non_zero_importance.mean():.6f}")
    
    # Adaptive thresholding based on graph mode and data
    if pipeline.graph_mode == 'family':
        # For family mode, use much lower threshold or percentile-based selection
        print("Using adaptive thresholding for family mode...")
        
        if len(non_zero_importance) > 0:
            # Simple mapping: importance_threshold directly controls percentage of edges to keep
            # 0.1 -> keep 10%, 0.2 -> keep 20%, 0.3 -> keep 30%, etc.
            percentage_to_keep = importance_threshold * 100
            top_percentile = max(5, min(95, 100 - percentage_to_keep))  # Convert to percentile
            threshold_value = torch.quantile(non_zero_importance, top_percentile / 100.0).item()
            print(f"  Keeping top {percentage_to_keep:.0f}% of edges (using {top_percentile:.0f}th percentile)")
            print(f"  Threshold value: {threshold_value:.6f}")
        else:
            # Fallback to very low absolute threshold
            threshold_value = importance_threshold * 0.1
            print(f"  Using low absolute threshold: {threshold_value:.6f}")
    else:
        # For OTU mode, use original threshold but potentially lower it
        if len(non_zero_importance) > 0:
            max_importance = non_zero_importance.max().item()
            if max_importance < importance_threshold:
                threshold_value = max_importance * 0.5  # Use 50% of max importance
                print(f"  Max importance ({max_importance:.6f}) < threshold, using {threshold_value:.6f}")
            else:
                threshold_value = importance_threshold
        else:
            threshold_value = importance_threshold
    
    # Create sparsified adjacency matrix by thresholding
    adj_matrix = combined_edge_importance.clone()
    adj_matrix[adj_matrix < threshold_value] = 0
    
    # Count edges after thresholding
    edges_after_threshold = (adj_matrix > 0).sum().item()
    print(f"Edges after thresholding: {edges_after_threshold}")
    
    # If still no edges, try even more aggressive selection
    if edges_after_threshold == 0 and len(non_zero_importance) > 0:
        print("No edges after thresholding, using top-k selection...")
        # Select top k edges (k = min(num_nodes, 20))
        k = min(len(pipeline.dataset.node_feature_names), 20)
        flat_importance = combined_edge_importance.flatten()
        top_k_values, top_k_indices = torch.topk(flat_importance, k)
        
        # Create new adjacency matrix with only top-k edges
        adj_matrix = torch.zeros_like(combined_edge_importance)
        for i, idx in enumerate(top_k_indices):
            if top_k_values[i] > 0:  # Only add if importance > 0
                row = idx // adj_matrix.shape[1]
                col = idx % adj_matrix.shape[1]
                adj_matrix[row, col] = top_k_values[i]
        
        edges_after_top_k = (adj_matrix > 0).sum().item()
        print(f"Edges after top-{k} selection: {edges_after_top_k}")
    
    # Convert adjacency matrix to edge index and edge weight format
    num_nodes = len(pipeline.dataset.node_feature_names)
    new_edge_index = []
    new_edge_weight = []
    new_edge_type = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                new_edge_index.append([i, j])
                new_edge_weight.append(adj_matrix[i, j].item())
                
                # Determine edge type (sign of correlation)
                corr, _ = pearsonr(pipeline.dataset.feature_matrix[i], pipeline.dataset.feature_matrix[j])
                new_edge_type.append(1 if corr > 0 else 0)
    
    # Handle case when no edges meet the threshold
    if len(new_edge_index) == 0:
        print("Warning: No edges meet the importance threshold. Creating a graph with no edges.")
        # Create empty tensors with proper shape
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_weight = torch.empty((0,), dtype=torch.float32)
        new_edge_type = torch.empty((0,), dtype=torch.long)
        num_edges = 0
    else:
        new_edge_index = torch.tensor(new_edge_index, device=device).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32, device=device)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long, device=device)
        num_edges = new_edge_index.shape[1] if new_edge_index.dim() > 1 else 0
    
    print(f"Explainer sparsified graph has {num_edges//2} undirected edges")
    
    # Store sparsified graph data for visualization
    pipeline.dataset.explainer_sparsified_graph_data = {
        'edge_index': new_edge_index.clone(),
        'edge_weight': new_edge_weight.clone(),
        'edge_type': new_edge_type.clone()
    }
    
    # Create new data objects with sparsified graph
    new_data_list = []
    feature_matrix_samples = pipeline.dataset.feature_matrix.T
    
    for s in range(feature_matrix_samples.shape[0]):
        x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32, device=device).view(-1, 1)
        targets = torch.tensor(pipeline.dataset.target_df.iloc[s].values, dtype=torch.float32, device=device).view(1, -1)
        
        data = Data(
            x=x,
            edge_index=new_edge_index,
            edge_weight=new_edge_weight,
            edge_attr=new_edge_weight.view(-1, 1),
            edge_type=new_edge_type,
            y=targets
        )
        
        new_data_list.append(data)
    
    # Calculate node importance from edge importance matrix
    node_importance = calculate_node_importance(combined_edge_importance, pipeline.dataset.node_feature_names)
    
    # Save node importance ranking
    save_node_importance_ranking(node_importance, f"{pipeline.save_dir}/feature_importance")
    
    # Visualize both original and sparsified graphs
    pipeline.dataset.visualize_graphs(save_dir=f"{pipeline.save_dir}/graphs")
    
    return new_data_list 


def calculate_node_importance(edge_importance_matrix, node_feature_names):
    """
    Calculate node importance based on edge importance matrix.
    
    Args:
        edge_importance_matrix: Square tensor of shape (num_nodes, num_nodes)
        node_feature_names: List of node feature names
        
    Returns:
        List of tuples (node_name, importance_score) sorted by importance
    """
    # Calculate node importance as the sum of all edge weights connected to each node
    # This represents how important each node is in the overall graph structure
    node_importance_scores = []
    
    for i, node_name in enumerate(node_feature_names):
        # Sum of all incoming and outgoing edge weights for this node
        # Since the matrix is symmetric, we can just sum the row and divide by 2
        # to avoid double counting, but we'll sum both for robustness
        incoming_importance = edge_importance_matrix[:, i].sum().item()
        outgoing_importance = edge_importance_matrix[i, :].sum().item()
        
        # Average of incoming and outgoing (should be same for symmetric matrix)
        total_importance = (incoming_importance + outgoing_importance) / 2.0
        
        node_importance_scores.append((node_name, total_importance))
    
    # Sort by importance (descending order)
    node_importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    return node_importance_scores


def save_node_importance_ranking(node_importance_scores, save_dir):
    """
    Save node importance ranking to files.
    
    Args:
        node_importance_scores: List of tuples (node_name, importance_score)
        save_dir: Directory to save the results
    """
    import os
    import pandas as pd
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame for easy saving and visualization
    df = pd.DataFrame(node_importance_scores, columns=['Node_Name', 'Importance_Score'])
    df['Rank'] = range(1, len(df) + 1)
    
    # Save as CSV
    csv_path = f"{save_dir}/node_importance_ranking.csv"
    df.to_csv(csv_path, index=False)
    
    # Save top 10 as separate file for quick reference
    top_10 = df.head(10)
    top_10_path = f"{save_dir}/top_10_important_nodes.csv"
    top_10.to_csv(top_10_path, index=False)
    
    # Save as text file for easy reading
    txt_path = f"{save_dir}/node_importance_ranking.txt"
    with open(txt_path, 'w') as f:
        f.write("Node Importance Ranking (Based on GNNExplainer Edge Importance)\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Rank':<6} {'Node Name':<40} {'Importance Score':<15}\n")
        f.write("-"*70 + "\n")
        
        for rank, (node_name, score) in enumerate(node_importance_scores, 1):
            f.write(f"{rank:<6} {node_name:<40} {score:<15.6f}\n")
    
    print(f"\nNode importance analysis saved:")
    print(f"  Full ranking: {csv_path}")
    print(f"  Top 10: {top_10_path}")
    print(f"  Text summary: {txt_path}")
    
    # Print top 10 to console
    print(f"\nTop 10 Most Important Nodes:")
    print("-"*50)
    for rank, (node_name, score) in enumerate(node_importance_scores[:10], 1):
        print(f"{rank:2}. {node_name:<30} {score:8.4f}")