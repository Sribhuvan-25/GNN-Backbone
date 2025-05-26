import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.data import Data
from explainer_regression import GNNExplainerRegression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # Get sample data
        data = pipeline.dataset.data_list[i]
        
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
            # Use percentile-based selection to keep top edges
            top_percentile = 80  # Keep top 20% of edges
            threshold_value = torch.quantile(non_zero_importance, top_percentile / 100.0).item()
            print(f"  Using {100-top_percentile}th percentile threshold: {threshold_value:.6f}")
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
        new_edge_index = torch.tensor(new_edge_index).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
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
        x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
        targets = torch.tensor(pipeline.dataset.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
        
        data = Data(
            x=x,
            edge_index=new_edge_index,
            edge_weight=new_edge_weight,
            edge_attr=new_edge_weight.view(-1, 1),
            edge_type=new_edge_type,
            y=targets
        )
        
        new_data_list.append(data)
    
    # Visualize both original and sparsified graphs
    pipeline.dataset.visualize_graphs(save_dir=f"{pipeline.save_dir}/graphs")
    
    return new_data_list 