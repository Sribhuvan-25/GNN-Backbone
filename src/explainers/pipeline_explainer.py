import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.data import Data
from explainers.explainer_regression import GNNExplainerRegression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_explainer_sparsified_graph(pipeline, model, target_idx=0, importance_threshold=0.3, 
                                     use_node_pruning=True, use_attention_pruning=True, target_name=None):
    """
    Create a sparsified graph based on GNNExplainer results with node-based pruning
    
    Args:
        pipeline: RegressionPipeline instance
        model: Trained GNN model
        target_idx: Index of the target variable to explain
        importance_threshold: Threshold for importance (nodes or edges)
        use_node_pruning: Whether to use node-based pruning instead of edge-based pruning
        use_attention_pruning: Whether to use attention-based node pruning (requires GAT model)
        target_name: Name of the target variable for filename generation
        
    Returns:
        List of sparsified graph data objects
    """
    print(f"\nCreating explainer-based sparsified graph...")
    print(f"Graph mode: {pipeline.graph_mode}")
    print(f"Current number of nodes: {len(pipeline.dataset.node_feature_names)}")
    print(f"Original importance threshold: {importance_threshold}")
    
    # Get original node count (before any pruning)
    original_node_count = pipeline.dataset.original_node_count if hasattr(pipeline.dataset, 'original_node_count') and pipeline.dataset.original_node_count is not None else len(pipeline.dataset.node_feature_names)
    print(f"Using original node count for explainer: {original_node_count}")
    
    # Initialize explainer
    explainer = GNNExplainerRegression(model, device)
    
    # Create a combined edge importance matrix from multiple samples using original size
    num_explain = min(10, len(pipeline.dataset.data_list))  # Use up to 10 samples
    combined_edge_importance = torch.zeros((original_node_count, original_node_count), device=device)
    
    importance_matrices = []
    num_processed = 0
    
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
        
        # Handle size mismatch - resize if necessary
        if edge_importance_matrix.size() != combined_edge_importance.size():
            print(f"WARNING: Size mismatch detected - edge_importance_matrix: {edge_importance_matrix.size()}, combined: {combined_edge_importance.size()}")
            
            # If the edge importance matrix is smaller, we might be working with pruned data
            # In this case, skip this sample or pad it
            if edge_importance_matrix.size(0) < combined_edge_importance.size(0):
                print("Edge importance matrix is smaller - likely working with pruned data. Skipping this sample.")
                continue
            elif edge_importance_matrix.size(0) > combined_edge_importance.size(0):
                print("Combined matrix is smaller - resizing combined matrix to match")
                # Resize the combined matrix to match  
                combined_edge_importance = torch.zeros_like(edge_importance_matrix, device=device)
                num_processed = 0  # Reset counter since we're starting over
        
        # Add to combined matrix
        combined_edge_importance += edge_importance_matrix
        num_processed += 1
    
    # Average the importance (could also use median for robustness)
    if num_processed > 0:
        combined_edge_importance /= num_processed
        print(f"Successfully processed {num_processed} out of {num_explain} samples for explainer")
    else:
        print("ERROR: No samples could be processed for explainer")
        return None
    
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
    
    # Generate node importance report and choose pruning method
    print(f"\nUsing {'NODE-BASED' if use_node_pruning else 'EDGE-BASED'} pruning approach")
    
    # Create save path for feature importance report
    import os
    if target_name:
        filename = f"feature_importance_report_{target_name.replace('-', '_')}"
    else:
        filename = "feature_importance_report"
    save_path = os.path.join(pipeline.save_dir, filename)
    
    explainer.get_node_importance(combined_edge_importance, pipeline.dataset.node_feature_names, save_path)
    
    # Choose pruning method
    print(f"DEBUG: Choosing pruning method - use_node_pruning={use_node_pruning}, use_attention_pruning={use_attention_pruning}")
    print(f"DEBUG: Model type: {type(model).__name__}, is_gat_model={is_gat_model(model) if use_attention_pruning else 'N/A'}")
    
    if use_node_pruning:
        if use_attention_pruning:
            print("DEBUG: Using UNIVERSAL ATTENTION-based node pruning (works for ALL GNN types)")
            return create_attention_pruned_graph_pipeline(pipeline, explainer, model, importance_threshold, combined_edge_importance, target_name)
        else:
            print("DEBUG: Using REGULAR node-based pruning with edge importance")
            return create_node_pruned_graph_pipeline(pipeline, explainer, combined_edge_importance, importance_threshold, target_name)
    else:
        print("DEBUG: Using EDGE-based pruning")
        return create_edge_pruned_graph_pipeline(pipeline, explainer, combined_edge_importance, importance_threshold, non_zero_importance)

def create_node_pruned_graph_pipeline(pipeline, explainer, combined_edge_importance, importance_threshold, target_name=None):
    """Create node-pruned graph for the pipeline"""
    
    # Use first sample as template for node pruning
    template_data = pipeline.dataset.data_list[0]
    
    # Create node-pruned version of the template
    pruned_data, kept_nodes, pruned_node_names = explainer.create_node_pruned_graph(
        template_data, 
        combined_edge_importance, 
        pipeline.dataset.node_feature_names,
        importance_threshold=importance_threshold,
        min_nodes=10
    )
    
    # Now create new data list with only the important nodes
    new_data_list = []
    feature_matrix_samples = pipeline.dataset.feature_matrix.T
    
    for s in range(feature_matrix_samples.shape[0]):
        # Only keep features for important nodes
        x = torch.tensor(feature_matrix_samples[s][kept_nodes], dtype=torch.float32).view(-1, 1)
        targets = torch.tensor(pipeline.dataset.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
        
        data = Data(
            x=x,
            edge_index=pruned_data.edge_index.clone(),
            y=targets
        )
        
        new_data_list.append(data)
    
    print(f"Node-pruned graph created with {len(kept_nodes)} nodes and {pruned_data.edge_index.shape[1]} edges")
    
    # Store for visualization with actual edge weights from pruned data
    pipeline.dataset.explainer_sparsified_graph_data = {
        'edge_index': pruned_data.edge_index.clone(),
        'edge_weight': getattr(pruned_data, 'edge_weight', torch.ones(pruned_data.edge_index.shape[1])),
        'edge_type': getattr(pruned_data, 'edge_type', torch.ones(pruned_data.edge_index.shape[1], dtype=torch.long)),
        'pruning_type': 'node_based',
        'kept_nodes': kept_nodes,
        'pruned_node_names': pruned_node_names
    }
    
    # Update pipeline dataset node names to reflect pruning
    pipeline.dataset.node_feature_names = pruned_node_names
    
    # Visualize graphs
    pipeline.dataset.visualize_graphs(save_dir=f"{pipeline.save_dir}/graphs")
    
    return new_data_list

def create_edge_pruned_graph_pipeline(pipeline, explainer, combined_edge_importance, importance_threshold, non_zero_importance):
    """Create edge-pruned graph for the pipeline (original method)"""
    
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
    from scipy.stats import pearsonr
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
        'edge_type': new_edge_type.clone(),
        'pruning_type': 'edge_based'
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

def is_gat_model(model):
    """Check if the model is a GAT (Graph Attention Network) model"""
    for name, module in model.named_modules():
        if 'conv' in name.lower() and 'gat' in type(module).__name__.lower():
            return True
    return False

def create_attention_pruned_graph_pipeline(pipeline, explainer, model, importance_threshold, combined_edge_importance, target_name=None):
    """Create attention-based node-pruned graph for the pipeline"""
    
    print(f"\n{'='*60}")
    print("ATTENTION-BASED NODE PRUNING")
    print(f"{'='*60}")
    
    # Use first sample as template for attention-based node pruning
    template_data = pipeline.dataset.data_list[0]
    
    # Create attention-based node-pruned version of the template
    result = explainer.create_attention_based_node_pruning(
        template_data, 
        model,
        pipeline.dataset.node_feature_names,
        attention_threshold=importance_threshold,
        min_nodes=10,
        protected_nodes=getattr(pipeline.dataset, 'protected_nodes', None)  # Use dataset's protected nodes if available
    )
    
    # Check if attention-based pruning failed
    if result[0] is None:
        print("Attention-based pruning failed, falling back to edge-importance based node pruning")
        # Fall back to regular node pruning using the combined edge importance from explainer
        return create_node_pruned_graph_pipeline(pipeline, explainer, combined_edge_importance, importance_threshold, target_name)
    
    pruned_data, kept_nodes, pruned_node_names, attention_scores = result
    
    # Save attention scores for analysis
    if target_name:
        attention_filename = f"attention_scores_{target_name.replace('-', '_')}.csv"
    else:
        attention_filename = "attention_scores.csv"
    attention_report_path = f"{pipeline.save_dir}/{attention_filename}"
    if pruned_node_names is not None:
        import pandas as pd
        attention_df = pd.DataFrame({
            'node_name': pipeline.dataset.node_feature_names,
            'attention_score': attention_scores,
            'kept': [i in kept_nodes for i in range(len(pipeline.dataset.node_feature_names))]
        })
        attention_df = attention_df.sort_values('attention_score', ascending=False)
        attention_df.to_csv(attention_report_path, index=False)
        print(f"Attention scores saved to: {attention_report_path}")
    
    # Now create new data list with only the important nodes
    new_data_list = []
    feature_matrix_samples = pipeline.dataset.feature_matrix.T
    
    for s in range(feature_matrix_samples.shape[0]):
        # Only keep features for important nodes
        x = torch.tensor(feature_matrix_samples[s][kept_nodes], dtype=torch.float32).view(-1, 1)
        targets = torch.tensor(pipeline.dataset.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
        
        data = Data(
            x=x,
            edge_index=pruned_data.edge_index.clone(),
            y=targets
        )
        
        new_data_list.append(data)
    
    print(f"Attention-pruned graph created with {len(kept_nodes)} nodes and {pruned_data.edge_index.shape[1]} edges")
    
    # Store for visualization with actual edge weights and attention information
    pipeline.dataset.explainer_sparsified_graph_data = {
        'edge_index': pruned_data.edge_index.clone(),
        'edge_weight': getattr(pruned_data, 'edge_weight', torch.ones(pruned_data.edge_index.shape[1])),
        'edge_type': getattr(pruned_data, 'edge_type', torch.ones(pruned_data.edge_index.shape[1], dtype=torch.long)),
        'pruning_type': 'attention_based',
        'kept_nodes': kept_nodes,
        'pruned_node_names': pruned_node_names,
        'attention_scores': attention_scores
    }
    
    # Update pipeline dataset node names to reflect pruning
    pipeline.dataset.node_feature_names = pruned_node_names
    
    # Visualize graphs with attention information
    pipeline.dataset.visualize_graphs(save_dir=f"{pipeline.save_dir}/graphs")
    
    return new_data_list