import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.data import Data
from explainers.explainer_regression import GNNExplainerRegression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_explainer_sparsified_graph(pipeline, model, target_idx=0, importance_threshold=0.2,  # Relaxed from 0.3 to 0.2
                                     use_node_pruning=True, use_attention_pruning=None, target_name=None):
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
    
    # Use unified pruning method that combines edge and attention importance
    print(f"DEBUG: Using UNIFIED pruning method (combines edge + attention importance for ALL models)")
    print(f"DEBUG: Model type: {type(model).__name__}")
    
    if use_node_pruning:
        return create_unified_pruned_graph_pipeline(pipeline, explainer, model, importance_threshold, combined_edge_importance, target_name)
    else:
        print("DEBUG: Using EDGE-based pruning only")
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


def create_unified_pruned_graph_pipeline(pipeline, explainer, model, importance_threshold, combined_edge_importance, target_name=None):
    """
    UNIFIED node pruning method that combines edge importance + attention scores
    
    This method:
    1. Always uses edge-based importance (from GNNExplainer) as baseline
    2. Attempts to extract attention scores (model-specific)  
    3. Combines both scores intelligently based on quality
    4. Uses adaptive thresholding to guarantee actual pruning
    5. Works consistently for ALL model types
    
    Args:
        pipeline: Pipeline instance
        explainer: GNNExplainer instance
        model: Trained model
        importance_threshold: Base threshold (will be adapted)
        combined_edge_importance: Edge importance matrix from GNNExplainer
        target_name: Target name for saving results
    """
    from torch_geometric.data import Data
    import numpy as np
    import torch
    
    print(f"\n{'='*60}")
    print("UNIFIED NODE PRUNING (Edge + Attention)")
    print(f"{'='*60}")
    
    # Use first sample as template
    template_data = pipeline.dataset.data_list[0]
    
    # Step 1: Get edge-based node importance (always available)
    print("Step 1: Computing edge-based node importance...")
    edge_node_importance = explainer.compute_node_importance_from_edges(combined_edge_importance)
    print(f"Edge importance range: {edge_node_importance.min():.4f} - {edge_node_importance.max():.4f}")
    
    # Step 2: Try to get attention-based importance  
    print("Step 2: Extracting attention scores...")
    try:
        attention_scores = explainer.extract_universal_attention_scores(model, template_data, pipeline.dataset.node_feature_names)
        if attention_scores is not None:
            attention_std = np.std(attention_scores)
            print(f"Attention scores range: {attention_scores.min():.4f} - {attention_scores.max():.4f} (std: {attention_std:.4f})")
        else:
            attention_scores = np.ones(len(edge_node_importance))
            attention_std = 0.0
            print("Attention extraction failed, using uniform scores")
    except Exception as e:
        print(f"Attention extraction error: {e}")
        attention_scores = np.ones(len(edge_node_importance))
        attention_std = 0.0
    
    # Step 3: Add network topology analysis
    print("Step 3: Computing network topology scores...")
    try:
        from utils.network_topology_analysis import NetworkTopologyAnalyzer
        
        # Initialize topology analyzer
        topology_analyzer = NetworkTopologyAnalyzer(device=str(pipeline.dataset.data_list[0].x.device))
        
        # Analyze network topology
        topology_scores, network_properties = topology_analyzer.analyze_network_topology(template_data)
        
        print(f"Topology scores range: {topology_scores.min():.4f} - {topology_scores.max():.4f}")
        print(f"Network modularity: {network_properties.get('modularity', 0.0):.3f}")
        print(f"Network clustering: {network_properties.get('clustering_coefficient', 0.0):.3f}")
        
    except Exception as e:
        print(f"Warning: Topology analysis failed: {e}")
        topology_scores = np.ones(len(edge_node_importance)) * 0.5
        network_properties = {}
    
    # Step 4: Intelligently combine edge, attention, and topology importance
    print("Step 4: Combining importance scores (Edge + Attention + Topology)...")
    
    if attention_std > 0.1:
        # Good attention scores - combine all three
        print("Good attention variance detected, combining edge + attention + topology scores")
        # Normalize all to [0,1] range  
        edge_norm = (edge_node_importance - edge_node_importance.min()) / (edge_node_importance.max() - edge_node_importance.min() + 1e-8)
        attention_norm = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
        topology_norm = (topology_scores - topology_scores.min()) / (topology_scores.max() - topology_scores.min() + 1e-8)
        
        # Weighted combination: 40% edge + 30% attention + 30% topology
        combined_importance = 0.4 * edge_norm + 0.3 * attention_norm + 0.3 * topology_norm
        print("Using weighted combination: 40% edge + 30% attention + 30% topology")
    else:
        # Poor attention scores - combine edge + topology
        print("Poor attention variance detected, combining edge + topology scores")
        edge_norm = (edge_node_importance - edge_node_importance.min()) / (edge_node_importance.max() - edge_node_importance.min() + 1e-8)
        topology_norm = (topology_scores - topology_scores.min()) / (topology_scores.max() - topology_scores.min() + 1e-8)
        
        # Weighted combination: 60% edge + 40% topology
        combined_importance = 0.6 * edge_norm + 0.4 * topology_norm
        print("Using weighted combination: 60% edge + 40% topology")
    
    # Step 5: Adaptive thresholding based on network properties
    print("Step 5: Applying network-aware adaptive thresholding...")
    
    # Use topology analyzer for adaptive thresholding if available
    if 'topology_analyzer' in locals():
        try:
            adaptive_threshold = topology_analyzer.determine_adaptive_threshold(
                combined_importance, network_properties, importance_threshold
            )
            print(f"Network-aware adaptive threshold: {adaptive_threshold:.4f}")
        except Exception as e:
            print(f"Warning: Network-aware thresholding failed: {e}")
            # Fallback to percentile-based thresholding (relaxed)
            target_retention = 0.75  # Keep 75% of nodes (relaxed from 55%)
            adaptive_threshold = np.percentile(combined_importance, (1 - target_retention) * 100)
            print(f"Fallback threshold (75% retention): {adaptive_threshold:.4f}")
    else:
        # Standard percentile-based thresholding (relaxed)
        target_retention = 0.75  # Keep 75% of nodes (relaxed from 55%)
        adaptive_threshold = np.percentile(combined_importance, (1 - target_retention) * 100)
        print(f"Standard threshold (75% retention): {adaptive_threshold:.4f}")
    
    print(f"Combined importance range: {combined_importance.min():.4f} - {combined_importance.max():.4f}")
    print(f"Original threshold: {importance_threshold:.4f}")
    print(f"Final adaptive threshold: {adaptive_threshold:.4f}")

    # Save combined importance scores for analysis
    import pandas as pd
    import os
    if target_name:
        combined_importance_filename = f"combined_importance_scores_{target_name.replace('-', '_')}.csv"
    else:
        combined_importance_filename = "combined_importance_scores.csv"
    combined_importance_path = os.path.join(pipeline.save_dir, combined_importance_filename)

    combined_df = pd.DataFrame({
        'node_name': pipeline.dataset.node_feature_names,
        'edge_importance': edge_node_importance,
        'attention_score': attention_scores,
        'topology_score': topology_scores,
        'combined_importance': combined_importance,
        'above_adaptive_threshold': combined_importance > adaptive_threshold
    })
    combined_df = combined_df.sort_values('combined_importance', ascending=False)
    combined_df.to_csv(combined_importance_path, index=False)
    print(f"💾 Combined importance scores saved to: {combined_importance_path}")
    
    # Use the more aggressive threshold
    final_threshold = max(adaptive_threshold, importance_threshold)
    print(f"Final threshold: {final_threshold:.4f}")
    
    # Step 5: Get nodes above threshold + handle protected nodes
    important_nodes_mask = combined_importance > final_threshold
    important_nodes = np.where(important_nodes_mask)[0]
    
    # Handle protected nodes - always include methanogenic families
    protected_node_names = ['Methanobacteriaceae', 'Methanosaetaceae', 'Methanoregulaceae', 'Methanospirillaceae']
    protected_indices = set()
    
    for protected_name in protected_node_names:
        if protected_name in pipeline.dataset.node_feature_names:
            idx = pipeline.dataset.node_feature_names.index(protected_name)
            protected_indices.add(idx)
            print(f"Protected node '{protected_name}' found at index {idx}")
    
    # Combine threshold-based selection with protected nodes
    important_nodes_set = set(important_nodes) | protected_indices
    
    # If too few nodes, keep top nodes by importance (more conservative minimum)
    min_nodes = max(8, int(0.4 * len(pipeline.dataset.node_feature_names)))  # At least 40% of original nodes or minimum 8 (relaxed)
    if len(important_nodes_set) < min_nodes:
        print(f"Only {len(important_nodes_set)} nodes exceed threshold, ensuring minimum {min_nodes} nodes")
        sorted_indices = np.argsort(combined_importance)[::-1]  # Sort by importance descending
        for idx in sorted_indices:
            if len(important_nodes_set) >= min_nodes:
                break
            important_nodes_set.add(idx)
    
    kept_nodes = np.array(sorted(important_nodes_set), dtype=np.int64)
    pruned_node_names = [pipeline.dataset.node_feature_names[i] for i in kept_nodes]
    
    print(f"Pruning results:")
    print(f"  Original nodes: {template_data.x.shape[0]}")
    print(f"  Pruned nodes: {len(kept_nodes)} ({len(kept_nodes)/template_data.x.shape[0]*100:.1f}% retained)")
    
    # Step 6: Apply pruning to all samples
    print("Step 6: Applying pruning to all samples...")
    
    new_data_list = []
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_nodes)}
    
    for i, original_data in enumerate(pipeline.dataset.data_list):
        # Apply same node selection to this sample
        pruned_x = original_data.x[kept_nodes]
        
        # Filter edges - keep only edges between kept nodes
        new_edges = []
        for j in range(original_data.edge_index.shape[1]):
            u, v = original_data.edge_index[0, j].item(), original_data.edge_index[1, j].item()
            if u in old_to_new and v in old_to_new:
                new_u, new_v = old_to_new[u], old_to_new[v]
                new_edges.append([new_u, new_v])
        
        if len(new_edges) > 0:
            pruned_edge_index = torch.tensor(new_edges, dtype=torch.long).T
        else:
            pruned_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create new data object with completely detached tensors
        # Preserve edge weights and types from original data
        pruned_edge_weight = None
        pruned_edge_type = None
        
        if len(new_edges) > 0:
            # Extract corresponding edge weights and types for kept edges
            pruned_edge_weight = []
            pruned_edge_type = []
            
            for j in range(original_data.edge_index.shape[1]):
                u, v = original_data.edge_index[0, j].item(), original_data.edge_index[1, j].item()
                if u in old_to_new and v in old_to_new:
                    # Get original edge weight and type
                    if hasattr(original_data, 'edge_weight') and original_data.edge_weight is not None:
                        pruned_edge_weight.append(original_data.edge_weight[j].item())
                    else:
                        pruned_edge_weight.append(1.0)  # Default weight
                    
                    if hasattr(original_data, 'edge_type') and original_data.edge_type is not None:
                        pruned_edge_type.append(original_data.edge_type[j].item())
                    else:
                        pruned_edge_type.append(1)  # Default type (positive)
            
            pruned_edge_weight = torch.tensor(pruned_edge_weight, dtype=torch.float32)
            pruned_edge_type = torch.tensor(pruned_edge_type, dtype=torch.long)
        else:
            # Empty graph case
            pruned_edge_weight = torch.empty((0,), dtype=torch.float32)
            pruned_edge_type = torch.empty((0,), dtype=torch.long)
        
        pruned_sample = Data(
            x=pruned_x.detach().clone() if pruned_x.requires_grad else pruned_x.clone(),
            edge_index=pruned_edge_index.detach().clone() if pruned_edge_index.requires_grad else pruned_edge_index.clone(),
            edge_weight=pruned_edge_weight,
            edge_type=pruned_edge_type,
            y=original_data.y.detach().clone() if original_data.y.requires_grad else original_data.y.clone()
        )
        
        new_data_list.append(pruned_sample)
    
    # Update dataset node names
    pipeline.dataset.node_feature_names = pruned_node_names
    
    # Save combined importance scores for analysis
    if target_name:
        importance_filename = f"unified_importance_{target_name.replace('-', '_')}.csv"
    else:
        importance_filename = "unified_importance.csv"
    
    import pandas as pd
    
    # Ensure all arrays have the same length
    num_original_nodes = len(pipeline.dataset.node_feature_names)
    original_node_names = pipeline.dataset.node_feature_names[:num_original_nodes]
    
    # Truncate or pad arrays to match node names length
    edge_imp_adj = edge_node_importance[:num_original_nodes] if len(edge_node_importance) >= num_original_nodes else np.pad(edge_node_importance, (0, num_original_nodes - len(edge_node_importance)), 'constant')
    attention_adj = attention_scores[:num_original_nodes] if len(attention_scores) >= num_original_nodes else np.pad(attention_scores, (0, num_original_nodes - len(attention_scores)), 'constant') 
    combined_adj = combined_importance[:num_original_nodes] if len(combined_importance) >= num_original_nodes else np.pad(combined_importance, (0, num_original_nodes - len(combined_importance)), 'constant')
    
    importance_df = pd.DataFrame({
        'node_name': original_node_names,
        'edge_importance': edge_imp_adj,
        'attention_score': attention_adj,
        'combined_importance': combined_adj,
        'kept': [i in kept_nodes for i in range(num_original_nodes)]
    })
    importance_df.to_csv(f"{pipeline.save_dir}/{importance_filename}", index=False)
    print(f"Unified importance scores saved to: {importance_filename}")
    
    print(f"✓ UNIFIED pruning completed successfully!")
    
    # Store for visualization using the first pruned sample
    if len(new_data_list) > 0:
        first_pruned_sample = new_data_list[0]
        pipeline.dataset.explainer_sparsified_graph_data = {
            'edge_index': first_pruned_sample.edge_index.clone(),
            'edge_weight': getattr(first_pruned_sample, 'edge_weight', torch.ones(first_pruned_sample.edge_index.shape[1])),
            'edge_type': getattr(first_pruned_sample, 'edge_type', torch.ones(first_pruned_sample.edge_index.shape[1], dtype=torch.long)),
            'pruning_type': 'unified_node_based',
            'kept_nodes': kept_nodes,
            'pruned_node_names': pruned_node_names
        }
        print(f"Explainer graph data stored for visualization: {first_pruned_sample.edge_index.shape[1]} edges")
    else:
        print("WARNING: No pruned samples generated, cannot store explainer graph data")
    
    # Visualize graphs
    pipeline.dataset.visualize_graphs(save_dir=f"{pipeline.save_dir}/graphs")
    
    return new_data_list