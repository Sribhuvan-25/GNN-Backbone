"""
Visualization utilities for graph rendering and statistical plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_optimal_layout(G, seed=42, scale=2.0):
    """
    Get optimal layout for graph visualization with fallbacks.
    
    Args:
        G: NetworkX graph
        seed: Random seed for reproducibility
        scale: Scale factor for layout spacing
        
    Returns:
        dict: Node positions
    """
    # Try Kamada-Kawai first (best for showing natural clustering)
    try:
        if nx.is_connected(G):
            # Set random seed before calling kamada_kawai_layout (it doesn't accept seed parameter)
            np.random.seed(seed)
            pos = nx.kamada_kawai_layout(G, scale=scale)
            print("Using Kamada-Kawai layout")
            return pos
        else:
            print("Graph is disconnected, using spring layout")
            pos = nx.spring_layout(G, k=scale, iterations=150, seed=seed)
            return pos
    except Exception as e:
        print(f"Kamada-Kawai failed ({e}), falling back to spring layout")
        try:
            pos = nx.spring_layout(G, k=scale, iterations=150, seed=seed)
            return pos
        except Exception as e2:
            print(f"Spring layout failed ({e2}), using circular layout")
            return nx.circular_layout(G, scale=scale)


def adjust_color_brightness(hex_color, factor=0.8):
    """
    Adjust the brightness of a hex color.
    
    Args:
        hex_color: Hex color string (e.g., '#FF5733')
        factor: Brightness factor (0.0-1.0, lower = darker)
        
    Returns:
        str: Adjusted hex color
    """
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Adjust brightness
    r = int(r * factor)
    g = int(g * factor) 
    b = int(b * factor)
    
    # Ensure values stay within bounds
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'


def get_vibrant_node_colors():
    """
    Get vibrant color mapping for functional groups.
    
    Returns:
        dict: Mapping of functional group to color
    """
    return {
        'acetoclastic': '#FF6B35',      # Vibrant orange
        'hydrogenotrophic': '#4ECDC4',  # Turquoise
        'syntrophic': '#9B5DE5',        # Purple
        'other': '#95A5A6',             # Gray for others
        'fermentative': '#F7DC6F',      # Light yellow
        'sulfate_reducing': '#F1948A',   # Light red
        'unknown': '#D5DBDB'            # Light gray
    }


def get_functional_group_colors(node_features, acetoclastic_features, hydrogenotrophic_features, syntrophic_features):
    """
    Assign colors to nodes based on functional groups.
    
    Args:
        node_features: List of node feature names
        acetoclastic_features: List of acetoclastic organism features
        hydrogenotrophic_features: List of hydrogenotrophic organism features
        syntrophic_features: List of syntrophic organism features
        
    Returns:
        dict: Node name to color mapping
    """
    colors = get_vibrant_node_colors()
    node_colors = {}
    
    for node in node_features:
        if node in acetoclastic_features:
            node_colors[node] = colors['acetoclastic']
        elif node in hydrogenotrophic_features:
            node_colors[node] = colors['hydrogenotrophic']
        elif node in syntrophic_features:
            node_colors[node] = colors['syntrophic']
        else:
            node_colors[node] = colors['other']
    
    return node_colors


def create_networkx_graph_from_edge_data(edge_index, edge_weight, node_features):
    """
    Create a NetworkX graph from PyTorch Geometric edge data.

    Args:
        edge_index: Tensor of shape (2, num_edges) containing edge indices
        edge_weight: Tensor of edge weights
        node_features: List of node feature names

    Returns:
        nx.Graph: NetworkX graph object
    """
    G = nx.Graph()

    # For edge-only sparsification, include ALL nodes, not just those with edges
    # This ensures isolated nodes (nodes with no edges) are still shown
    for node_idx in range(len(node_features)):
        node_name = node_features[node_idx]
        G.add_node(node_idx, name=node_name)

    # Add edges with weights
    # Handle None edge_weight gracefully
    if edge_index is not None and edge_index.shape[1] > 0:
        edge_index_np = edge_index.cpu().numpy()
        
        if edge_weight is not None:
            edge_weight_np = edge_weight.cpu().numpy()
        else:
            edge_weight_np = np.ones(edge_index_np.shape[1])  # Default weights

        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[:, i]
            weight = edge_weight_np[i]
            G.add_edge(src, dst, weight=weight)

    return G


def create_legend_for_functional_groups(acetoclastic_features, hydrogenotrophic_features, syntrophic_features):
    """
    Create a legend for functional group colors.
    
    Args:
        acetoclastic_features: List of acetoclastic features
        hydrogenotrophic_features: List of hydrogenotrophic features  
        syntrophic_features: List of syntrophic features
        
    Returns:
        list: List of legend patches
    """
    colors = get_vibrant_node_colors()
    legend_elements = []
    
    if acetoclastic_features:
        legend_elements.append(Patch(facecolor=colors['acetoclastic'], label='Acetoclastic'))
    if hydrogenotrophic_features:
        legend_elements.append(Patch(facecolor=colors['hydrogenotrophic'], label='Hydrogenotrophic'))
    if syntrophic_features:
        legend_elements.append(Patch(facecolor=colors['syntrophic'], label='Syntrophic'))
    
    legend_elements.append(Patch(facecolor=colors['other'], label='Other'))
    
    return legend_elements


def save_graph_visualization(G, node_colors, output_path, title="Graph Visualization", 
                           legend_elements=None, figsize=(15, 12), show_edge_weights=True):
    """
    Save a graph visualization to file with enhanced edge weight representation.
    
    Args:
        G: NetworkX graph
        node_colors: Dictionary mapping node indices to colors
        output_path: Path to save the figure
        title: Plot title
        legend_elements: List of legend elements
        figsize: Figure size tuple
        show_edge_weights: Whether to visually represent edge weights
    """
    plt.figure(figsize=figsize)
    
    # Calculate layout with improved spacing for less cluttered graphs
    pos = get_optimal_layout(G, seed=42, scale=2.0)
    
    # Draw nodes with uniform size as requested
    # Convert node index to node name for color lookup
    node_color_list = []
    for node in G.nodes():
        # Get node name from the graph's node data
        node_name = G.nodes[node].get('name', f'node_{node}')
        color = node_colors.get(node_name, '#B0C4DE')  # Use node name as key
        node_color_list.append(color)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, 
                          node_size=500, alpha=0.9, edgecolors='black', linewidths=1.5)
    
    if show_edge_weights and G.edges():
        # Get edge weights for visualization
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        
        if len(edge_weights) > 0:
            # Normalize edge weights for visualization
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            
            if max_weight > min_weight:
                # Create different edge categories based on weight ranges
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
                
                # Draw all edges with uniform thickness and color
                nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8, edge_color='gray')
                
                # Add edge weight labels (absolute values)
                edge_labels = {}
                for (u, v), weight in zip(G.edges(), edge_weights):
                    edge_labels[(u, v)] = f'{abs(weight):.2f}'
                
                # Draw edge labels with smaller font to reduce clutter
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
            else:
                # All edges have the same weight - draw with uniform style
                nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8, edge_color='gray')
                # Still show the weight values even if they're all the same
                edge_labels = {(u, v): f'{abs(edge_weights[0]):.2f}' for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        else:
            # No edges
            pass
    else:
        # Simple edge drawing without weight visualization
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Add labels (only for smaller graphs to avoid clutter)
    if len(G.nodes()) <= 50:
        node_labels = nx.get_node_attributes(G, 'name')
        if not node_labels:
            node_labels = {i: str(i) for i in G.nodes()}
        
        # Truncate long labels
        truncated_labels = {k: (v[:15] + '...' if len(v) > 18 else v) 
                           for k, v in node_labels.items()}
        
        nx.draw_networkx_labels(G, pos, labels=truncated_labels, font_size=7, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize=9, framealpha=0.9)
    
    # Add graph statistics as text
    stats_text = f"Nodes: {len(G.nodes())}  |  Edges: {len(G.edges())}"
    if G.edges():
        avg_weight = np.mean([G[u][v].get('weight', 1.0) for u, v in G.edges()])
        stats_text += f"  |  Avg. Weight: {avg_weight:.3f}"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, style='italic', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Graph visualization saved: {output_path}")

def create_enhanced_graph_comparison(knn_graph_data, explainer_graph_data, node_features,
                                   output_dir, functional_groups=None, protected_nodes=None, abundance_data=None):
    """
    Create side-by-side comparison of k-NN and explainer graphs with enhanced edge weight visualization.
    Only generates the 4 required files: 3 individual stage graphs + 1 comprehensive comparison.

    Args:
        knn_graph_data: Dictionary with k-NN graph data (edge_index, edge_weight, etc.)
        explainer_graph_data: Dictionary with explainer graph data
        node_features: List of node feature names
        output_dir: Directory to save visualizations
        functional_groups: Dictionary with functional group features for coloring
        protected_nodes: List of protected node names
        abundance_data: Dictionary with abundance data for node sizing
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Only create the three-panel comparison with individual saves
    # This generates exactly 4 files: 3 individual + 1 comprehensive
    create_side_by_side_comparison(
        knn_graph_data, explainer_graph_data, node_features,
        output_dir, functional_groups, protected_nodes, abundance_data
    )

def create_side_by_side_comparison(knn_graph_data, explainer_graph_data, node_features,
                                 output_dir, functional_groups=None, protected_nodes=None, abundance_data=None):
    """Create a three-panel comparison plot: Spearman → k-NN → Attention-Pruned."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # VERIFICATION: Ensure data integrity across all stages
    print("\n🔍 GRAPH DATA INTEGRITY VERIFICATION:")
    print(f"Input node_features: {len(node_features)} nodes")
    print(f"Original node names: {len(knn_graph_data.get('original_node_names', []))} nodes")
    if knn_graph_data.get('edge_index') is not None:
        knn_edges = knn_graph_data['edge_index'].shape[1]
        print(f"k-NN graph: {knn_edges} edges")
    if explainer_graph_data and explainer_graph_data.get('edge_index') is not None:
        explainer_edges = explainer_graph_data['edge_index'].shape[1]
        pruned_nodes = len(explainer_graph_data.get('pruned_node_names', []))
        print(f"Explainer graph: {explainer_edges} edges, {pruned_nodes} pruned nodes")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15))

    # Enhanced color scheme: Protected nodes = pink, others = light blue-gray
    def get_node_colors_with_protection(node_list, protected_list=None):
        colors = []
        for node in node_list:
            if protected_list and node in protected_list:
                colors.append('#FF69B4')  # Hot pink for protected/anchored nodes
            else:
                colors.append('#B0C4DE')  # Light steel blue for others
        return colors

    # Function to calculate node sizes based on abundance with better scaling
    def get_node_sizes_from_abundance(node_list, abundance_dict=None):
        if abundance_dict and len(abundance_dict) > 0:
            sizes = []
            # Get abundance range for better scaling
            abundances = [abundance_dict.get(node, 0.01) for node in node_list]
            min_abundance = min(abundances)
            max_abundance = max(abundances)

            for node in node_list:
                abundance = abundance_dict.get(node, 0.01)
                # Normalize abundance to 0-1 range, then scale more sensitively
                if max_abundance > min_abundance:
                    normalized = (abundance - min_abundance) / (max_abundance - min_abundance)
                    # Use square root to make differences more visible
                    size = 300 + (normalized ** 0.7 * 1200)  # Range: 300-1500
                else:
                    size = 800  # All same abundance
                sizes.append(size)
            return sizes
        else:
            return [800] * len(node_list)  # Default size

    # Function to calculate edge widths with better sensitivity
    def get_edge_widths_from_correlations(graph):
        if not graph.edges():
            return []

        edge_weights = [abs(graph[u][v].get('weight', 0.5)) for u, v in graph.edges()]

        if len(edge_weights) == 0:
            return []

        # Get weight range for better scaling
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)

        edge_widths = []
        for weight in edge_weights:
            if max_weight > min_weight:
                # Normalize to 0-1, then apply more sensitive scaling
                normalized = (weight - min_weight) / (max_weight - min_weight)
                # Use exponential scaling to amplify differences (less aggressive than before)
                width = 0.5 + (normalized ** 1.2 * 4.5)  # Range: 0.5-5.0
            else:
                width = 2.0  # All weights are the same
            edge_widths.append(width)

        return edge_widths
    
    # Panel 1: Spearman Correlation Graph (Original)
    # IMPORTANT: Use original node names for Panel 1, not current node_features which might be pruned
    original_node_names = knn_graph_data.get('original_node_names', node_features)

    if 'original_edge_index' in knn_graph_data:
        # Use original correlation data if available
        original_G = create_networkx_graph_from_edge_data(
            knn_graph_data['original_edge_index'],
            knn_graph_data.get('original_edge_weight', None),
            original_node_names  # Use original names, not current node_features
        )
        print(f"Panel 1 (Spearman): {len(original_G.nodes())} nodes, {len(original_G.edges())} edges")
    else:
        # Fallback to k-NN graph for Panel 1
        original_G = create_networkx_graph_from_edge_data(
            knn_graph_data['edge_index'],
            knn_graph_data.get('edge_weight', None),
            original_node_names  # Use original names, not current node_features
        )
        print(f"Panel 1 (k-NN fallback): {len(original_G.nodes())} nodes, {len(original_G.edges())} edges")

    pos1 = get_optimal_layout(original_G, seed=42, scale=2.0)

    # Get full node names and colors - ensure arrays match graph size
    num_graph_nodes = len(original_G.nodes())
    # IMPORTANT: Get actual node names from the graph, not just first N from original list
    graph_node_features = []
    for node_id in sorted(original_G.nodes()):  # Sort for consistent ordering
        node_name = original_G.nodes[node_id].get('name', f'Node_{node_id}')
        graph_node_features.append(node_name)

    original_node_colors = get_node_colors_with_protection(graph_node_features, protected_nodes)
    original_node_sizes = get_node_sizes_from_abundance(graph_node_features, abundance_data)
    original_edge_widths = get_edge_widths_from_correlations(original_G)

    # Ensure arrays match graph size
    if len(original_node_colors) != num_graph_nodes:
        original_node_colors = original_node_colors[:num_graph_nodes] + ['#B0C4DE'] * max(0, num_graph_nodes - len(original_node_colors))
    if len(original_node_sizes) != num_graph_nodes:
        original_node_sizes = original_node_sizes[:num_graph_nodes] + [800] * max(0, num_graph_nodes - len(original_node_sizes))

    # Draw Panel 1: Spearman Correlation Graph
    nx.draw_networkx_nodes(original_G, pos1, ax=ax1, node_color=original_node_colors,
                          node_size=original_node_sizes, alpha=0.9, edgecolors='black', linewidths=1.5)

    if original_G.edges():
        nx.draw_networkx_edges(original_G, pos1, ax=ax1, alpha=0.6, width=original_edge_widths, edge_color='darkgray')

        # Add edge weight labels
        edge_labels = {(u, v): f'{abs(original_G[u][v].get("weight", 0)):.2f}' for u, v in original_G.edges()}
        nx.draw_networkx_edge_labels(original_G, pos1, edge_labels, ax=ax1, font_size=8)

    # Add full family names as labels - use original_node_names directly by position
    node_labels = {}
    for node_id in original_G.nodes():
        if node_id < len(original_node_names):
            # Wrap long family names for better readability
            family_name = original_node_names[node_id]
            if len(family_name) > 18:
                # Split long names into multiple lines
                words = family_name.split('_')
                if len(words) > 1:
                    mid_point = len(words) // 2
                    line1 = '_'.join(words[:mid_point])
                    line2 = '_'.join(words[mid_point:])
                    family_name = f"{line1}\n{line2}"
                else:
                    # If no underscore, just split at midpoint
                    mid = len(family_name) // 2
                    family_name = f"{family_name[:mid]}\n{family_name[mid:]}"
            node_labels[node_id] = family_name
        else:
            node_labels[node_id] = f'Node_{node_id}'
    nx.draw_networkx_labels(original_G, pos1, labels=node_labels, ax=ax1, font_size=8, font_weight='bold')

    ax1.set_title('Spearman Correlation Graph (Original)', fontsize=16, fontweight='bold', pad=20)
    ax1.text(0.02, 0.98, f"Nodes: {len(original_G.nodes())}\nEdges: {len(original_G.edges())}",
            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Panel 2: k-NN Graph
    # IMPORTANT: Also use original node names for Panel 2, same reason as Panel 1
    knn_G = create_networkx_graph_from_edge_data(
        knn_graph_data['edge_index'],
        knn_graph_data.get('edge_weight', None),
        original_node_names  # Use original names, not current node_features
    )
    print(f"Panel 2 (k-NN): {len(knn_G.nodes())} nodes, {len(knn_G.edges())} edges")
    pos2 = get_optimal_layout(knn_G, seed=42, scale=2.0)

    # Get node data for k-NN graph - ensure arrays match graph size
    num_knn_nodes = len(knn_G.nodes())
    # IMPORTANT: Get actual node names from the k-NN graph, not just first N from original list
    knn_graph_node_features = []
    for node_id in sorted(knn_G.nodes()):  # Sort for consistent ordering
        node_name = knn_G.nodes[node_id].get('name', f'Node_{node_id}')
        knn_graph_node_features.append(node_name)

    knn_node_colors = get_node_colors_with_protection(knn_graph_node_features, protected_nodes)
    knn_node_sizes = get_node_sizes_from_abundance(knn_graph_node_features, abundance_data)
    knn_edge_widths = get_edge_widths_from_correlations(knn_G)

    # Ensure arrays match graph size
    if len(knn_node_colors) != num_knn_nodes:
        knn_node_colors = knn_node_colors[:num_knn_nodes] + ['#B0C4DE'] * max(0, num_knn_nodes - len(knn_node_colors))
    if len(knn_node_sizes) != num_knn_nodes:
        knn_node_sizes = knn_node_sizes[:num_knn_nodes] + [800] * max(0, num_knn_nodes - len(knn_node_sizes))

    # Draw Panel 2: k-NN Graph
    nx.draw_networkx_nodes(knn_G, pos2, ax=ax2, node_color=knn_node_colors,
                          node_size=knn_node_sizes, alpha=0.9, edgecolors='black', linewidths=1.5)

    if knn_G.edges():
        nx.draw_networkx_edges(knn_G, pos2, ax=ax2, alpha=0.6, width=knn_edge_widths, edge_color='darkgray')

        # Add edge weight labels
        edge_labels = {(u, v): f'{abs(knn_G[u][v].get("weight", 0)):.2f}' for u, v in knn_G.edges()}
        nx.draw_networkx_edge_labels(knn_G, pos2, edge_labels, ax=ax2, font_size=8)

    # Add full family names as labels - use original_node_names directly by position
    node_labels = {}
    for node_id in knn_G.nodes():
        if node_id < len(original_node_names):
            # Wrap long family names for better readability
            family_name = original_node_names[node_id]
            if len(family_name) > 18:
                # Split long names into multiple lines
                words = family_name.split('_')
                if len(words) > 1:
                    mid_point = len(words) // 2
                    line1 = '_'.join(words[:mid_point])
                    line2 = '_'.join(words[mid_point:])
                    family_name = f"{line1}\n{line2}"
                else:
                    # If no underscore, just split at midpoint
                    mid = len(family_name) // 2
                    family_name = f"{family_name[:mid]}\n{family_name[mid:]}"
            node_labels[node_id] = family_name
        else:
            node_labels[node_id] = f'Node_{node_id}'
    nx.draw_networkx_labels(knn_G, pos2, labels=node_labels, ax=ax2, font_size=8, font_weight='bold')

    ax2.set_title('k-NN Graph (Sparsified)', fontsize=16, fontweight='bold', pad=20)
    ax2.text(0.02, 0.98, f"Nodes: {len(knn_G.nodes())}\nEdges: {len(knn_G.edges())}",
            transform=ax2.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # Panel 3: Attention-Pruned Graph
    if explainer_graph_data and 'edge_index' in explainer_graph_data:
        # Get the actual pruned node names from explainer data
        explainer_node_names = explainer_graph_data.get('pruned_node_names', node_features)

        # Ensure we have valid node names
        if not explainer_node_names or len(explainer_node_names) == 0:
            print("Warning: No pruned_node_names found, using fallback")
            explainer_node_names = node_features

        explainer_G = create_networkx_graph_from_edge_data(
            explainer_graph_data['edge_index'],
            explainer_graph_data.get('edge_weight', None),
            explainer_node_names
        )
        print(f"Panel 3 (Explainer): {len(explainer_G.nodes())} nodes, {len(explainer_G.edges())} edges")
        pos3 = get_optimal_layout(explainer_G, seed=42, scale=2.0)

        # Get node data for pruned graph - extract actual node names from the explainer graph
        explainer_graph_node_features = []
        for node_id in sorted(explainer_G.nodes()):  # Sort for consistent ordering
            node_name = explainer_G.nodes[node_id].get('name', f'Node_{node_id}')
            explainer_graph_node_features.append(node_name)

        pruned_node_colors = get_node_colors_with_protection(explainer_graph_node_features, protected_nodes)

        # For pruned nodes, need to get abundance data for remaining nodes
        pruned_abundance_data = {}
        if abundance_data:
            for node_name in explainer_graph_node_features:
                if node_name in abundance_data:
                    pruned_abundance_data[node_name] = abundance_data[node_name]

        pruned_node_sizes = get_node_sizes_from_abundance(explainer_graph_node_features, pruned_abundance_data)
        pruned_edge_widths = get_edge_widths_from_correlations(explainer_G)

        # Draw Panel 3: Attention-Pruned Graph - ensure array lengths match
        num_nodes = len(explainer_G.nodes())
        if len(pruned_node_colors) != num_nodes:
            pruned_node_colors = pruned_node_colors[:num_nodes] + ['#B0C4DE'] * max(0, num_nodes - len(pruned_node_colors))
        if len(pruned_node_sizes) != num_nodes:
            pruned_node_sizes = pruned_node_sizes[:num_nodes] + [800] * max(0, num_nodes - len(pruned_node_sizes))

        nx.draw_networkx_nodes(explainer_G, pos3, ax=ax3, node_color=pruned_node_colors,
                              node_size=pruned_node_sizes, alpha=0.9, edgecolors='black', linewidths=1.5)

        if explainer_G.edges():
            # Get all edges with their weights and sort by weight
            edge_weights_list = []
            for u, v in explainer_G.edges():
                weight = abs(explainer_G[u][v].get("weight", 0))
                edge_weights_list.append(((u, v), weight))
            
            # Sort edges by weight (descending) and get top 10
            edge_weights_list.sort(key=lambda x: x[1], reverse=True)
            top_10_edges = [edge for edge, weight in edge_weights_list[:10]]
            regular_edges = [edge for edge, weight in edge_weights_list[10:]]
            
            # Draw regular edges (all except top 10) in gray
            if regular_edges:
                regular_widths = [pruned_edge_widths[list(explainer_G.edges()).index(edge)] 
                                 for edge in regular_edges if edge in explainer_G.edges()]
                nx.draw_networkx_edges(explainer_G, pos3, edgelist=regular_edges, ax=ax3, 
                                      alpha=0.6, width=regular_widths, edge_color='darkgray')
            
            # Draw top 10 edges in orange/gold to highlight them
            if top_10_edges:
                top_widths = [pruned_edge_widths[list(explainer_G.edges()).index(edge)] 
                             for edge in top_10_edges if edge in explainer_G.edges()]
                nx.draw_networkx_edges(explainer_G, pos3, edgelist=top_10_edges, ax=ax3, 
                                      alpha=0.9, width=top_widths, edge_color='#FF8C00')  # Dark orange

            # Add edge weight labels
            edge_labels = {(u, v): f'{abs(explainer_G[u][v].get("weight", 0)):.2f}' for u, v in explainer_G.edges()}
            nx.draw_networkx_edge_labels(explainer_G, pos3, edge_labels, ax=ax3, font_size=8)

        # Add full family names as labels - use stored node names from graph
        node_labels = {}
        for node_id in explainer_G.nodes():
            # Try to get name from graph first, then fall back to explainer_node_names
            node_name = explainer_G.nodes[node_id].get('name', None)
            if not node_name and explainer_node_names:
                # Find index in explainer graph node list
                node_list = list(explainer_G.nodes())
                if node_id in node_list:
                    idx = node_list.index(node_id)
                    if idx < len(explainer_node_names):
                        node_name = explainer_node_names[idx]

            # Wrap long family names for better readability
            family_name = node_name or f"Node_{node_id}"
            if len(family_name) > 18:
                # Split long names into multiple lines
                words = family_name.split('_')
                if len(words) > 1:
                    mid_point = len(words) // 2
                    line1 = '_'.join(words[:mid_point])
                    line2 = '_'.join(words[mid_point:])
                    family_name = f"{line1}\n{line2}"
                else:
                    # If no underscore, just split at midpoint
                    mid = len(family_name) // 2
                    family_name = f"{family_name[:mid]}\n{family_name[mid:]}"
            node_labels[node_id] = family_name
        nx.draw_networkx_labels(explainer_G, pos3, labels=node_labels, ax=ax3, font_size=8, font_weight='bold')

        pruning_type = explainer_graph_data.get('pruning_type', 'attention_based')
        title_text = "Attention-Pruned Graph" if pruning_type == 'attention_based' else "Explainer-Pruned Graph"
        ax3.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
        ax3.text(0.02, 0.98, f"Nodes: {len(explainer_G.nodes())}\nEdges: {len(explainer_G.edges())}",
                transform=ax3.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))

    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    # Add overall title
    plt.suptitle('Graph Comparison: Spearman → k-NN → Attention-Pruned', fontsize=20, fontweight='bold', y=0.95)

    # Add enhanced legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#FF69B4', label='Protected/Anchored Nodes'),
        Patch(facecolor='#B0C4DE', label='Other Nodes'),
        Line2D([0], [0], color='#FF8C00', linewidth=3, label='Top 10 Edges by Weight'),
        Line2D([0], [0], color='darkgray', linewidth=2, label='Other Edges')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
              ncol=4, fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.92)

    # Save comprehensive comparison
    output_path = os.path.join(output_dir, 'comprehensive_graph_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    # Also save individual panels
    for i, (ax, title) in enumerate([(ax1, 'spearman_correlation_graph'),
                                     (ax2, 'knn_sparsified_graph'),
                                     (ax3, 'attention_pruned_graph')]):
        # Extract individual subplot and save
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        individual_path = os.path.join(output_dir, f'{title}_individual.png')
        plt.savefig(individual_path, dpi=300, bbox_inches=extent.expanded(1.2, 1.2), facecolor='white')
        print(f"Individual graph saved: {individual_path}")

    plt.close()

    print(f"Comprehensive graph comparison saved: {output_path}")


def format_statistics_with_std(results_dict, decimals=3):
    """
    Format results dictionary with mean ± std format.
    
    Args:
        results_dict: Dictionary with metric names as keys and lists of values
        decimals: Number of decimal places
        
    Returns:
        dict: Formatted results with mean±std strings
    """
    formatted_results = {}
    
    for metric, values in results_dict.items():
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            formatted_results[metric] = f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
        else:
            formatted_results[metric] = "No data"
    
    return formatted_results


def create_performance_comparison_plot(results_data, output_path, title="Model Performance Comparison"):
    """
    Create a bar plot comparing model performance metrics.
    
    Args:
        results_data: Dictionary with model names as keys and metric dictionaries as values
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['R²', 'MSE', 'MAE']
    
    for i, metric in enumerate(metrics):
        model_names = []
        values = []
        errors = []
        
        for model_name, model_results in results_data.items():
            if metric in model_results:
                # Extract mean and std from formatted string
                if '±' in model_results[metric]:
                    mean_str, std_str = model_results[metric].split(' ± ')
                    mean_val = float(mean_str)
                    std_val = float(std_str)
                else:
                    mean_val = float(model_results[metric])
                    std_val = 0.0
                
                model_names.append(model_name)
                values.append(mean_val)
                errors.append(std_val)
        
        axes[i].bar(model_names, values, yerr=errors, capsize=5, alpha=0.8)
        axes[i].set_title(f'{metric} Comparison', fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_vs_actual_plots(predictions_dict, output_dir, target_names):
    """
    Create prediction vs actual plots for all models and targets.
    
    Args:
        predictions_dict: Dictionary containing fold predictions for each model
        output_dir: Directory to save plots
        target_names: List of target variable names
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for target_name in target_names:
        # Create plots for each model type
        for model_name, model_results in predictions_dict.items():
            if 'fold_predictions' in model_results:
                create_single_model_prediction_plot(
                    model_results['fold_predictions'],
                    os.path.join(output_dir, f'{model_name}_{target_name}_pred_vs_actual.png'),
                    title=f'{model_name.upper()} - {target_name} Predictions',
                    target_name=target_name
                )
        
        # Create combined comparison plot for all models
        create_combined_prediction_comparison(
            predictions_dict,
            os.path.join(output_dir, f'all_models_{target_name}_comparison.png'),
            target_name=target_name
        )

def create_single_model_prediction_plot(fold_predictions, output_path, title, target_name):
    """Create prediction vs actual plot for a single model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 5 folds + 1 combined
    axes = axes.flatten()
    
    # Collect all predictions for combined plot
    all_actual = []
    all_predicted = []
    # Use single color for all points instead of different fold colors
    single_color = '#1f77b4'  # Blue color for all points
    
    # Plot individual folds
    for fold_idx, fold_data in enumerate(fold_predictions):
        if fold_idx < 5:  # Only plot first 5 folds
            ax = axes[fold_idx]
            actual = fold_data['actual']
            predicted = fold_data['predicted']
            
            # Calculate metrics
            r2 = r2_score(actual, predicted) if len(actual) > 1 else 0
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            # Scatter plot with single color
            ax.scatter(actual, predicted, alpha=0.7, s=50, c=single_color, 
                      edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line with outlier-robust axis limits
            actual_array = np.array(actual)
            predicted_array = np.array(predicted)
            
            # Handle extreme outliers (especially for RGGC models)
            # Use percentile-based bounds to avoid extreme axis ranges
            all_values = np.concatenate([actual_array, predicted_array])
            q1, q99 = np.percentile(all_values, [1, 99])  # Use 1st and 99th percentiles
            
            # Set axis range starting from 0 (no negative values)
            # If range is still reasonable, use actual min/max, otherwise use percentiles
            if (q99 - q1) < 1e6 and (q99 - q1) > 0:
                max_val = max(actual_array.max(), predicted_array.max())
            else:
                max_val = q99
                
            # Always start from 0 for both axes
            min_val = 0
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Set axis limits starting from 0
            margin = max_val * 0.05  # 5% margin
            ax.set_xlim(0, max_val + margin)
            ax.set_ylim(0, max_val + margin)
            
            ax.set_xlabel(f'Actual {target_name}')
            ax.set_ylabel(f'Predicted {target_name}')
            ax.set_title(f'Fold {fold_idx + 1}\nR² = {r2:.3f}, MSE = {rmse**2:.3f}')
            ax.grid(True, alpha=0.3)
            
            # Add data to combined plot
            all_actual.extend(actual)
            all_predicted.extend(predicted)
    
    # Combined plot (all folds)
    if len(fold_predictions) > 0:
        ax = axes[5]  # Last subplot
        
        # Use single color for all points (no fold-based coloring)
        ax.scatter(all_actual, all_predicted, alpha=0.7, s=30, c=single_color, 
                  edgecolors='black', linewidth=0.3)
        
        # Overall metrics
        overall_r2 = r2_score(all_actual, all_predicted) if len(all_actual) > 1 else 0
        overall_rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
        overall_mae = mean_absolute_error(all_actual, all_predicted)
        
        # Perfect prediction line with outlier-robust limits
        all_actual_array = np.array(all_actual)
        all_predicted_array = np.array(all_predicted)
        all_values = np.concatenate([all_actual_array, all_predicted_array])
        
        # Set axis range starting from 0 for combined plot
        # Use percentile-based bounds for extreme outliers
        q1, q99 = np.percentile(all_values, [1, 99])
        if (q99 - q1) < 1e6 and (q99 - q1) > 0:
            max_val = max(all_actual_array.max(), all_predicted_array.max())
        else:
            max_val = q99
            
        # Always start from 0 for both axes
        min_val = 0
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Set axis limits starting from 0
        margin = max_val * 0.05
        ax.set_xlim(0, max_val + margin)
        ax.set_ylim(0, max_val + margin)
        
        ax.set_xlabel(f'Actual {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'Combined (All Folds)\nR² = {overall_r2:.3f}, MSE = {overall_rmse**2:.3f}, MAE = {overall_mae:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Prediction plot saved: {output_path}")

def create_combined_prediction_comparison(predictions_dict, output_path, target_name):
    """Create a comparison plot showing all models' predictions."""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    model_names = list(predictions_dict.keys())
    
    for idx, (model_name, model_results) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        if 'fold_predictions' in model_results:
            fold_predictions = model_results['fold_predictions']
            
            # Collect all predictions
            all_actual = []
            all_predicted = []
            
            for fold_data in fold_predictions:
                all_actual.extend(fold_data['actual'])
                all_predicted.extend(fold_data['predicted'])
            
            if len(all_actual) > 0:
                # Calculate metrics
                overall_r2 = r2_score(all_actual, all_predicted) if len(all_actual) > 1 else 0
                overall_rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
                
                # Scatter plot
                ax.scatter(all_actual, all_predicted, alpha=0.6, s=40, 
                          edgecolors='black', linewidth=0.3)
                
                # Perfect prediction line with outlier-robust axis limits
                all_actual_array = np.array(all_actual)
                all_predicted_array = np.array(all_predicted)
                all_values = np.concatenate([all_actual_array, all_predicted_array])
                
                # Set axis range starting from 0 for model comparison
                # Use percentile-based bounds for extreme outliers (RGGC models)
                q1, q99 = np.percentile(all_values, [1, 99])
                if (q99 - q1) < 1e6 and (q99 - q1) > 0:
                    max_val = max(all_actual_array.max(), all_predicted_array.max())
                else:
                    max_val = q99
                    
                # Always start from 0 for both axes
                min_val = 0
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # Set axis limits starting from 0
                margin = max_val * 0.05  # 5% margin
                ax.set_xlim(0, max_val + margin)
                ax.set_ylim(0, max_val + margin)
                
                ax.set_xlabel(f'Actual {target_name}')
                ax.set_ylabel(f'Predicted {target_name}')
                ax.set_title(f'{model_name.upper()}\nR² = {overall_r2:.3f}, MSE = {overall_rmse**2:.3f}')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model Comparison - {target_name} Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Model comparison plot saved: {output_path}")

def generate_feature_importance_report(importance_scores, feature_names, output_path, top_n=20):
    """
    Generate and save a feature importance report.
    
    Args:
        importance_scores: Array of importance scores for each feature
        feature_names: List of feature names
        output_path: Path to save the report
        top_n: Number of top features to highlight
    """
    # Create DataFrame for easier handling
    import pandas as pd
    
    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_score': importance_scores
    }).sort_values('importance_score', ascending=False)
    
    # Save full report
    importance_df.to_csv(output_path.replace('.png', '.csv'), index=False)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Top features bar plot
    top_features = importance_df.head(top_n)
    y_pos = np.arange(len(top_features))
    
    bars = ax1.barh(y_pos, top_features['importance_score'], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([name[:30] + '...' if len(name) > 33 else name 
                        for name in top_features['feature_name']], fontsize=8)
    ax1.set_xlabel('Importance Score')
    ax1.set_title(f'Top {top_n} Most Important Features')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Color bars by importance level
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Distribution histogram
    ax2.hist(importance_scores, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Importance Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Feature Importance Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(importance_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(importance_scores):.3f}')
    ax2.legend()
    
    plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Feature importance report saved: {output_path}")
    print(f"Feature importance CSV saved: {output_path.replace('.png', '.csv')}")
    
    return importance_df