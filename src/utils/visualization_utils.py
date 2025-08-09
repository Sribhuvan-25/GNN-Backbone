"""
Visualization utilities for graph rendering and statistical plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch


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
    
    # Add nodes with names
    for i, node_name in enumerate(node_features):
        G.add_node(i, name=node_name)
    
    # Add edges with weights
    edge_index_np = edge_index.cpu().numpy()
    edge_weight_np = edge_weight.cpu().numpy()
    
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
                           legend_elements=None, figsize=(15, 12)):
    """
    Save a graph visualization to file.
    
    Args:
        G: NetworkX graph
        node_colors: Dictionary mapping node indices to colors
        output_path: Path to save the figure
        title: Plot title
        legend_elements: List of legend elements
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Draw nodes
    node_color_list = [node_colors.get(node, '#95A5A6') for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, 
                          node_size=800, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Add labels
    node_labels = nx.get_node_attributes(G, 'name')
    if not node_labels:
        node_labels = {i: str(i) for i in G.nodes()}
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
    metrics = ['R²', 'RMSE', 'MAE']
    
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