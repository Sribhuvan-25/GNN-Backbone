#!/usr/bin/env python3

"""
Enhanced Graph Visualizer with Node Labels
Creates publication-quality graph visualizations with labeled nodes like the example image
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

def create_labeled_graph_visualization(edge_index, node_names, title, save_path, 
                                     node_importance=None, figsize=(12, 10)):
    """
    Create a publication-quality graph visualization with labeled nodes
    
    Args:
        edge_index: PyTorch tensor of edges (2, num_edges)
        node_names: List of node names
        title: Title for the graph
        save_path: Path to save the image
        node_importance: Optional node importance scores for sizing
        figsize: Figure size tuple
    """
    
    if edge_index.shape[1] == 0:
        print(f"Warning: No edges to visualize for {title}")
        return
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with names
    for i, name in enumerate(node_names):
        G.add_node(i, name=name)
    
    # Add edges (avoid self-loops and duplicates)
    edges_added = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u != v and (min(u,v), max(u,v)) not in edges_added:
            G.add_edge(u, v)
            edges_added.add((min(u,v), max(u,v)))
    
    print(f"Creating {title}: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Create figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Choose layout based on graph size
    if len(G.nodes) <= 30:
        # Use spring layout for small graphs - better quality
        try:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        except:
            pos = nx.circular_layout(G)
    elif len(G.nodes) <= 100:
        # Use spring layout with fewer iterations for medium graphs
        try:
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
    else:
        # Use faster layout for large graphs
        pos = nx.random_layout(G, seed=42)
    
    # Calculate node sizes based on importance or degree
    if node_importance is not None:
        # Normalize importance scores
        importance_array = np.array([node_importance.get(i, 0) for i in range(len(node_names))])
        node_sizes = 300 + 1000 * (importance_array - importance_array.min()) / (importance_array.max() - importance_array.min() + 1e-8)
    else:
        # Base size on degree centrality
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [300 + 500 * degrees.get(i, 0) / max_degree for i in range(len(node_names))]
    
    # Color nodes based on importance or degree
    if node_importance is not None:
        node_colors = [plt.cm.viridis(importance_array[i] / (importance_array.max() + 1e-8)) for i in range(len(node_names))]
    else:
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_colors = [plt.cm.viridis(degrees.get(i, 0) / max_degree) for i in range(len(node_names))]
    
    # Draw edges first (so they appear behind nodes)
    edge_width = 0.5 if len(G.edges) > 50 else 1.0
    nx.draw_networkx_edges(G, pos, 
                          width=edge_width,
                          alpha=0.4,
                          edge_color='gray',
                          ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8,
                          ax=ax)
    
    # Add node labels with better formatting
    label_pos = {}
    for node, (x, y) in pos.items():
        # Offset labels slightly to avoid overlap with nodes
        label_pos[node] = (x, y + 0.05)
    
    # Create labels dictionary
    labels = {}
    for i, name in enumerate(node_names):
        # Truncate long names for readability
        if len(name) > 15:
            labels[i] = name[:12] + "..."
        else:
            labels[i] = name
    
    # Draw labels with background boxes for better readability
    for node, label in labels.items():
        x, y = label_pos[node]
        ax.text(x, y, label, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8 if len(G.nodes) > 30 else 10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor='white', 
                         edgecolor='black',
                         alpha=0.7))
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add graph statistics
    density = nx.density(G)
    avg_degree = np.mean(list(dict(G.degree()).values())) if G.nodes else 0
    
    stats_text = f"Nodes: {len(G.nodes)}\nEdges: {len(G.edges)}\nDensity: {density:.3f}\nAvg Degree: {avg_degree:.1f}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Remove axes
    ax.set_axis_off()
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Saved: {save_path}")

def create_comparison_graphs(original_edge_index, original_node_names,
                           edge_pruned_edge_index, edge_pruned_node_names,
                           node_pruned_edge_index, node_pruned_node_names,
                           save_dir, node_importance=None):
    """Create comparison of original, edge-pruned, and node-pruned graphs"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Individual high-quality graphs
    create_labeled_graph_visualization(
        original_edge_index, original_node_names,
        f"Original k-NN Graph\n{len(original_node_names)} nodes, {original_edge_index.shape[1]//2} edges",
        f"{save_dir}/original_knn_graph_labeled.png",
        node_importance, figsize=(15, 12)
    )
    
    create_labeled_graph_visualization(
        edge_pruned_edge_index, edge_pruned_node_names,
        f"Edge-Based Sparsified\n{len(edge_pruned_node_names)} nodes, {edge_pruned_edge_index.shape[1]//2} edges",
        f"{save_dir}/edge_based_sparsified_labeled.png",
        node_importance, figsize=(15, 12)
    )
    
    create_labeled_graph_visualization(
        node_pruned_edge_index, node_pruned_node_names,
        f"Node-Based Sparsified\n{len(node_pruned_node_names)} nodes, {node_pruned_edge_index.shape[1]//2} edges",
        f"{save_dir}/node_based_sparsified_labeled.png",
        node_importance, figsize=(15, 12)
    )
    
    # Side-by-side comparison (smaller individual graphs)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Original graph
    create_graph_on_axis(original_edge_index, original_node_names, axes[0],
                        f"Original k-NN Graph\n{len(original_node_names)} nodes",
                        node_importance)
    
    # Edge-pruned graph  
    create_graph_on_axis(edge_pruned_edge_index, edge_pruned_node_names, axes[1],
                        f"Edge-Based Sparsified\n{len(edge_pruned_node_names)} nodes",
                        node_importance)
    
    # Node-pruned graph
    create_graph_on_axis(node_pruned_edge_index, node_pruned_node_names, axes[2],
                        f"Node-Based Sparsified\n{len(node_pruned_node_names)} nodes",
                        node_importance)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/graph_comparison_labeled.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Saved: {save_dir}/graph_comparison_labeled.png")

def create_graph_on_axis(edge_index, node_names, ax, title, node_importance=None):
    """Create a graph on a specific matplotlib axis"""
    
    if edge_index.shape[1] == 0:
        ax.text(0.5, 0.5, "No edges to display", ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        return
    
    # Create NetworkX graph
    G = nx.Graph()
    for i, name in enumerate(node_names):
        G.add_node(i, name=name)
    
    # Add edges
    edges_added = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u != v and (min(u,v), max(u,v)) not in edges_added:
            G.add_edge(u, v)
            edges_added.add((min(u,v), max(u,v)))
    
    # Layout
    if len(G.nodes) <= 30:
        try:
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)
    
    # Node sizes and colors
    if node_importance is not None:
        importance_array = np.array([node_importance.get(i, 0) for i in range(len(node_names))])
        node_sizes = 100 + 300 * (importance_array - importance_array.min()) / (importance_array.max() - importance_array.min() + 1e-8)
        node_colors = [plt.cm.viridis(importance_array[i] / (importance_array.max() + 1e-8)) for i in range(len(node_names))]
    else:
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [100 + 200 * degrees.get(i, 0) / max_degree for i in range(len(node_names))]
        node_colors = [plt.cm.viridis(degrees.get(i, 0) / max_degree) for i in range(len(node_names))]
    
    # Draw graph
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.4, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    
    # Add labels for small graphs only
    if len(G.nodes) <= 50:
        labels = {}
        for i, name in enumerate(node_names):
            if len(name) > 12:
                labels[i] = name[:9] + "..."
            else:
                labels[i] = name
        
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

if __name__ == "__main__":
    print("Graph visualizer with labels - use as module")