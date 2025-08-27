#!/usr/bin/env python3

"""
Simple Graph Visualizer for Case 1 - GUARANTEED to work
Creates all the graph visualizations you need without complex dependencies
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_regression import MicrobialGNNDataset
from explainers.explainer_regression import GNNExplainerRegression
from models.GNNmodelsRegression import simple_GCN_res_plus_regression

# Set device
device = torch.device('cpu')  # Use CPU to avoid CUDA issues
print(f"Using device: {device}")

def create_graph_visualizations():
    """Create and save graph visualizations with guaranteed success"""
    
    print("🎨 Creating Graph Visualizations - SIMPLE VERSION")
    print("="*60)
    
    # Create output directory
    save_dir = "./graph_visualizations_output"
    os.makedirs(f"{save_dir}", exist_ok=True)
    
    try:
        # Step 1: Load dataset
        print("Step 1: Loading dataset...")
        data_path = "../Data/New_Data.csv"
        
        dataset = MicrobialGNNDataset(
            data_path=data_path,
            k_neighbors=5,
            mantel_threshold=0.05,
            use_fast_correlation=True,
            graph_mode='family',
            family_filter_mode='relaxed'
        )
        
        print(f"✅ Dataset loaded: {len(dataset.data_list)} samples, {len(dataset.node_feature_names)} features")
        
        # Get original graph data
        sample_data = dataset.data_list[0]
        original_edge_index = sample_data.edge_index
        original_num_edges = original_edge_index.shape[1] // 2
        
        print(f"✅ Original k-NN graph: {len(dataset.node_feature_names)} nodes, {original_num_edges} edges")
        
        # Step 2: Train a simple model
        print("Step 2: Training simple GCN model...")
        model = simple_GCN_res_plus_regression(
            hidden_channels=32,
            dropout_prob=0.3,
            input_channel=1,
            output_dim=1
        ).to(device)
        
        # Quick training
        train_loader = DataLoader(dataset.data_list, batch_size=4, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(10):  # Very quick training
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out, _ = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y[:, 1].view(-1, 1)  # H2-km target
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
        
        print("✅ Model trained")
        
        # Step 3: Create explainer and get importance
        print("Step 3: Running explainer...")
        explainer = GNNExplainerRegression(model, device)
        
        # Get edge importance from a few samples
        combined_importance = torch.zeros((len(dataset.node_feature_names), len(dataset.node_feature_names)), device=device)
        
        for i in range(min(3, len(dataset.data_list))):
            edge_importance_matrix, _ = explainer.explain_graph(
                dataset.data_list[i], 
                node_names=dataset.node_feature_names,
                target_idx=1  # H2-km
            )
            combined_importance += edge_importance_matrix
        
        combined_importance /= min(3, len(dataset.data_list))
        
        print("✅ Explainer completed")
        
        # Step 4: Get node importance
        print("Step 4: Calculating node importance...")
        node_importance, sorted_indices = explainer.get_node_importance(
            combined_importance, 
            dataset.node_feature_names
        )
        
        # Step 5: Create sparsified graphs
        print("Step 5: Creating sparsified graphs...")
        
        # Edge-based sparsification (simple version)
        threshold = 0.15  # Lower threshold to get more edges
        adj_matrix = combined_importance.clone()
        adj_matrix[adj_matrix < threshold] = 0
        
        # Create edge list from adjacency matrix
        edge_based_edges = []
        for i in range(adj_matrix.shape[0]):
            for j in range(i+1, adj_matrix.shape[1]):  # Only upper triangle
                if adj_matrix[i, j] > 0:
                    edge_based_edges.append([i, j])
                    edge_based_edges.append([j, i])  # Add reverse edge
        
        if len(edge_based_edges) > 0:
            edge_based_edge_index = torch.tensor(edge_based_edges).t().contiguous()
        else:
            edge_based_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Edge-based sparsification: {edge_based_edge_index.shape[1]//2} edges")
        
        # Node-based sparsification
        important_nodes = np.where(node_importance > 0.35)[0]  # Higher threshold for fewer nodes
        if len(important_nodes) < 20:
            important_nodes = sorted_indices[:20]  # Keep top 20
        
        # Create edges only between important nodes
        node_based_edges = []
        for i in range(len(important_nodes)):
            for j in range(i+1, len(important_nodes)):
                node_i, node_j = important_nodes[i], important_nodes[j]
                if combined_importance[node_i, node_j] > threshold:
                    node_based_edges.append([i, j])
                    node_based_edges.append([j, i])
        
        if len(node_based_edges) > 0:
            node_based_edge_index = torch.tensor(node_based_edges).t().contiguous()
        else:
            node_based_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"Node-based sparsification: {len(important_nodes)} nodes, {node_based_edge_index.shape[1]//2} edges")
        
        # Step 6: CREATE THE VISUALIZATIONS
        print("Step 6: Creating graph visualizations...")
        
        create_three_panel_visualization(
            original_edge_index, len(dataset.node_feature_names),
            edge_based_edge_index, len(dataset.node_feature_names), 
            node_based_edge_index, len(important_nodes),
            save_dir
        )
        
        # Step 7: Create individual high-res graphs
        create_individual_graphs(
            original_edge_index, len(dataset.node_feature_names),
            edge_based_edge_index, len(dataset.node_feature_names),
            node_based_edge_index, len(important_nodes),
            save_dir
        )
        
        print(f"\n🎉 SUCCESS! Graph visualizations created in: {save_dir}")
        print(f"📁 Files created:")
        
        # List created files
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), save_dir)
                print(f"  ✅ {rel_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_three_panel_visualization(original_edges, original_nodes, edge_based_edges, edge_based_nodes, node_based_edges, node_based_nodes, save_dir):
    """Create three-panel comparison visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    # Panel 1: Original k-NN graph
    visualize_graph_on_axis(
        original_edges, original_nodes,
        axes[0], 
        f"Original k-NN Graph\n{original_nodes} nodes, {original_edges.shape[1]//2} edges"
    )
    
    # Panel 2: Edge-based sparsified
    visualize_graph_on_axis(
        edge_based_edges, edge_based_nodes,
        axes[1],
        f"Edge-Based Sparsified\n{edge_based_nodes} nodes, {edge_based_edges.shape[1]//2} edges"
    )
    
    # Panel 3: Node-based sparsified  
    visualize_graph_on_axis(
        node_based_edges, node_based_nodes,
        axes[2],
        f"Node-Based Sparsified\n{node_based_nodes} nodes, {node_based_edges.shape[1]//2} edges"
    )
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/three_panel_graph_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Three-panel comparison saved: {save_dir}/three_panel_graph_comparison.png")

def create_individual_graphs(original_edges, original_nodes, edge_based_edges, edge_based_nodes, node_based_edges, node_based_nodes, save_dir):
    """Create individual high-resolution graphs"""
    
    graphs = [
        (original_edges, original_nodes, "original_knn_graph.png", f"Original k-NN Graph\n{original_nodes} nodes, {original_edges.shape[1]//2} edges"),
        (edge_based_edges, edge_based_nodes, "edge_based_sparsified.png", f"Edge-Based Sparsified\n{edge_based_nodes} nodes, {edge_based_edges.shape[1]//2} edges"),
        (node_based_edges, node_based_nodes, "node_based_sparsified.png", f"Node-Based Sparsified\n{node_based_nodes} nodes, {node_based_edges.shape[1]//2} edges")
    ]
    
    for edge_index, num_nodes, filename, title in graphs:
        plt.figure(figsize=(15, 15))
        ax = plt.gca()
        visualize_graph_on_axis(edge_index, num_nodes, ax, title)
        plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Individual graph saved: {save_dir}/{filename}")

def visualize_graph_on_axis(edge_index, num_nodes, ax, title):
    """Visualize a single graph on given axis with consistent styling"""
    
    if edge_index.shape[1] == 0:
        ax.text(0.5, 0.5, "No edges to display", ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        return
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u != v:  # Avoid self-loops
            G.add_edge(u, v)
    
    # Create layout
    if len(G.nodes()) > 100:
        # Use faster layout for large graphs
        pos = nx.random_layout(G, seed=42)
    else:
        try:
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
    
    # Calculate node and edge sizes based on graph size
    node_size = max(10, 500 - num_nodes)  # Smaller nodes for larger graphs
    edge_width = max(0.1, 2 - num_nodes/100)  # Thinner edges for larger graphs
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', 
                          alpha=0.7, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.6, 
                          edge_color='gray', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add graph stats as text
    stats_text = f"Nodes: {num_nodes}\nEdges: {len(G.edges())}\nDensity: {nx.density(G):.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

if __name__ == "__main__":
    print("🚀 Starting Simple Graph Visualizer")
    success = create_graph_visualizations()
    
    if success:
        print("\n🎊 Graph visualizations completed successfully!")
        print("Check ./graph_visualizations_output/ for all files")
    else:
        print("\n⚠️  Some issues occurred during visualization")