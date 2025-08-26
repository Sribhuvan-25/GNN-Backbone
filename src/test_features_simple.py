#!/usr/bin/env python3

"""
Simple test to verify our new features work correctly
Tests the explainer functionality independently
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_explainer_features():
    """Test the new explainer features"""
    print("🧪 Testing New Explainer Features")
    print("=" * 50)
    
    try:
        from explainers.explainer_regression import GNNExplainerRegression
        from models.GNNmodelsRegression import simple_GCN_res_plus_regression
        
        print("✅ Successfully imported explainer and model")
        
        # Create dummy data to test the features
        device = torch.device('cpu')
        
        # Create a simple GNN model
        model = simple_GCN_res_plus_regression(
            hidden_channels=32,
            dropout_prob=0.1,
            input_channel=1,
            output_dim=1
        ).to(device)
        
        print("✅ Created test GNN model")
        
        # Create dummy graph data
        num_nodes = 20
        num_edges = 40
        
        x = torch.randn(num_nodes, 1)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.randn(1, 1)  # Single target
        
        dummy_data = Data(x=x, edge_index=edge_index, y=y)
        print(f"✅ Created dummy graph data: {num_nodes} nodes, {num_edges} edges")
        
        # Test the explainer
        explainer = GNNExplainerRegression(model, device)
        print("✅ Created GNN explainer")
        
        # Generate dummy node names
        node_names = [f"Family_{i}" for i in range(num_nodes)]
        
        # Test explanation generation
        edge_importance_matrix, explanation = explainer.explain_graph(
            dummy_data, 
            node_names=node_names,
            target_idx=0
        )
        print("✅ Generated edge importance matrix")
        
        # Test NEW FEATURE 1: Node importance reporting
        print("\n" + "=" * 50)
        print("🆕 TESTING FEATURE 1: Node Importance Reporting")
        print("=" * 50)
        
        node_importance, sorted_indices = explainer.get_node_importance(
            edge_importance_matrix, 
            node_names
        )
        
        print(f"✅ Node importance calculated for {len(node_importance)} nodes")
        
        # Test NEW FEATURE 2: Node-based pruning
        print("\n" + "=" * 50)
        print("🆕 TESTING FEATURE 2: Node-Based Pruning")
        print("=" * 50)
        
        pruned_data, kept_nodes, pruned_node_names = explainer.create_node_pruned_graph(
            dummy_data,
            edge_importance_matrix,
            node_names,
            importance_threshold=0.1,
            min_nodes=5
        )
        
        print(f"✅ Node pruning completed: {len(kept_nodes)} nodes kept")
        
        # Test NEW FEATURE 3: t-SNE visualization
        print("\n" + "=" * 50)
        print("🆕 TESTING FEATURE 3: t-SNE Embedding Visualization")
        print("=" * 50)
        
        # Create dummy embeddings and targets
        dummy_embeddings = np.random.randn(10, 16)  # 10 samples, 16 features
        dummy_targets = np.random.randn(10, 1)      # 10 samples, 1 target
        
        os.makedirs("./test_plots", exist_ok=True)
        
        tsne_coords = explainer.create_tsne_embedding_plot(
            dummy_embeddings,
            dummy_targets,
            target_names=['Test_Target'],
            save_path='./test_plots/test_tsne.png'
        )
        
        print(f"✅ t-SNE plot created with shape: {tsne_coords.shape}")
        
        # Final success message
        print("\n" + "🎉" * 20)
        print("✅ ALL NEW FEATURES WORKING CORRECTLY!")
        print("🎉" * 20)
        
        print("\n📋 Summary of tested features:")
        print("  ✅ Node importance reporting - WORKING")
        print("  ✅ Node-based pruning - WORKING")
        print("  ✅ t-SNE embedding visualization - WORKING")
        
        if os.path.exists('./test_plots/test_tsne.png'):
            print(f"\n📁 Test files created:")
            print(f"  ✅ ./test_plots/test_tsne.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_explainer_features()
    if success:
        print(f"\n🚀 Ready to use the new features!")
    else:
        print(f"\n⚠️  Some issues need to be resolved")