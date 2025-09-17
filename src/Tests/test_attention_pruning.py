#!/usr/bin/env python3
"""
Test script to verify attention-based node pruning works for all GNN models:
- GAT: Uses explicit attention weights 
- RGGC: Uses gating mechanism + gradients
- GCN: Uses gradient-based node importance
- Graph Transformer (kg_gt): Uses TransformerConv attention weights

This script tests that each model can extract attention/importance scores correctly.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from models.GNNmodelsRegression import (
    simple_GCN_res_regression, 
    simple_GAT_regression, 
    simple_RGGC_regression,
    simple_GraphTransformer_regression
)
from explainers.explainer_regression import GNNExplainerRegression


def create_sample_data(num_nodes=10, num_features=5):
    """Create a small sample graph for testing"""
    # Create sample node features
    x = torch.randn(num_nodes, num_features)
    
    # Create a simple connected graph (ring topology)
    edge_list = []
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        edge_list.append([i, next_node])
        edge_list.append([next_node, i])  # Make undirected
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create sample data object
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)
    
    return data


def test_model_attention_extraction():
    """Test attention extraction for each model type"""
    print("="*80)
    print("TESTING ATTENTION-BASED NODE PRUNING FOR ALL MODELS")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(num_nodes=15, num_features=8)
    device = torch.device('cpu')  # Use CPU for testing
    
    # Create explainer (will be created for each model)
    explainer = None
    
    # Test parameters
    model_configs = {
        'GCN': {
            'class': simple_GCN_res_regression,
            'params': {
                'hidden_channels': 32,
                'output_dim': 1,
                'dropout_prob': 0.1,
                'input_channel': data.x.shape[1]
            }
        },
        'GAT': {
            'class': simple_GAT_regression,
            'params': {
                'hidden_channels': 32,
                'output_dim': 1,
                'dropout_prob': 0.1,
                'input_channel': data.x.shape[1],
                'num_heads': 4
            }
        },
        'RGGC': {
            'class': simple_RGGC_regression,
            'params': {
                'hidden_channels': 32,
                'output_dim': 1,
                'dropout_prob': 0.1,
                'input_channel': data.x.shape[1]
            }
        },
        'Graph Transformer': {
            'class': simple_GraphTransformer_regression,
            'params': {
                'hidden_channels': 32,
                'output_dim': 1,
                'dropout_prob': 0.1,
                'input_channel': data.x.shape[1],
                'num_heads': 4,
                'num_layers': 2,
                'activation': 'identity'
            }
        }
    }
    
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"TESTING {model_name.upper()} ATTENTION EXTRACTION")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = config['class'](**config['params'])
            model.to(device)
            model.eval()
            
            print(f"✅ {model_name} model created successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create explainer for this model
            explainer = GNNExplainerRegression(model=model, device=device)
            
            # Try to extract attention scores
            if model_name == 'GAT':
                attention_scores = explainer._extract_gat_attention_scores(model, data)
            elif model_name == 'RGGC':
                attention_scores = explainer._extract_rggc_attention_scores(model, data)
            elif model_name == 'Graph Transformer':
                attention_scores = explainer._extract_kg_gt_attention_scores(model, data)
            elif model_name == 'GCN':
                attention_scores = explainer._extract_gradient_attention_scores(model, data)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            if attention_scores is not None:
                print(f"✅ {model_name} attention extraction successful!")
                print(f"   Attention scores shape: {attention_scores.shape}")
                print(f"   Score range: {attention_scores.min():.6f} - {attention_scores.max():.6f}")
                print(f"   Score sum: {attention_scores.sum():.6f}")
                
                # Test node pruning  
                pruned_data, kept_nodes, pruned_names, att_scores = explainer.create_attention_based_node_pruning(
                    data=data,
                    model=model, 
                    node_names=[f"Node_{i}" for i in range(data.x.shape[0])],
                    attention_threshold=0.1,
                    min_nodes=5
                )
                
                print(f"✅ {model_name} node pruning successful!")
                print(f"   Original nodes: {data.x.shape[0]} → Pruned nodes: {pruned_data.x.shape[0]}")
                print(f"   Original edges: {data.edge_index.shape[1]} → Pruned edges: {pruned_data.edge_index.shape[1]}")
                
                results[model_name] = {
                    'attention_extraction': True,
                    'node_pruning': True,
                    'attention_range': (attention_scores.min(), attention_scores.max()),
                    'pruning_ratio': pruned_data.x.shape[0] / data.x.shape[0]
                }
            else:
                print(f"❌ {model_name} attention extraction failed!")
                results[model_name] = {
                    'attention_extraction': False,
                    'node_pruning': False,
                    'attention_range': None,
                    'pruning_ratio': None
                }
                
        except Exception as e:
            print(f"❌ {model_name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'attention_extraction': False,
                'node_pruning': False,
                'error': str(e),
                'attention_range': None,
                'pruning_ratio': None
            }
    
    # Print summary
    print(f"\n{'='*80}")
    print("ATTENTION-BASED NODE PRUNING TEST SUMMARY")
    print(f"{'='*80}")
    
    all_passed = True
    for model_name, result in results.items():
        status = "✅ PASS" if result['attention_extraction'] and result['node_pruning'] else "❌ FAIL"
        print(f"{model_name:20s}: {status}")
        if result['attention_extraction'] and result['node_pruning']:
            print(f"{'':22s}Attention range: {result['attention_range'][0]:.6f} - {result['attention_range'][1]:.6f}")
            print(f"{'':22s}Pruning ratio: {result['pruning_ratio']:.2f}")
        elif 'error' in result:
            print(f"{'':22s}Error: {result['error']}")
        
        if not (result['attention_extraction'] and result['node_pruning']):
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("🎉 ALL MODELS PASSED ATTENTION-BASED NODE PRUNING TESTS!")
        print("\nYour attention-based node pruning implementation is ready for:")
        print("- GAT: Using explicit attention weights from multi-head attention")
        print("- RGGC: Using gating mechanism + gradient-based importance")
        print("- GCN: Using gradient-based node importance")
        print("- Graph Transformer: Using TransformerConv attention weights")
    else:
        print("⚠️  SOME MODELS FAILED - Please check the implementation")
    
    return all_passed


if __name__ == "__main__":
    test_model_attention_extraction()