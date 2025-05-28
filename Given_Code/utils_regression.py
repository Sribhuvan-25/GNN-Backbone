#!/usr/bin/env python3
"""
Test script for the improved embedding pipeline with family-level nodes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_pipeline import EmbeddingPipeline

def test_improved_pipeline():
    """Test the improved embedding pipeline"""
    print("="*80)
    print("TESTING IMPROVED EMBEDDING PIPELINE")
    print("="*80)
    print("Key improvements:")
    print("- Family-level nodes (instead of OTU-level)")
    print("- Lower GNNExplainer threshold (0.1 instead of 0.3)")
    print("- Individual plots for both ExtraTrees and LinearSVR")
    print("- No individual fold plots (only overall results)")
    print("- Better sparsification monitoring")
    print("="*80)
    
    # Test GAT-only pipeline with family-level nodes
    print("\nTesting GAT-ONLY pipeline with family-level nodes...")
    
    gat_pipeline = EmbeddingPipeline(
        data_path="../Data/New_data.csv",
        k_neighbors=10,
        hidden_dim=64,
        num_epochs=30,  # Reduced for faster testing
        num_folds=3,    # Reduced for faster testing
        save_dir="./improved_gat_test_results",
        single_model_type='gat',
        graph_mode='family',  # Use family-level nodes
        importance_threshold=0.1,  # Lower threshold for better sparsification
        patience=10  # Reduced for faster testing
    )
    
    print(f"\nPipeline configuration:")
    print(f"- Graph mode: {gat_pipeline.graph_mode}")
    print(f"- Single model type: {gat_pipeline.single_model_type}")
    print(f"- Importance threshold: {gat_pipeline.importance_threshold}")
    print(f"- Number of folds: {gat_pipeline.num_folds}")
    print(f"- Number of epochs: {gat_pipeline.num_epochs}")
    
    # Run the pipeline
    results = gat_pipeline.run_pipeline()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nResults saved to: ./improved_gat_test_results/")
    print("\nCheck the following:")
    print("1. results_summary.csv - Should show family-level node counts")
    print("2. plots/ directory - Should have individual ExtraTrees and LinearSVR plots")
    print("3. explanations/ directory - Should have sparsification info")
    print("4. graphs/ directory - Should show family-level graph visualizations")
    
    # Print summary of what was generated
    import os
    results_dir = "./improved_gat_test_results"
    if os.path.exists(results_dir):
        print(f"\nGenerated files in {results_dir}:")
        for root, dirs, files in os.walk(results_dir):
            level = root.replace(results_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    return results

if __name__ == "__main__":
    results = test_improved_pipeline() 