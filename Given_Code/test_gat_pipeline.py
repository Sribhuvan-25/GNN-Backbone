#!/usr/bin/env python3
"""
Test script for GAT-only embedding pipeline

This script demonstrates how to run a GAT-only pipeline where:
1. Only GAT models are trained (no GCN or RGGC)
2. GAT model is used for GNNExplainer sparsification
3. GAT embeddings are extracted and used for ML models
4. End-to-end consistency with GAT throughout
"""

from embedding_pipeline import EmbeddingPipeline

def main():
    print("="*80)
    print("GAT-ONLY EMBEDDING PIPELINE TEST")
    print("="*80)
    
    # Initialize GAT-only pipeline
    pipeline = EmbeddingPipeline(
        data_path="../Data/New_data.csv",
        k_neighbors=10,
        hidden_dim=64,
        num_epochs=300,  # Reduced for quick testing
        num_folds=5,    # Reduced for quick testing
        save_dir="./gat_test_results",
        single_model_type='gat'  # This is the key parameter for GAT-only flow
    )
    
    print("\nPipeline Configuration:")
    print(f"- Model Type: GAT-ONLY")
    print(f"- Models to train: {pipeline.gnn_models_to_train}")
    print(f"- Target variables: {pipeline.target_names}")
    print(f"- Save directory: {pipeline.save_dir}")
    
    # Run the pipeline
    print("\nStarting GAT-only pipeline...")
    results = pipeline.run_pipeline()
    
    print("\n" + "="*80)
    print("GAT-ONLY PIPELINE COMPLETED!")
    print("="*80)
    
    # Print summary of what was done
    print("\nWhat happened in this GAT-only pipeline:")
    print("1. ✅ Trained ONLY GAT models on KNN-sparsified graph")
    print("2. ✅ Used GAT model for GNNExplainer sparsification")
    print("3. ✅ Trained ONLY GAT models on explainer-sparsified graph")
    print("4. ✅ Selected best GAT model (KNN vs explainer)")
    print("5. ✅ Extracted embeddings from best GAT model")
    print("6. ✅ Trained ML models (LinearSVR, ExtraTrees) on GAT embeddings")
    print("7. ✅ Generated plots comparing GAT variants + ML models")
    
    print(f"\nResults saved to: {pipeline.save_dir}")
    print("Check these files:")
    print("- results_summary.csv: Performance metrics")
    print("- embeddings/*_gat_embeddings.npy: GAT embeddings")
    print("- plots/*_comprehensive_results.png: Comparison plots")
    
    return results

if __name__ == "__main__":
    results = main() 