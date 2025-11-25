#!/usr/bin/env python3
"""
Run GNN Embeddings Pipeline for Genus-Level Analysis (No Knowledge Anchoring)

This script runs a GNN-based embedding extraction pipeline without domain expert
case constraints (knowledge anchoring). It uses ALL genus features for a fair
comparison with other baselines.

Pipeline Flow:
1. Load ALL genus-level features (no RFE, no domain expert filtering)
2. Build k-NN graph from genus abundance data
3. Train GNN models (GCN, GAT, RGGC) with nested CV hyperparameter tuning
4. Apply GNNExplainer for graph sparsification (optional)
5. Retrain GNNs on sparsified graph (optional)
6. Extract embeddings from best GNN model
7. Train classical ML models (LinearSVR, ExtraTrees) on GNN embeddings
8. Report metrics in "mean ± std" format

Usage:
    python run_genus_gnn_embeddings.py --target ACE-km
    python run_genus_gnn_embeddings.py --target H2-km
    python run_genus_gnn_embeddings.py --target both  # Run both targets
"""

# Set matplotlib to use non-GUI backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import the embeddings pipeline
from pipelines.embeddings_pipeline import MixedEmbeddingPipeline

def create_directories():
    """Create directories for organizing outputs"""
    base_dir = 'results_gnn_embeddings_genus'
    subdirs = ['plots', 'metrics', 'embeddings', 'models', 'graphs']

    os.makedirs(base_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(f'{base_dir}/{subdir}', exist_ok=True)
        print(f"Created directory: {base_dir}/{subdir}")

    return base_dir

def run_gnn_embeddings_pipeline(data_path, target='ACE-km', base_dir='results_gnn_embeddings_genus'):
    """
    Run GNN embeddings pipeline for a specific target

    Parameters:
    ----------
    data_path : str
        Path to the input data file with genus-level abundance data
    target : str
        Target variable to predict ('ACE-km', 'H2-km', or 'both')
    base_dir : str
        Base directory for saving results
    """
    print(f"\n{'='*80}")
    print(f"Running GNN Embeddings Pipeline for {target}")
    print(f"{'='*80}")

    # Create target-specific save directory
    save_dir = f"{base_dir}/{target}"
    os.makedirs(save_dir, exist_ok=True)

    # Pipeline configuration
    config = {
        'data_path': data_path,
        'k_neighbors': 10,                    # k-NN graph construction
        'mantel_threshold': 0.05,             # Mantel test threshold
        'hidden_dim': 64,                     # GNN hidden dimension
        'dropout_rate': 0.3,                  # Dropout rate
        'batch_size': 8,                      # Batch size
        'learning_rate': 0.001,               # Learning rate
        'weight_decay': 1e-4,                 # Weight decay
        'num_epochs': 200,                    # Training epochs
        'patience': 20,                       # Early stopping patience
        'num_folds': 5,                       # Cross-validation folds
        'save_dir': save_dir,
        'importance_threshold': 0.2,          # Explainer threshold
        'use_fast_correlation': False,        # Use standard correlation
        'graph_mode': 'genus',                # GENUS-LEVEL ANALYSIS (not family)
        'family_filter_mode': 'strict',       # Not used for genus mode
        'use_enhanced_training': True,        # Enhanced training
        'adaptive_hyperparameters': True,     # Adaptive hyperparameters
        'use_nested_cv': True,                # Nested CV for hyperparameter tuning
        'use_node_sparsification': False,     # Edge-based sparsification only
        'graph_construction_method': 'original'
    }

    print(f"\nPipeline Configuration:")
    print(f"  Graph Mode: {config['graph_mode']} (using ALL genus features)")
    print(f"  K-Neighbors: {config['k_neighbors']}")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Folds: {config['num_folds']}")
    print(f"  Nested CV: {config['use_nested_cv']}")
    print(f"  Save Directory: {save_dir}")
    print(f"\nNOTE: No RFE, no domain expert filtering - using ALL genus features")

    try:
        # Initialize pipeline
        print(f"\n{'-'*60}")
        print(f"Initializing MixedEmbeddingPipeline...")
        print(f"{'-'*60}")

        pipeline = MixedEmbeddingPipeline(**config)

        # Run the full pipeline
        print(f"\n{'-'*60}")
        print(f"Running full GNN embeddings pipeline...")
        print(f"{'-'*60}")

        results = pipeline.run_full_pipeline(target_name=target)

        # Extract and save summary metrics
        print(f"\n{'-'*60}")
        print(f"Extracting Summary Metrics")
        print(f"{'-'*60}")

        if results and 'ml_results' in results:
            save_summary_metrics(results, target, base_dir)
        else:
            print(f"WARNING: No ML results found for {target}")

        print(f"\n✅ Successfully completed pipeline for {target}")
        return results

    except Exception as e:
        print(f"\n❌ ERROR running pipeline for {target}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_summary_metrics(results, target, base_dir):
    """
    Save summary metrics in mean ± std format (matching RFE_Simple_CV.py format)

    Parameters:
    ----------
    results : dict
        Results dictionary from pipeline
    target : str
        Target variable name
    base_dir : str
        Base directory for saving results
    """
    print(f"\nSaving summary metrics for {target}...")

    ml_results = results.get('ml_results', {})

    if not ml_results:
        print(f"WARNING: No ML results to save for {target}")
        return

    # Prepare summary data
    summary_data = []

    for model_name, model_results in ml_results.items():
        if 'cv_scores' in model_results:
            cv_scores = model_results['cv_scores']

            # Calculate mean ± std for each metric
            r2_mean = np.mean(cv_scores['r2'])
            r2_std = np.std(cv_scores['r2'])
            mse_mean = np.mean(cv_scores['mse'])
            mse_std = np.std(cv_scores['mse'])
            mae_mean = np.mean(cv_scores['mae'])
            mae_std = np.std(cv_scores['mae'])
            rmse_mean = np.sqrt(mse_mean)
            rmse_std = mse_std / (2 * np.sqrt(mse_mean))  # Delta method

            summary_data.append({
                'Target': target,
                'Model': model_name,
                'R2_Mean': r2_mean,
                'R2_Std': r2_std,
                'R2_Mean_Plus_Minus_Std': f"{r2_mean:.4f} ± {r2_std:.4f}",
                'MSE_Mean': mse_mean,
                'MSE_Std': mse_std,
                'MSE_Mean_Plus_Minus_Std': f"{mse_mean:.4f} ± {mse_std:.4f}",
                'MAE_Mean': mae_mean,
                'MAE_Std': mae_std,
                'MAE_Mean_Plus_Minus_Std': f"{mae_mean:.4f} ± {mae_std:.4f}",
                'RMSE_Mean': rmse_mean,
                'RMSE_Std': rmse_std,
                'RMSE_Mean_Plus_Minus_Std': f"{rmse_mean:.4f} ± {rmse_std:.4f}"
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Save to CSV
        metrics_dir = f"{base_dir}/metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        csv_path = f"{metrics_dir}/{target}_gnn_embeddings_metrics_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"✅ Saved metrics summary: {csv_path}")

        # Print summary to console
        print(f"\n{'='*60}")
        print(f"SUMMARY METRICS FOR {target} (Mean ± Std)")
        print(f"{'='*60}")
        print(f"\n{summary_df.to_string(index=False)}")
        print(f"{'='*60}")

        # Find and print best model
        best_idx = summary_df['R2_Mean'].idxmax()
        best_model = summary_df.loc[best_idx]
        print(f"\n🏆 Best Model: {best_model['Model']}")
        print(f"   R² = {best_model['R2_Mean_Plus_Minus_Std']}")
        print(f"   MSE = {best_model['MSE_Mean_Plus_Minus_Std']}")
        print(f"   RMSE = {best_model['RMSE_Mean_Plus_Minus_Std']}")
    else:
        print(f"WARNING: No summary data to save for {target}")

def main():
    parser = argparse.ArgumentParser(
        description='Run GNN Embeddings Pipeline for Genus-Level Analysis (No Knowledge Anchoring)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for ACE-km target
  python run_genus_gnn_embeddings.py --target ACE-km

  # Run for H2-km target
  python run_genus_gnn_embeddings.py --target H2-km

  # Run for both targets
  python run_genus_gnn_embeddings.py --target both

  # Use custom data path
  python run_genus_gnn_embeddings.py --target ACE-km --data_path /path/to/data.csv
"""
    )

    parser.add_argument('--target',
                        default='both',
                        choices=['ACE-km', 'H2-km', 'both'],
                        help='Target variable to predict (default: both)')
    parser.add_argument('--data_path',
                        default='../Data/New_Data.csv',
                        help='Path to the dataset (default: ../Data/New_Data.csv)')

    args = parser.parse_args()

    # Print header
    print(f"""
{'='*80}
GNN EMBEDDINGS PIPELINE - GENUS-LEVEL ANALYSIS
{'='*80}

Configuration:
  Data Path: {args.data_path}
  Target(s): {args.target}
  Graph Mode: genus (ALL genus features, no filtering)

Key Features:
  ✅ No RFE feature selection (uses all genus features)
  ✅ No domain expert case filtering (no knowledge anchoring)
  ✅ Genus-level microbial analysis (higher taxonomic resolution)
  ✅ k-NN graph construction from genus abundances
  ✅ Multiple GNN models (GCN, GAT, RGGC) with nested CV
  ✅ GNNExplainer-based graph sparsification
  ✅ Embedding extraction from best GNN model
  ✅ Classical ML models (LinearSVR, ExtraTrees) trained on embeddings
  ✅ Metrics reported in "mean ± std" format

{'='*80}
""")

    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ ERROR: Data file not found: {args.data_path}")
        sys.exit(1)

    # Create base directories
    base_dir = create_directories()

    # Run pipeline for specified target(s)
    start_time = time.time()

    if args.target == 'both':
        targets = ['ACE-km', 'H2-km']
        print(f"Running pipeline for both targets: {targets}")
    else:
        targets = [args.target]

    all_results = {}

    for target in targets:
        print(f"\n{'#'*80}")
        print(f"# Processing Target: {target}")
        print(f"{'#'*80}")

        results = run_gnn_embeddings_pipeline(
            data_path=args.data_path,
            target=target,
            base_dir=base_dir
        )

        if results:
            all_results[target] = results
            print(f"✅ Successfully completed {target}")
        else:
            print(f"❌ Failed to complete {target}")

    # Print final summary
    elapsed_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Targets Processed: {len(all_results)}/{len(targets)}")
    print(f"Total Time: {elapsed_time/60:.2f} minutes")
    print(f"Results Directory: {base_dir}")
    print(f"\nAll results saved successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
