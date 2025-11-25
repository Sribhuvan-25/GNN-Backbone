#!/usr/bin/env python3
"""
Run GNN Embeddings Pipeline for Genus-Level Analysis (No Knowledge Anchoring)

This script runs a GNN-based embedding extraction pipeline without domain expert
case constraints (knowledge anchoring). It uses RFE-selected genus features for a
fair comparison with the baseline ML models.

Pipeline Flow:
1. Apply RFE feature selection on genus-level features (optional, recommended)
2. Build k-NN graph from selected genus abundance data
3. Train GNN models (GCN, GAT, RGGC) with nested CV hyperparameter tuning
4. Apply GNNExplainer for graph sparsification
5. Retrain GNNs on sparsified graph
6. Extract embeddings from best GNN model
7. Train classical ML models (LinearSVR, ExtraTrees) on GNN embeddings
8. Report metrics in "mean ± std" format

Usage:
    # Quick test run (5 epochs, 2 folds, no nested CV)
    python run_genus_gnn_embeddings.py --n_rfe_features 20 --quick

    # Full research run with RFE (recommended for fair comparison with baseline ML)
    python run_genus_gnn_embeddings.py --n_rfe_features 100

    # Without RFE (use all genus features)
    python run_genus_gnn_embeddings.py --n_rfe_features all

    # Custom configuration
    python run_genus_gnn_embeddings.py --n_rfe_features 100 --rfe_model_type linearsvr --target_for_rfe both

    # Custom data path
    python run_genus_gnn_embeddings.py --n_rfe_features 50 --data_path /path/to/data.csv

Note: The pipeline automatically processes ALL targets (ACE-km, H2-km) in the dataset.
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

def run_gnn_embeddings_pipeline(data_path, base_dir='results_gnn_embeddings_genus',
                                use_rfe=False, n_rfe_features=100,
                                target_for_rfe='first', rfe_model_type='extratrees',
                                epochs=200, folds=5, nested_cv=True):
    """
    Run GNN embeddings pipeline for all targets

    Parameters:
    ----------
    data_path : str
        Path to the input data file with genus-level abundance data
    base_dir : str
        Base directory for saving results
    use_rfe : bool
        If True, use RFE for feature selection before graph construction
    n_rfe_features : int
        Number of features to select using RFE (20, 40, 50, 80, or 100)
    target_for_rfe : str
        Target to use for RFE ('first', 'both', or target name)
    rfe_model_type : str
        Model type for RFE ('extratrees', 'linearsvr', etc.)
    epochs : int
        Number of training epochs (default: 200)
    folds : int
        Number of cross-validation folds (default: 5)
    nested_cv : bool
        Whether to use nested cross-validation (default: True)

    Note:
    ----
    The pipeline automatically processes ALL targets (ACE-km, H2-km) in the dataset.
    """
    print(f"\n{'='*80}")
    print(f"Running GNN Embeddings Pipeline for ALL Targets")
    print(f"{'='*80}")

    # Create save directory with RFE info
    if use_rfe:
        save_dir = f"{base_dir}_rfe{n_rfe_features}"
    else:
        save_dir = f"{base_dir}_all_features"
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
        'num_epochs': epochs,                 # Training epochs
        'patience': 20,                       # Early stopping patience
        'num_folds': folds,                   # Cross-validation folds
        'save_dir': save_dir,
        'importance_threshold': 0.2,          # Explainer threshold
        'use_fast_correlation': False,        # Use standard correlation
        'graph_mode': 'genus',                # GENUS-LEVEL ANALYSIS (not family)
        'family_filter_mode': 'strict',       # Not used for genus mode
        'use_enhanced_training': True,        # Enhanced training
        'adaptive_hyperparameters': True,     # Adaptive hyperparameters
        'use_nested_cv': nested_cv,           # Nested CV for hyperparameter tuning
        'use_node_sparsification': False,     # Edge-based sparsification only
        'graph_construction_method': 'original',
        'rfe_feature_selection': use_rfe,     # RFE feature selection
        'n_rfe_features': n_rfe_features,     # Number of RFE features
        'target_for_rfe': target_for_rfe,     # Target for RFE
        'rfe_model_type': rfe_model_type      # RFE model type
    }

    print(f"\nPipeline Configuration:")
    print(f"  Graph Mode: {config['graph_mode']} (genus-level)")
    print(f"  RFE Enabled: {use_rfe}")
    if use_rfe:
        print(f"  RFE Features: {n_rfe_features}")
        print(f"  RFE Target: {target_for_rfe}")
        print(f"  RFE Model: {rfe_model_type}")
    else:
        print(f"  Using ALL genus features (no RFE)")
    print(f"  K-Neighbors: {config['k_neighbors']}")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Folds: {config['num_folds']}")
    print(f"  Nested CV: {config['use_nested_cv']}")
    print(f"  Save Directory: {save_dir}")
    print(f"\nNOTE: No domain expert filtering (no knowledge anchoring)")

    try:
        # Initialize pipeline
        print(f"\n{'-'*60}")
        print(f"Initializing MixedEmbeddingPipeline...")
        print(f"{'-'*60}")

        pipeline = MixedEmbeddingPipeline(**config)

        # Run the full pipeline (processes all targets automatically)
        print(f"\n{'-'*60}")
        print(f"Running full GNN embeddings pipeline...")
        print(f"{'-'*60}")

        results = pipeline.run_pipeline()

        # Extract and save summary metrics for each target
        print(f"\n{'-'*60}")
        print(f"Extracting Summary Metrics")
        print(f"{'-'*60}")

        if results:
            for target_name, target_results in results.items():
                if target_name == 'summary':  # Skip summary key if present
                    continue
                if 'ml_models' in target_results:
                    save_summary_metrics(target_results, target_name, base_dir)
                else:
                    print(f"WARNING: No ML results found for {target_name}")
        else:
            print(f"WARNING: No results returned from pipeline")

        print(f"\n✅ Successfully completed pipeline for all targets")
        return results

    except Exception as e:
        print(f"\n❌ ERROR running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_summary_metrics(target_results, target, base_dir):
    """
    Save summary metrics in mean ± std format (matching RFE_Simple_CV.py format)

    Parameters:
    ----------
    target_results : dict
        Results dictionary for a specific target from pipeline
    target : str
        Target variable name
    base_dir : str
        Base directory for saving results
    """
    print(f"\nSaving summary metrics for {target}...")

    # Get ML results from target_results
    ml_results = target_results.get('ml_models', {})

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
  # Quick test run (5 epochs, 2 folds, no nested CV) - FAST!
  python run_genus_gnn_embeddings.py --n_rfe_features 20 --quick

  # Full research run with RFE (200 epochs, 5 folds, nested CV)
  python run_genus_gnn_embeddings.py --n_rfe_features 100

  # Run with ALL features (no RFE)
  python run_genus_gnn_embeddings.py --n_rfe_features all

  # Use different number of RFE features
  python run_genus_gnn_embeddings.py --n_rfe_features 50

  # Custom RFE configuration
  python run_genus_gnn_embeddings.py --n_rfe_features 100 --rfe_model_type linearsvr --target_for_rfe both

  # Use custom data path
  python run_genus_gnn_embeddings.py --n_rfe_features 100 --data_path /path/to/data.csv

Note: The pipeline automatically processes ALL targets (ACE-km, H2-km) found in the dataset.
      For fair comparison with baseline ML models, use --n_rfe_features with the same count.
"""
    )

    parser.add_argument('--data_path',
                        default='../Data/New_Data.csv',
                        help='Path to the dataset (default: ../Data/New_Data.csv)')
    parser.add_argument('--n_rfe_features',
                        type=str,
                        default='all',
                        help='Number of features to select using RFE. Use "all" for no RFE (all features), or specify a number (20, 40, 50, 80, 100). Default: all')
    parser.add_argument('--target_for_rfe',
                        default='first',
                        choices=['first', 'both'],
                        help='Target to use for RFE selection (default: first)')
    parser.add_argument('--rfe_model_type',
                        default='extratrees',
                        choices=['extratrees', 'linearsvr', 'randomforest', 'gradientboosting', 'xgboost', 'lightgbm'],
                        help='Model type for RFE feature selection (default: extratrees)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with minimal configuration (5 epochs, 2 folds, no nested CV)')

    args = parser.parse_args()

    # Parse n_rfe_features
    if args.n_rfe_features.lower() == 'all':
        use_rfe = False
        n_rfe_features = 100  # Not used, but need a value
    else:
        use_rfe = True
        try:
            n_rfe_features = int(args.n_rfe_features)
            if n_rfe_features not in [20, 40, 50, 80, 100]:
                print(f"Error: n_rfe_features must be 'all' or one of [20, 40, 50, 80, 100]")
                sys.exit(1)
        except ValueError:
            print(f"Error: n_rfe_features must be 'all' or a number (20, 40, 50, 80, 100)")
            sys.exit(1)

    # Adjust parameters for quick run
    if args.quick:
        epochs = 5
        folds = 2
        nested_cv = False
        print("🚀 QUICK TEST MODE: Running with minimal configuration")
    else:
        epochs = 200
        folds = 5
        nested_cv = True
        print("🔬 FULL RESEARCH MODE: Running with complete validation")

    # Print header
    print(f"""
{'='*80}
GNN EMBEDDINGS PIPELINE - GENUS-LEVEL ANALYSIS
{'='*80}

Mode: {'QUICK TEST (fast validation)' if args.quick else 'FULL RESEARCH (complete validation)'}

Configuration:
  Data Path: {args.data_path}
  Target(s): ALL (ACE-km, H2-km)
  Graph Mode: genus
  RFE Enabled: {use_rfe}
  {'RFE Features: ' + str(n_rfe_features) if use_rfe else 'Using ALL genus features'}
  Epochs: {epochs}
  Folds: {folds}
  Nested CV: {nested_cv}

Key Features:
  ✅ RFE feature selection: {'ENABLED (' + str(n_rfe_features) + ' features)' if use_rfe else 'DISABLED (all features)'}
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

    # Run pipeline (processes all targets automatically)
    start_time = time.time()

    print(f"\nRunning pipeline for ALL targets (ACE-km, H2-km)...")

    results = run_gnn_embeddings_pipeline(
        data_path=args.data_path,
        base_dir=base_dir,
        use_rfe=use_rfe,
        n_rfe_features=n_rfe_features,
        target_for_rfe=args.target_for_rfe,
        rfe_model_type=args.rfe_model_type,
        epochs=epochs,
        folds=folds,
        nested_cv=nested_cv
    )

    # Print final summary
    elapsed_time = time.time() - start_time

    if results:
        # Count how many targets were successfully processed
        num_targets = sum(1 for k in results.keys() if k != 'summary')

        print(f"\n{'='*80}")
        print(f"PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Targets Processed: {num_targets}")
        print(f"Total Time: {elapsed_time/60:.2f} minutes")
        print(f"Results Directory: {base_dir}")
        print(f"\nAll results saved successfully!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"❌ Pipeline failed to complete")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
