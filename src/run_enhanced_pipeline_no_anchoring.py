#!/usr/bin/env python3
"""
Run Enhanced Edge-Based Sparsification Pipeline WITHOUT Feature Anchoring

This script is identical to run_enhanced_pipeline.py but with feature anchoring DISABLED.
Feature anchoring normally protects certain domain-expert-selected features from being
removed during RFE selection or graph pruning. This version runs the pipeline without
any protected/anchored features.

Key Differences from run_enhanced_pipeline.py:
- NO feature anchoring (all features treated equally)
- NO protected nodes during RFE selection
- NO protected nodes during graph sparsification
- NO special visualization for anchored features

Usage:
    python run_enhanced_pipeline_no_anchoring.py [--case case1] [--epochs 100] [--use_rfe] [--n_rfe_features 100]

Examples:
    # Run full pipeline with case 1 (hydrogenotrophic focus) - NO RFE, NO ANCHORING
    python run_enhanced_pipeline_no_anchoring.py --case case1

    # Run with RFE feature selection (select top 40 features) - NO ANCHORING
    python run_enhanced_pipeline_no_anchoring.py --case case1 --use_rfe --n_rfe_features 40

    # Quick test run with RFE (minimal epochs) - NO ANCHORING
    python run_enhanced_pipeline_no_anchoring.py --case case1 --use_rfe --n_rfe_features 20 --quick

    # Custom configuration with 80 RFE features and combined target - NO ANCHORING
    python run_enhanced_pipeline_no_anchoring.py --case case2 --epochs 50 --use_rfe --n_rfe_features 80 --target_for_rfe both --rfe_model_type linearsvr
"""

import argparse
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run Enhanced Node Pruning Pipeline WITHOUT Feature Anchoring')
    parser.add_argument('--case', default='case1', choices=['case1', 'case2', 'case3', 'case4', 'case5', 'all'],
                        help='Domain expert case to run (default: case1, use "all" to run all cases)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with minimal configuration')
    parser.add_argument('--data_path', default='../Data/New_Data.csv',
                        help='Path to the dataset (default: ../Data/New_Data.csv)')
    parser.add_argument('--graph_method', default='original',
                        choices=['original', 'hybrid'],
                        help='Graph construction method (default: original, paper_correlation removed)')
    parser.add_argument('--use_rfe', action='store_true',
                        help='Enable RFE feature selection before graph construction')
    parser.add_argument('--n_rfe_features', type=int, default=100, choices=[20, 40, 50, 80, 100],
                        help='Number of features to select using RFE (default: 100)')
    parser.add_argument('--target_for_rfe', default='first', choices=['first', 'both'],
                        help='Target to use for RFE selection (default: first)')
    parser.add_argument('--rfe_model_type', default='extratrees',
                        choices=['extratrees', 'linearsvr', 'randomforest', 'gradientboosting', 'xgboost', 'lightgbm'],
                        help='Model type for RFE feature selection (default: extratrees)')
    parser.add_argument('--importance_threshold', type=float, default=0.5,
                        help='Threshold for explainer edge importance (default: 0.5 = keep top 50%% of edges)')
    parser.add_argument('--no_knn_sparsification', action='store_false', dest='use_knn_sparsification',
                        help='Disable k-NN sparsification (use full correlation graph for ablation study, default: enabled)')

    args = parser.parse_args()

    # Handle "all" cases option
    if args.case == 'all':
        return run_all_cases(args)

    # Adjust parameters for quick run
    if args.quick:
        epochs = 5
        folds = 2
        nested_cv = False
        print("🚀 QUICK TEST MODE: Running with minimal configuration")
    else:
        epochs = args.epochs
        folds = 5
        nested_cv = True
        print("🔬 FULL RESEARCH MODE: Running with complete validation")

    print(f"""
{'='*80}
ENHANCED EDGE-BASED SPARSIFICATION PIPELINE (NO FEATURE ANCHORING)
{'='*80}
Case: {args.case}
Epochs: {epochs}
Folds: {folds}
Nested CV: {nested_cv}
Data: {args.data_path}
Graph Mode: genus (genus-level analysis for higher taxonomic resolution)
KNN Sparsification: {'Enabled' if args.use_knn_sparsification else 'DISABLED (Ablation: Full Correlation Graph)'}
Explainer Sparsification: Edge-based using GNNExplainer (NO node pruning)
RFE Feature Selection: {'Enabled' if args.use_rfe else 'Disabled'} ({args.n_rfe_features} features if enabled, model: {args.rfe_model_type})
Feature Anchoring: DISABLED (all features treated equally)
{'='*80}

Key Features Enabled:
✅ Edge-based sparsification with GNNExplainer (NO node pruning)
✅ Genus-level microbial analysis (higher taxonomic resolution)
✅ Statistical validation with significance testing
✅ Enhanced Graph Transformer with proper architecture
✅ Comprehensive baseline comparisons
✅ Biological validation with pathway enrichment
✅ Ablation studies for component analysis
{f'✅ RFE feature selection ({args.n_rfe_features} features, {args.rfe_model_type})' if args.use_rfe else '⚠️  RFE feature selection disabled (using all features)'}
⚠️  Feature anchoring DISABLED (no protected nodes)
{'='*80}
""")

    try:
        from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline

        # Configuration - identical to original but with anchoring disabled
        config = {
            'data_path': args.data_path,
            'case_type': args.case,
            'num_epochs': epochs,
            'num_folds': folds,
            'use_nested_cv': nested_cv,
            'save_dir': f'no_anchor_results_{args.case}',  # Different save directory
            'k_neighbors': 10,              # Increased for better connectivity
            'hidden_dim': 64,
            'dropout_rate': 0.2,            # Reduced for limited data
            'batch_size': 4,                # Reduced to prevent CUDA OOM errors
            'learning_rate': 0.001,         # Lower for stability
            'patience': 30 if not args.quick else 5,  # More patience for convergence
            'importance_threshold': 0.5,    # Keep 50% of edges (was 30%)
            'graph_construction_method': 'original',  # Always 'original' (handles RFE internally)
            'use_node_pruning': False,
            'weight_decay': 1e-4,
            'family_filter_mode': 'strict',
            'use_rfe_feature_selection': args.use_rfe,
            'n_rfe_features': args.n_rfe_features,
            'target_for_rfe': args.target_for_rfe,
            'rfe_model_type': args.rfe_model_type,
            'use_knn_sparsification': args.use_knn_sparsification,
            # NEW: Disable feature anchoring
            'disable_anchoring': True,  # This tells the pipeline not to use anchored features
        }

        print("Initializing enhanced pipeline WITHOUT feature anchoring...")
        start_time = time.time()

        # Initialize and run pipeline
        pipeline = DomainExpertCasesPipeline(**config)

        # CRITICAL: Override anchored features to disable anchoring
        # This ensures no features are protected during RFE or graph pruning
        pipeline.anchored_features = []
        if hasattr(pipeline.dataset, 'anchored_features'):
            pipeline.dataset.anchored_features = []
        if hasattr(pipeline.dataset, 'protected_nodes'):
            pipeline.dataset.protected_nodes = []

        print(f"✅ Pipeline initialized successfully!")
        print(f"⚠️  Feature anchoring DISABLED: protected_nodes = {getattr(pipeline.dataset, 'protected_nodes', [])}")

        print(f"Dataset: {len(pipeline.dataset.data_list)} samples with {len(pipeline.dataset.node_feature_names)} features")

        # Run the complete pipeline
        print("\n🚀 Starting pipeline execution...")
        results = pipeline.run_case_specific_pipeline()

        end_time = time.time()
        runtime = end_time - start_time

        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION COMPLETED (NO ANCHORING)!")
        print(f"{'='*80}")
        print(f"Runtime: {runtime/60:.2f} minutes")
        print(f"Results saved to: {pipeline.save_dir}")

        # Print key results
        if results:
            print(f"\nKey Results:")
            for target_name, target_results in results.items():
                if isinstance(target_results, dict):
                    print(f"\n{target_name.upper()}:")

                    # Performance metrics
                    if 'knn_training' in target_results:
                        knn_scores = []
                        for model_key, model_data in target_results['knn_training'].items():
                            if 'test_metrics' in model_data:
                                knn_scores.append(model_data['test_metrics'].get('r2_score', 0))
                        if knn_scores:
                            print(f"  Original k-NN performance: R² = {max(knn_scores):.4f}")

                    if 'explainer_training' in target_results:
                        explainer_scores = []
                        for model_key, model_data in target_results['explainer_training'].items():
                            if 'test_metrics' in model_data:
                                explainer_scores.append(model_data['test_metrics'].get('r2_score', 0))
                        if explainer_scores:
                            print(f"  Enhanced pruned performance: R² = {max(explainer_scores):.4f}")
                            if knn_scores:
                                improvement = max(explainer_scores) - max(knn_scores)
                                print(f"  Improvement: {improvement:+.4f}")

                    # Validation results
                    if 'comprehensive_validation' in target_results:
                        validation = target_results['comprehensive_validation']

                        if 'statistical_validation' in validation:
                            stat_val = validation['statistical_validation']
                            print(f"  Statistical significance: p = {stat_val.get('p_value', 'N/A')}")
                            print(f"  Effect size: d = {stat_val.get('effect_size', 'N/A')}")

                        if 'biological_validation' in validation:
                            bio_val = validation['biological_validation']
                            overall_score = bio_val.get('overall_biological_validity', {}).get('overall_score', 'N/A')
                            print(f"  Biological validity: {overall_score}")

        print(f"\n🎉 Enhanced edge-based sparsification pipeline completed successfully (NO ANCHORING)!")
        print(f"📁 Check results directory: {pipeline.save_dir}")
        print(f"📊 Validation results include statistical tests and biological pathway analysis")
        print(f"🔬 Analysis performed at genus-level with edge-based graph sparsification")
        print(f"⚠️  NO feature anchoring - all features treated equally")

        if args.quick:
            print(f"\n💡 For full research results, run without --quick flag")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        print(f"Please check that the data file exists at: {args.data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_all_cases(args):
    """Run all domain expert cases (case1, case2, case3) with enhanced pipeline WITHOUT anchoring"""
    print("="*80)
    print("RUNNING ALL DOMAIN EXPERT CASES WITHOUT FEATURE ANCHORING")
    print("="*80)
    print("Features enabled:")
    print("✓ Edge-based sparsification using GNNExplainer (NO node pruning)")
    print("✓ Genus-level microbial analysis (higher taxonomic resolution)")
    print("✓ Spearman correlation graph initialization")
    print("✗ NO anchored features (all features treated equally)")
    print("✓ Working transformer models")
    print("✓ Comprehensive graph visualizations")
    print("="*80)

    cases = ['case1', 'case2', 'case3']
    all_results = {}
    total_start_time = time.time()

    for i, case in enumerate(cases, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()} ({i}/{len(cases)}) - NO ANCHORING")
        print(f"{'='*60}")

        # Create a copy of args for this case
        case_args = argparse.Namespace(**vars(args))
        case_args.case = case

        try:
            # Adjust parameters for quick run
            if args.quick:
                epochs = 5
                folds = 2
                nested_cv = False
                print("🚀 QUICK TEST MODE: Running with minimal configuration")
            else:
                epochs = args.epochs
                folds = 5
                nested_cv = True
                print("🔬 FULL RESEARCH MODE: Running with complete validation")

            print(f"Case: {case}")
            print(f"Epochs: {epochs}")
            print(f"Graph Method: {args.graph_method}")
            print(f"Feature Anchoring: DISABLED")

            # Import and configure pipeline
            from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline

            config = {
                'data_path': args.data_path,
                'case_type': case,
                'num_epochs': epochs,
                'num_folds': folds,
                'use_nested_cv': nested_cv,
                'save_dir': f'no_anchor_results_{case}',  # Different save directory
                'k_neighbors': 10,
                'hidden_dim': 64,
                'dropout_rate': 0.3,
                'batch_size': 4,  # Reduced to prevent CUDA OOM errors
                'learning_rate': 0.001,
                'patience': 20 if not args.quick else 5,
                'importance_threshold': args.importance_threshold,
                'graph_construction_method': 'original',  # Always 'original' (handles RFE internally)
                'use_rfe_feature_selection': args.use_rfe,
                'n_rfe_features': args.n_rfe_features,
                'target_for_rfe': args.target_for_rfe,
                'rfe_model_type': args.rfe_model_type,
                'use_knn_sparsification': args.use_knn_sparsification,
                'disable_anchoring': True,  # Disable anchoring
            }

            start_time = time.time()
            pipeline = DomainExpertCasesPipeline(**config)

            # CRITICAL: Override anchored features to disable anchoring
            pipeline.anchored_features = []
            if hasattr(pipeline.dataset, 'anchored_features'):
                pipeline.dataset.anchored_features = []
            if hasattr(pipeline.dataset, 'protected_nodes'):
                pipeline.dataset.protected_nodes = []

            results = pipeline.run_case_specific_pipeline()
            end_time = time.time()

            all_results[case] = results
            runtime = end_time - start_time

            print(f"\n✅ {case.upper()} completed successfully (NO ANCHORING)!")
            print(f"Runtime: {runtime/60:.2f} minutes")
            print(f"Results saved to: {pipeline.save_dir}")

        except Exception as e:
            print(f"❌ {case.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[case] = None

    # Summary
    total_runtime = time.time() - total_start_time
    print(f"\n{'='*80}")
    print("ALL CASES EXECUTION SUMMARY (NO ANCHORING)")
    print(f"{'='*80}")

    successful_cases = [case for case, result in all_results.items() if result is not None]
    failed_cases = [case for case, result in all_results.items() if result is None]

    if successful_cases:
        print(f"✅ Successfully completed: {', '.join(successful_cases)}")

    if failed_cases:
        print(f"❌ Failed cases: {', '.join(failed_cases)}")

    print(f"🕐 Total runtime: {total_runtime/60:.2f} minutes")
    print(f"📁 Results saved to no_anchor_results_case1/, no_anchor_results_case2/, no_anchor_results_case3/")
    print(f"🔬 Success rate: {len(successful_cases)}/{len(cases)} ({len(successful_cases)/len(cases)*100:.1f}%)")
    print(f"⚠️  All runs completed WITHOUT feature anchoring")

    if args.quick:
        print(f"\n💡 For full research results, run without --quick flag")

    return all_results


if __name__ == "__main__":
    main()
