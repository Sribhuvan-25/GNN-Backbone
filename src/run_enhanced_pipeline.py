#!/usr/bin/env python3
"""
Run Enhanced Edge-Based Sparsification Pipeline

This script demonstrates how to run the enhanced domain expert pipeline
with comprehensive validation framework including:
- Edge-based sparsification using GNNExplainer (NO node pruning)
- Family-level microbial analysis only
- Statistical validation with significance testing
- Enhanced Graph Transformer architecture
- Baseline comparisons (PageRank, Integrated Gradients, etc.)
- Biological validation with pathway enrichment
- Ablation studies for component analysis

Usage:
    python run_enhanced_pipeline.py [--case case1] [--epochs 100] [--quick]

Examples:
    # Run full pipeline with case 1 (hydrogenotrophic focus)
    python run_enhanced_pipeline.py --case case1

    # Keep top 80% of edges in explainer pruning (default is 20%)
    python run_enhanced_pipeline.py --case case1 --importance_threshold 0.8

    # Quick test run (minimal epochs)
    python run_enhanced_pipeline.py --case case1 --quick

    # Custom configuration with 50% edge retention
    python run_enhanced_pipeline.py --case case2 --epochs 50 --importance_threshold 0.5
"""

import argparse
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run Enhanced Node Pruning Pipeline')
    parser.add_argument('--case', default='case1', choices=['case1', 'case2', 'case3', 'case4', 'case5', 'all'],
                        help='Domain expert case to run (default: case1, use "all" to run all cases)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with minimal configuration')
    parser.add_argument('--data_path', default='../Data/New_Data.csv',
                        help='Path to the dataset (default: ../Data/New_Data.csv)')
    parser.add_argument('--graph_method', default='paper_correlation',
                        choices=['original', 'paper_correlation', 'hybrid'],
                        help='Graph construction method (default: paper_correlation)')
    parser.add_argument('--importance_threshold', type=float, default=0.8,
                        help='Threshold for explainer edge importance (default: 0.2 = keep top 20%% of edges)')

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
ENHANCED EDGE-BASED SPARSIFICATION PIPELINE
{'='*80}
Case: {args.case}
Epochs: {epochs}
Folds: {folds}
Nested CV: {nested_cv}
Data: {args.data_path}
Graph Mode: family (family-level analysis only)
Sparsification: Edge-based using GNNExplainer
{'='*80}

Key Features Enabled:
✅ Edge-based sparsification with GNNExplainer (NO node pruning)
✅ Family-level microbial analysis only
✅ Statistical validation with significance testing
✅ Enhanced Graph Transformer with proper architecture
✅ Comprehensive baseline comparisons
✅ Biological validation with pathway enrichment
✅ Ablation studies for component analysis
{'='*80}
""")
    
    try:
        from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline
        
        # Configuration
        # config = {
        #     'data_path': args.data_path,
        #     'case_type': args.case,
        #     'num_epochs': epochs,
        #     'num_folds': folds,
        #     'use_nested_cv': nested_cv,
        #     'save_dir': f'enhanced_results_{args.case}',
        #     'k_neighbors': 10,
        #     'hidden_dim': 64,
        #     'dropout_rate': 0.3,
        #     'batch_size': 8,
        #     'learning_rate': 0.001,
        #     'patience': 20 if not args.quick else 5,
        #     'importance_threshold': args.importance_threshold,
        #     'graph_construction_method': args.graph_method,  # User-selected graph construction method
        #     'use_node_pruning': False  # ✅ EDGE-ONLY SPARSIFICATION
        # }
        config = {
            'data_path': args.data_path,
            'case_type': args.case,
            'num_epochs': epochs,
            'num_folds': folds,
            'use_nested_cv': nested_cv,
            'save_dir': f'enhanced_results_{args.case}',
            'k_neighbors': 10,
            'hidden_dim': 64,
            'dropout_rate': 0.3,        # CHANGE TO 0.2
            'batch_size': 8,            # CHANGE TO 16
            'learning_rate': 0.003,     # CHANGE TO 0.01
            'patience': 25 if not args.quick else 5,  # CHANGE TO 50
            'importance_threshold': args.importance_threshold,
            'graph_construction_method': args.graph_method,
            'use_node_pruning': False,
            'weight_decay': 1e-4        # CHANGE TO 1e-5
        }
        
        print("Initializing enhanced pipeline...")
        start_time = time.time()
        
        # Initialize and run pipeline
        pipeline = DomainExpertCasesPipeline(**config)
        print(f"✅ Pipeline initialized successfully!")
        
        print(f"Dataset: {len(pipeline.dataset.data_list)} samples with {len(pipeline.dataset.node_feature_names)} features")
        
        # Run the complete pipeline
        print("\n🚀 Starting pipeline execution...")
        results = pipeline.run_case_specific_pipeline()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION COMPLETED!")
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
        
        print(f"\n🎉 Enhanced edge-based sparsification pipeline completed successfully!")
        print(f"📁 Check results directory: {pipeline.save_dir}")
        print(f"📊 Validation results include statistical tests and biological pathway analysis")
        print(f"🔬 Analysis performed at family-level with edge-based graph sparsification")
        
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
    """Run all domain expert cases (case1, case2, case3) with enhanced pipeline"""
    print("="*80)
    print("RUNNING ALL DOMAIN EXPERT CASES WITH ENHANCED PIPELINE")
    print("="*80)
    print("Features enabled:")
    print("✓ Edge-based sparsification using GNNExplainer (NO node pruning)")
    print("✓ Family-level microbial analysis only")
    print("✓ Spearman correlation graph initialization")
    print("✓ Protected anchored features during edge sparsification")
    print("✓ Working transformer models")
    print("✓ Comprehensive graph visualizations")
    print("="*80)

    cases = ['case1', 'case2', 'case3']
    all_results = {}
    total_start_time = time.time()

    for i, case in enumerate(cases, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()} ({i}/{len(cases)})")
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

            # Import and configure pipeline
            from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline

            config = {
                'data_path': args.data_path,
                'case_type': case,
                'num_epochs': epochs,
                'num_folds': folds,
                'use_nested_cv': nested_cv,
                'save_dir': f'enhanced_results_{case}',
                'k_neighbors': 10,
                'hidden_dim': 64,
                'dropout_rate': 0.3,
                'batch_size': 8,
                'learning_rate': 0.001,
                'patience': 20 if not args.quick else 5,
                'importance_threshold': args.importance_threshold,
                'graph_construction_method': args.graph_method
            }

            start_time = time.time()
            pipeline = DomainExpertCasesPipeline(**config)
            results = pipeline.run_case_specific_pipeline()
            end_time = time.time()

            all_results[case] = results
            runtime = end_time - start_time

            print(f"\n✅ {case.upper()} completed successfully!")
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
    print("ALL CASES EXECUTION SUMMARY")
    print(f"{'='*80}")

    successful_cases = [case for case, result in all_results.items() if result is not None]
    failed_cases = [case for case, result in all_results.items() if result is None]

    if successful_cases:
        print(f"✅ Successfully completed: {', '.join(successful_cases)}")

    if failed_cases:
        print(f"❌ Failed cases: {', '.join(failed_cases)}")

    print(f"🕐 Total runtime: {total_runtime/60:.2f} minutes")
    print(f"📁 Results saved to enhanced_results_case1/, enhanced_results_case2/, enhanced_results_case3/")
    print(f"🔬 Success rate: {len(successful_cases)}/{len(cases)} ({len(successful_cases)/len(cases)*100:.1f}%)")

    if args.quick:
        print(f"\n💡 For full research results, run without --quick flag")

    return all_results


if __name__ == "__main__":
    main()