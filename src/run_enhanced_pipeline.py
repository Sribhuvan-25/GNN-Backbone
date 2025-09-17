#!/usr/bin/env python3
"""
Run Enhanced Node Pruning Pipeline

This script demonstrates how to run the enhanced domain expert pipeline
with comprehensive validation framework including:
- Mathematical formulation for unified node pruning
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
    
    # Quick test run (minimal epochs)
    python run_enhanced_pipeline.py --case case1 --quick
    
    # Custom configuration
    python run_enhanced_pipeline.py --case case2 --epochs 50
"""

import argparse
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run Enhanced Node Pruning Pipeline')
    parser.add_argument('--case', default='case1', choices=['case1', 'case2', 'case3', 'case4', 'case5'],
                        help='Domain expert case to run (default: case1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with minimal configuration')
    parser.add_argument('--data_path', default='../Data/New_Data.csv',
                        help='Path to the dataset (default: ../Data/New_Data.csv)')
    parser.add_argument('--graph_method', default='paper_correlation',
                        choices=['original', 'paper_correlation', 'hybrid'],
                        help='Graph construction method (default: paper_correlation)')
    
    args = parser.parse_args()
    
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
ENHANCED NODE PRUNING PIPELINE
{'='*80}
Case: {args.case}
Epochs: {epochs}
Folds: {folds}
Nested CV: {nested_cv}
Data: {args.data_path}
{'='*80}

Key Features Enabled:
✅ Mathematical formulation for unified node importance scoring
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
        config = {
            'data_path': args.data_path,
            'case_type': args.case,
            'num_epochs': epochs,
            'num_folds': folds,
            'use_nested_cv': nested_cv,
            'save_dir': f'enhanced_results_{args.case}',
            'k_neighbors': 10,
            'hidden_dim': 64,
            'dropout_rate': 0.3,
            'batch_size': 8,
            'learning_rate': 0.001,
            'patience': 20 if not args.quick else 5,
            'graph_mode': 'family',
            'graph_construction_method': args.graph_method  # User-selected graph construction method
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
        
        print(f"\n🎉 Enhanced node pruning pipeline completed successfully!")
        print(f"📁 Check results directory: {pipeline.save_dir}")
        print(f"📊 Validation results include statistical tests and biological pathway analysis")
        
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


if __name__ == "__main__":
    main()