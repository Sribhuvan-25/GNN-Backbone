#!/usr/bin/env python3
"""
Test script to run domain expert cases 1, 2, and 3 for both ACE-km and H2-km targets.

This script demonstrates the updated functionality with:
- Signed edge weights in graph representation
- Improved visualizations with uniform node sizes
- All cases running for both target values
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline


def run_all_cases(data_path="./Data/New_data.csv", save_dir="./domain_expert_results_updated"):
    """Run all domain expert cases (1, 2, 3) for both targets with enhanced features."""
    print("="*80)
    print("RUNNING ENHANCED DOMAIN EXPERT CASES")
    print("="*80)
    print("Enhanced features implemented:")
    print("✓ Attention-based node pruning using GAT attention scores")
    print("✓ Enhanced visualizations with edge weight representation")
    print("✓ Cases 1, 2, 3 run for both ACE-km and H2-km targets")
    print("✓ Prediction vs actual graphs for validation points in each CV fold")
    print("✓ Node importance lists (feature importance scores) generated")
    print("✓ Comprehensive result integration and reporting")
    print("="*80)
    
    cases = ['case1', 'case2', 'case3']
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()}")
        print(f"{'='*60}")
        
        try:
            # Initialize pipeline with optimized parameters for faster execution
            pipeline = DomainExpertCasesPipeline(
                data_path=data_path,
                case_type=case,
                k_neighbors=8,            # Reduced for faster graph construction
                hidden_dim=64,            # Reduced for faster training
                num_epochs=50,            # Reduced for faster training
                num_folds=5,              # 5-fold cross-validation
                save_dir=f"{save_dir}/{case}_results",
                importance_threshold=0.3, # Node pruning threshold
                use_fast_correlation=True,     # Use fast correlation for speed
                family_filter_mode='relaxed',  # Relaxed filtering for speed
                use_nested_cv=False       # Disable nested CV for speed
            )
            
            # Ensure GAT is included for attention-based pruning
            if hasattr(pipeline, 'gnn_models_to_train'):
                if 'gat' not in pipeline.gnn_models_to_train:
                    pipeline.gnn_models_to_train.append('gat')
            
            # Run the case-specific pipeline
            case_results = pipeline.run_case_specific_pipeline()
            all_results[case] = case_results
            
            print(f"\n{case.upper()} completed successfully!")
            print(f"Results saved to: {save_dir}/{case}_results")
            
        except Exception as e:
            print(f"Error running {case}: {e}")
            import traceback
            traceback.print_exc()
            all_results[case] = None
    
    # Summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    successful_cases = [case for case, result in all_results.items() if result is not None]
    failed_cases = [case for case, result in all_results.items() if result is None]
    
    if successful_cases:
        print(f"✓ Successfully completed cases: {', '.join(successful_cases)}")
    
    if failed_cases:
        print(f"✗ Failed cases: {', '.join(failed_cases)}")
    
    print(f"\nResults saved in: {save_dir}")
    print(f"Total cases processed: {len(cases)}")
    print(f"Successful: {len(successful_cases)}")
    print(f"Failed: {len(failed_cases)}")
    
    return all_results


def main():
    """Main function to run domain expert cases."""
    parser = argparse.ArgumentParser(description="Run domain expert cases 1, 2, 3 for both targets")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../Data/New_Data.csv",
        help="Path to the input data CSV file"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./domain_expert_results_updated",
        help="Directory to save results"
    )
    parser.add_argument(
        "--case", 
        type=str, 
        choices=['case1', 'case2', 'case3', 'all'],
        default='all',
        help="Which case to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please provide the correct path to the data file.")
        return
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.case == 'all':
        # Run all cases
        results = run_all_cases(args.data_path, args.save_dir)
    else:
        # Run specific case
        print(f"Running {args.case} only...")
        try:
            pipeline = DomainExpertCasesPipeline(
                data_path=args.data_path,
                case_type=args.case,
                k_neighbors=10,
                hidden_dim=256,
                num_epochs=100,
                num_folds=5,
                save_dir=f"{args.save_dir}/{args.case}_results",
                importance_threshold=0.3,
                use_fast_correlation=False,
                family_filter_mode='relaxed'
            )
            
            results = pipeline.run_case_specific_pipeline()
            print(f"{args.case} completed successfully!")
            
        except Exception as e:
            print(f"Error running {args.case}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()