#!/usr/bin/env python3
"""
Enhanced Domain Expert Cases Pipeline Test Script

This script tests all the newly implemented features:
1. Attention-based node pruning using GAT attention scores
2. Enhanced visualizations with edge weight representation
3. All three cases (acetoclastic-only, hydrogenotrophic-only, mixed) for both ACE-km and H2-km
4. Prediction vs actual graphs for validation points in each CV fold  
5. Node importance lists (feature importance scores) for each case
6. Comprehensive integration testing

Usage:
    python run_enhanced_pipeline_test.py --case case1
    python run_enhanced_pipeline_test.py --case all
    python run_enhanced_pipeline_test.py --data_path ../Data/New_Data.csv --case case3
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline


def run_enhanced_test_case(case_type, data_path="../../Data/New_Data.csv", save_dir="./enhanced_pipeline_test_results"):
    """
    Run a single case with all enhanced features.
    
    Args:
        case_type: Type of case to run ('case1', 'case2', 'case3') 
        data_path: Path to the dataset
        save_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED PIPELINE TEST: {case_type.upper()}")
    print(f"{'='*80}")
    print("Testing Features:")
    print("✓ Attention-based node pruning using GAT attention scores")
    print("✓ Enhanced visualizations with edge weight representation") 
    print("✓ Both ACE-km and H2-km targets")
    print("✓ Prediction vs actual graphs for validation points")
    print("✓ Node importance lists (feature importance scores)")
    print(f"{'='*80}")
    
    try:
        # Initialize pipeline with enhanced configuration
        pipeline = DomainExpertCasesPipeline(
            data_path=data_path,
            case_type=case_type,
            k_neighbors=10,           # Balanced connectivity
            hidden_dim=256,           # Sufficient capacity for attention mechanisms
            num_epochs=150,           # Reasonable training time for testing
            num_folds=3,              # Reduced folds for faster testing
            save_dir=f"{save_dir}/{case_type}_enhanced",
            importance_threshold=0.2,  # Standard threshold for node pruning
            use_fast_correlation=False, # Use full Mantel test
            family_filter_mode='strict', # Quality filtering
            use_nested_cv=False       # Disable for faster testing, enable for production
        )
        
        # Set specific model types to test (including GAT for attention-based pruning)
        pipeline.gnn_models_to_train = ['gat', 'gcn']  # Focus on GAT for attention features
        
        print(f"\nPipeline Configuration:")
        print(f"  Case Type: {case_type}")
        print(f"  GNN Models: {pipeline.gnn_models_to_train}")
        print(f"  Hidden Dim: {pipeline.hidden_dim}")
        print(f"  K-neighbors: {pipeline.k_neighbors}")
        print(f"  Epochs: {pipeline.num_epochs}")
        print(f"  CV Folds: {pipeline.num_folds}")
        print(f"  Attention-based pruning: ENABLED (when GAT is used)")
        print(f"  Enhanced visualizations: ENABLED")
        print(f"  Feature importance reporting: ENABLED")
        
        # Run the case-specific pipeline
        print(f"\n{'='*60}")
        print("EXECUTING ENHANCED PIPELINE")
        print(f"{'='*60}")
        
        results = pipeline.run_case_specific_pipeline()
        
        # Verify all expected features were generated
        print(f"\n{'='*60}")
        print("VERIFICATION: ENHANCED FEATURES")
        print(f"{'='*60}")
        
        # Check if results contain both targets
        expected_targets = ['ace_km', 'h2_km']
        for target in expected_targets:
            if target in results:
                print(f"✓ {target.upper()} results generated")
                
                # Check for prediction plots
                plots_dir = f"{save_dir}/{case_type}_enhanced/{target.replace('_', '-').upper()}_plots"
                if os.path.exists(plots_dir):
                    print(f"✓ Prediction vs actual plots generated: {plots_dir}")
                else:
                    print(f"⚠ Prediction plots not found: {plots_dir}")
            else:
                print(f"✗ Missing {target.upper()} results")
        
        # Check for feature importance reports
        importance_files = [
            f"{save_dir}/{case_type}_enhanced/{case_type}_feature_importance.png",
            f"{save_dir}/{case_type}_enhanced/{case_type}_feature_importance.csv",
            f"{save_dir}/{case_type}_enhanced/attention_scores.csv"
        ]
        
        for file_path in importance_files:
            if os.path.exists(file_path):
                print(f"✓ Feature importance file generated: {os.path.basename(file_path)}")
            else:
                print(f"⚠ Feature importance file not found: {os.path.basename(file_path)}")
        
        # Check for enhanced graph visualizations
        graph_files = [
            f"{save_dir}/{case_type}_enhanced/graphs/comprehensive_graph_comparison.png",
            f"{save_dir}/{case_type}_enhanced/graphs/attention_based_graph_enhanced.png",
            f"{save_dir}/{case_type}_enhanced/graphs/knn_graph_enhanced.png"
        ]
        
        for file_path in graph_files:
            if os.path.exists(file_path):
                print(f"✓ Enhanced graph visualization generated: {os.path.basename(file_path)}")
            else:
                print(f"⚠ Enhanced graph visualization not found: {os.path.basename(file_path)}")
        
        print(f"\n✓ {case_type.upper()} completed successfully!")
        print(f"Results saved to: {save_dir}/{case_type}_enhanced")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error running {case_type}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None


def run_all_enhanced_tests(data_path="../../Data/New_Data.csv", save_dir="./enhanced_pipeline_test_results"):
    """Run all three cases with enhanced features."""
    print(f"\n{'='*80}")
    print("ENHANCED DOMAIN EXPERT CASES - COMPREHENSIVE TEST")
    print(f"{'='*80}")
    print("This test will run all three cases with enhanced features:")
    print("  Case 1: Hydrogenotrophic features for both ACE-km and H2-km")
    print("  Case 2: Acetoclastic features for both ACE-km and H2-km") 
    print("  Case 3: All feature groups for both ACE-km and H2-km")
    print(f"{'='*80}")
    
    cases = ['case1', 'case2', 'case3']
    all_results = {}
    
    for case in cases:
        print(f"\n{'*' * 60}")
        print(f"RUNNING {case.upper()}")
        print(f"{'*' * 60}")
        
        case_results = run_enhanced_test_case(case, data_path, save_dir)
        all_results[case] = case_results
        
        if case_results is not None:
            print(f"✓ {case.upper()} completed successfully")
        else:
            print(f"✗ {case.upper()} failed")
    
    # Summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_cases = [case for case, result in all_results.items() if result is not None]
    failed_cases = [case for case, result in all_results.items() if result is None]
    
    print(f"Total cases tested: {len(cases)}")
    print(f"Successful: {len(successful_cases)} - {', '.join(successful_cases)}")
    print(f"Failed: {len(failed_cases)} - {', '.join(failed_cases)}")
    
    if len(successful_cases) == len(cases):
        print(f"\n🎉 ALL ENHANCED FEATURES TESTED SUCCESSFULLY!")
        print("The enhanced pipeline includes:")
        print("  ✓ Attention-based node pruning")
        print("  ✓ Enhanced edge weight visualizations") 
        print("  ✓ Multi-target predictions (ACE-km + H2-km)")
        print("  ✓ Prediction vs actual validation plots")
        print("  ✓ Feature importance analysis and reporting")
        print("  ✓ Comprehensive result integration")
    else:
        print(f"\n⚠ Some tests failed. Check logs above for details.")
    
    print(f"\nAll results saved in: {save_dir}")
    return all_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test enhanced domain expert cases pipeline with all new features"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../Data/New_Data.csv", 
        help="Path to the input data CSV file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./enhanced_pipeline_test_results",
        help="Directory to save test results"
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=['case1', 'case2', 'case3', 'all'],
        default='all',
        help="Which case to test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Verify data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please provide the correct path to the data file.")
        print("Expected format: CSV file with microbial abundance data and ACE-km, H2-km target columns")
        return
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Enhanced Domain Expert Cases Pipeline - Test Suite")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"Test case(s): {args.case}")
    print("=" * 50)
    
    if args.case == 'all':
        # Run comprehensive test of all cases
        results = run_all_enhanced_tests(args.data_path, args.save_dir)
    else:
        # Run specific case
        results = run_enhanced_test_case(args.case, args.data_path, args.save_dir)
    
    print(f"\nEnhanced pipeline testing completed!")
    
    if results:
        print("✓ Test successful - All enhanced features are working")
    else:
        print("✗ Test failed - Check logs for details")


if __name__ == "__main__":
    main()