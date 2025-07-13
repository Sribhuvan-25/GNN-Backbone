#!/usr/bin/env python3
"""
Quick test script for nested CV pipeline with minimal computational requirements
"""

import os
import sys
import torch
import datetime
import traceback
from domain_expert_cases_pipeline_nested_cv import run_all_cases_nested_cv

def print_system_info():
    """Print system information for debugging"""
    print("="*80)
    print("NESTED CV PIPELINE - QUICK TEST")
    print("="*80)
    print(f"Start time: {datetime.datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {memory_gb:.1f} GB")
    else:
        print("CUDA not available - using CPU")
    
    print("="*80)

def main():
    """Main function with error handling"""
    try:
        print_system_info()
        
        # QUICK TEST CONFIGURATION
        print("Running QUICK TEST with minimal computational requirements...")
        print("Configuration:")
        print("  - Outer CV folds: 2 (instead of 5)")
        print("  - Inner CV folds: 2 (instead of 5)")  
        print("  - Max hyperparameter combinations: 3 (instead of 12)")
        print("  - Epochs: 50 (instead of 200)")
        print("  - Patience: 10 (instead of 20)")
        print()
        
        # Create pipeline with test configuration
        from domain_expert_cases_pipeline_nested_cv import NestedCVDomainExpertCasesPipeline
        
        # Quick test parameters
        pipeline = NestedCVDomainExpertCasesPipeline(
            data_path="../Data/New_data.csv",
            case_type='case1',
            # REDUCED PARAMETERS FOR QUICK TEST
            num_folds=2,                    # 2 outer folds instead of 5
            inner_cv_folds=2,               # 2 inner folds instead of 5
            max_hyperparameter_combinations=1,  # 3 combinations instead of 12
            num_epochs=5,                  # 50 epochs instead of 200
            patience=10,                    # 10 patience instead of 20
            # Other parameters
            k_neighbors=5,
            hidden_dim=64,
            dropout_rate=0.3,
            batch_size=8,
            learning_rate=0.01,
            weight_decay=1e-4,
            save_dir='./nested_cv_quick_test',
            importance_threshold=0.2,
            use_fast_correlation=True,      # Use fast correlation for speed
            graph_mode='family',
            family_filter_mode='relaxed'    # Use relaxed filtering for more features
        )
        
        # Run only case1 for quick test
        print("Running CASE1 only for quick test...")
        results = pipeline._run_case1()
        
        print("\n" + "="*80)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print quick summary
        if results and 'knn_nested_cv' in results:
            print("Quick Test Results Summary:")
            for model_type, model_results in results['knn_nested_cv'].items():
                if model_results and 'avg_metrics' in model_results:
                    r2 = model_results['avg_metrics']['r2']
                    r2_std = model_results['avg_metrics']['r2_std']
                    mse = model_results['avg_metrics']['mse']
                    rmse = model_results['avg_metrics']['rmse']
                    print(f"  {model_type.upper()}: R² = {r2:.4f}±{r2_std:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}")
        
        print(f"\nResults saved to: {pipeline.save_dir}")
        print("Check the directory for plots and detailed results.")
        
    except Exception as e:
        print(f"\nERROR during quick test: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 1
    
    finally:
        end_time = datetime.datetime.now()
        print(f"\nEnd time: {end_time}")
        if 'start_time' in locals():
            duration = end_time - start_time
            print(f"Total execution time: {duration}")
        print("="*80)
    
    return 0

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    sys.exit(main()) 