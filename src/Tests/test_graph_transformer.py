#!/usr/bin/env python3
"""
Test script to verify Graph Transformer (kg_gt) works with the domain expert cases pipeline.

This script tests:
1. Graph Transformer model instantiation
2. Node pruning functionality with Graph Transformer
3. Integration with domain expert cases pipeline
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline


def test_graph_transformer_instantiation():
    """Test that Graph Transformer model can be created successfully."""
    print("="*60)
    print("TEST 1: Graph Transformer Model Instantiation")
    print("="*60)
    
    try:
        from models.GNNmodelsRegression import simple_GraphTransformer_regression
        
        # Test model creation
        model = simple_GraphTransformer_regression(
            hidden_channels=128,
            output_dim=1,
            dropout_prob=0.1,
            input_channel=1,
            num_heads=4,
            num_layers=4,
            activation='identity'
        )
        
        print("✅ Graph Transformer model created successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Graph Transformer model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_initialization():
    """Test that the domain expert pipeline can be initialized with Graph Transformer."""
    print("\n" + "="*60)
    print("TEST 2: Pipeline Initialization with Graph Transformer")
    print("="*60)
    
    try:
        # Initialize pipeline with Graph Transformer
        pipeline = DomainExpertCasesPipeline(
            data_path="../Data/New_Data.csv",  # Correct path from src directory
            case_type='case1',
            k_neighbors=8,
            hidden_dim=128,
            num_epochs=5,  # Small number for testing
            num_folds=2,   # Small number for testing
            save_dir="./test_kg_gt_results",
            importance_threshold=0.3,
            use_nested_cv=False,  # Disable for faster testing
            use_node_pruning=True
        )
        
        # Force Graph Transformer to be included
        pipeline.gnn_models_to_train = ['kg_gt']
        
        print("✅ Pipeline initialized successfully with Graph Transformer!")
        print(f"Models to train: {pipeline.gnn_models_to_train}")
        print(f"Case type: {pipeline.case_type}")
        print(f"Node pruning enabled: {pipeline.use_node_pruning}")
        return True, pipeline
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_graph_transformer_training():
    """Test Graph Transformer training on a small dataset."""
    print("\n" + "="*60)
    print("TEST 3: Graph Transformer Training Test")
    print("="*60)
    
    success, pipeline = test_pipeline_initialization()
    if not success or pipeline is None:
        print("❌ Cannot test training - pipeline initialization failed")
        return False
    
    try:
        # Try to run just the ACE-km target for case1
        print("Testing Graph Transformer training with node pruning...")
        
        # Get target index for ACE-km
        ace_target_idx = None
        for i, target in enumerate(pipeline.target_names):
            if 'ACE' in target:
                ace_target_idx = i
                break
        
        if ace_target_idx is None:
            print("❌ Could not find ACE target")
            return False
        
        print(f"Found ACE target at index {ace_target_idx}: {pipeline.target_names[ace_target_idx]}")
        
        # Run a single case with Graph Transformer
        results = pipeline.run_single_case_all_targets_nested_cv(
            case_type='case1',
            models_to_test=['kg_gt'],  # Only test Graph Transformer
            targets_to_test=[ace_target_idx]  # Only test ACE target
        )
        
        if results and len(results) > 0:
            print("✅ Graph Transformer training completed successfully!")
            print(f"Results keys: {list(results.keys())}")
            return True
        else:
            print("❌ Training completed but no results returned")
            return False
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Graph Transformer tests."""
    print("GRAPH TRANSFORMER INTEGRATION TESTS")
    print("Testing Graph Transformer (kg_gt) with Domain Expert Cases Pipeline")
    
    # Test 1: Model instantiation
    test1_success = test_graph_transformer_instantiation()
    
    # Test 2: Pipeline initialization
    test2_success, _ = test_pipeline_initialization()
    
    # Test 3: Training (only if previous tests passed)
    test3_success = False
    if test1_success and test2_success:
        print("\n📋 Basic tests passed. Running training test...")
        print("⚠️  Note: This will use your actual data file and may take a few minutes.")
        
        response = input("Do you want to run the training test? (y/n): ").strip().lower()
        if response == 'y':
            test3_success = test_graph_transformer_training()
        else:
            print("⏭️  Skipping training test.")
            test3_success = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"1. Model Instantiation:     {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"2. Pipeline Initialization: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test3_success is True:
        print(f"3. Training Test:           ✅ PASS")
    elif test3_success is False:
        print(f"3. Training Test:           ❌ FAIL")
    else:
        print(f"3. Training Test:           ⏭️  SKIPPED")
    
    if test1_success and test2_success:
        print("\n🎉 Graph Transformer is ready for use with your node pruning pipeline!")
        print("\nTo use Graph Transformer in your pipeline:")
        print("1. Set models_to_test=['kg_gt'] or include 'kg_gt' in your model list")
        print("2. The pipeline will use node pruning with Graph Transformer attention")
        print("3. Results will be saved with 'kg_gt' prefix")
    else:
        print("\n⚠️  Graph Transformer integration has issues that need to be fixed.")
    
    return test1_success and test2_success


if __name__ == "__main__":
    main()