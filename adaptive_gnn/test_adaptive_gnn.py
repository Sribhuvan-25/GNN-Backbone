#!/usr/bin/env python3
"""
Test Adaptive Microbial GNN Implementation
==========================================

Simple test to verify the adaptive GNN pipeline works correctly.
"""

import os
from adaptive_microbial_pipeline import AdaptiveMicrobialPipeline


def test_adaptive_pipeline():
    """
    Test the adaptive pipeline with sample data
    """
    print("🧪 Testing Adaptive Microbial GNN Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AdaptiveMicrobialPipeline()
    
    # Check for data file in multiple possible locations
    possible_data_paths = [
        "../Given_Code/selected_features_with_targets.csv",
        "../../Given_Code/selected_features_with_targets.csv", 
        "./data/selected_features_with_targets.csv",
        "../Data/selected_features_with_targets.csv"
    ]
    
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print(f"❌ Data file not found in any of these locations:")
        for path in possible_data_paths:
            print(f"   - {path}")
        
        print("\nLet's check what CSV files are available:")
        try:
            # Check current directory and parent directories
            for search_dir in [".", "..", "../.."]:
                if os.path.exists(search_dir):
                    for root, dirs, files in os.walk(search_dir):
                        for file in files:
                            if file.endswith('.csv'):
                                print(f"   - {os.path.join(root, file)}")
        except:
            print("   - Could not search for files")
        return False
    
    print(f"📂 Found data file: {data_path}")
    
    try:
        # Load data
        pipeline.load_data(data_path)
        
        # Test data object creation
        data_objects = pipeline.create_data_objects()
        print(f"✅ Created {len(data_objects)} data objects")
        
        # Test model training on single target (quick test)
        print(f"\n🧠 Quick training test...")
        
        # Reduce epochs for quick test
        test_config = {
            'hidden_dim': 32,
            'epochs': 5,
            'batch_size': 8,
            'patience': 3
        }
        
        model, losses, config = pipeline.train_adaptive_model(
            target_name='ACE-km',
            target_idx=0, 
            data_objects=data_objects,
            model_config=test_config
        )
        
        print(f"✅ Model training completed")
        print(f"   - Final loss: {losses[-1]:.6f}")
        
        # Test embedding extraction
        embeddings = pipeline.extract_embeddings(model, data_objects, 'ACE-km')
        print(f"✅ Embeddings extracted: {embeddings.shape}")
        
        # Test learned graph analysis
        learned_graphs = pipeline.analyze_learned_graphs(model, data_objects, 'ACE-km')
        print(f"✅ Graph analysis completed: {len(learned_graphs)} structures")
        
        print(f"\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_adaptive_pipeline()
    
    if success:
        print(f"\n✅ Adaptive GNN implementation is working correctly!")
        print(f"🚀 Ready to run the full pipeline with:")
        print(f"   python adaptive_microbial_pipeline.py")
    else:
        print(f"\n❌ Tests failed. Please check the implementation.") 