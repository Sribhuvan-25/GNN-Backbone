#!/usr/bin/env python3
"""
Test LRP integration with the dataset pipeline.

This script tests that the new LRP implementation works correctly
with the microbial dataset preprocessing pipeline.
"""

import sys
import numpy as np

def test_lrp_integration():
    """Test LRP feature selection with dataset pipeline."""
    print("="*80)
    print("LRP INTEGRATION TEST")
    print("="*80)

    try:
        # Import dataset class
        from datasets.dataset_regression import MicrobialGNNDataset

        # Test with LRP enabled
        print("\n[1/2] Testing dataset with LRP enabled...")
        print("-" * 60)

        dataset_lrp = MicrobialGNNDataset(
            data_path='../Data/New_Data.csv',
            k_neighbors=10,
            graph_mode='genus',
            family_filter_mode='strict',
            lrp_feature_selection=True,
            n_lrp_features=20,  # Select top 20 features
            target_for_lrp='first'
        )

        print(f"\n✅ Dataset created successfully with LRP")
        print(f"   Features after LRP: {len(dataset_lrp.node_feature_names)}")
        print(f"   Samples: {len(dataset_lrp.data_list)}")
        print(f"   Graph edges: {dataset_lrp.edge_index.shape[1]}")
        print(f"   Top 5 selected features: {dataset_lrp.node_feature_names[:5]}")

        # Verify data objects are valid
        sample_data = dataset_lrp.data_list[0]
        print(f"\n   Sample data properties:")
        print(f"   - Node features: {sample_data.x.shape}")
        print(f"   - Edge index: {sample_data.edge_index.shape}")
        print(f"   - Targets: {sample_data.y.shape}")

        # Test without LRP for comparison
        print("\n[2/2] Testing dataset without LRP (for comparison)...")
        print("-" * 60)

        dataset_no_lrp = MicrobialGNNDataset(
            data_path='../Data/New_Data.csv',
            k_neighbors=10,
            graph_mode='genus',
            family_filter_mode='strict',
            lrp_feature_selection=False
        )

        print(f"\n✅ Dataset created successfully without LRP")
        print(f"   Features without LRP: {len(dataset_no_lrp.node_feature_names)}")
        print(f"   Samples: {len(dataset_no_lrp.data_list)}")
        print(f"   Graph edges: {dataset_no_lrp.edge_index.shape[1]}")

        # Compare
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Features - With LRP: {len(dataset_lrp.node_feature_names)}")
        print(f"Features - Without LRP: {len(dataset_no_lrp.node_feature_names)}")
        print(f"Reduction: {len(dataset_no_lrp.node_feature_names) - len(dataset_lrp.node_feature_names)} features")
        print(f"Reduction %: {(1 - len(dataset_lrp.node_feature_names)/len(dataset_no_lrp.node_feature_names))*100:.1f}%")

        print("\n✅ All integration tests passed!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_lrp_integration()
    sys.exit(0 if success else 1)
