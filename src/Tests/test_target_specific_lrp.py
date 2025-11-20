#!/usr/bin/env python3
"""
Test Target-Specific LRP Implementation

This script verifies that LRP is applied separately for each target.
"""

import sys
import os

def test_target_specific_lrp():
    """Test that LRP is applied separately for ACE-km and H2-km."""
    print("="*80)
    print("TARGET-SPECIFIC LRP IMPLEMENTATION TEST")
    print("="*80)
    
    try:
        # Import dataset class
        from datasets.dataset_regression import MicrobialGNNDataset
        
        print("\n[1/4] Creating dataset with LRP enabled...")
        print("-" * 60)
        
        dataset = MicrobialGNNDataset(
            data_path='../Data/New_Data.csv',
            k_neighbors=10,
            graph_mode='genus',
            family_filter_mode='strict',
            lrp_feature_selection=True,
            n_lrp_features=20,
            target_for_lrp='first'
        )
        
        print(f"\n✅ Dataset created successfully")
        print(f"   Pre-LRP features saved: {len(dataset.pre_lrp_node_names)}")
        print(f"   Current features (no LRP applied yet): {len(dataset.node_feature_names)}")
        print(f"   LRP applied flag: {dataset.lrp_applied}")
        
        # Test 1: Apply LRP for ACE-km
        print("\n[2/4] Testing LRP for ACE-km...")
        print("-" * 60)
        
        dataset.apply_lrp_for_specific_target('ACE-km')
        
        ace_features = dataset.node_feature_names.copy()
        ace_feature_count = len(ace_features)
        
        print(f"✅ LRP applied for ACE-km")
        print(f"   Features after ACE-km LRP: {ace_feature_count}")
        print(f"   Top 5 features: {ace_features[:5]}")
        print(f"   LRP applied flag: {dataset.lrp_applied}")
        
        # Test 2: Reset to pre-LRP state
        print("\n[3/4] Testing reset to pre-LRP state...")
        print("-" * 60)
        
        dataset.reset_to_pre_lrp_state()
        
        print(f"✅ Reset to pre-LRP state")
        print(f"   Features after reset: {len(dataset.node_feature_names)}")
        print(f"   LRP applied flag: {dataset.lrp_applied}")
        
        # Test 3: Apply LRP for H2-km
        print("\n[4/4] Testing LRP for H2-km...")
        print("-" * 60)
        
        dataset.apply_lrp_for_specific_target('H2-km')
        
        h2_features = dataset.node_feature_names.copy()
        h2_feature_count = len(h2_features)
        
        print(f"✅ LRP applied for H2-km")
        print(f"   Features after H2-km LRP: {h2_feature_count}")
        print(f"   Top 5 features: {h2_features[:5]}")
        print(f"   LRP applied flag: {dataset.lrp_applied}")
        
        # Compare features
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Pre-LRP features: {len(dataset.pre_lrp_node_names)}")
        print(f"ACE-km features: {ace_feature_count}")
        print(f"H2-km features: {h2_feature_count}")
        
        # Check if features are different
        ace_set = set(ace_features)
        h2_set = set(h2_features)
        
        common_features = ace_set & h2_set
        ace_only = ace_set - h2_set
        h2_only = h2_set - ace_set
        
        print(f"\nFeature overlap:")
        print(f"  Common to both: {len(common_features)}")
        print(f"  ACE-km only: {len(ace_only)}")
        print(f"  H2-km only: {len(h2_only)}")
        
        if ace_only:
            print(f"\n  Features unique to ACE-km: {sorted(list(ace_only))[:5]}...")
        if h2_only:
            print(f"  Features unique to H2-km: {sorted(list(h2_only))[:5]}...")
        
        # Verdict
        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        
        if len(ace_only) > 0 or len(h2_only) > 0:
            print("✅ SUCCESS: Target-specific LRP is working!")
            print("   Each target gets different features optimized for that target.")
            return True
        else:
            print("⚠️  WARNING: Both targets got identical features")
            print("   This might indicate an issue with target selection.")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_target_specific_lrp()
    sys.exit(0 if success else 1)

