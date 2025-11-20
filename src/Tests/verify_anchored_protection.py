#!/usr/bin/env python3
"""
Verify that all 17 anchored features are protected during LRP.
"""

import sys

def verify_anchored_protection():
    """Verify all anchored features are in the final feature set after LRP."""
    print("="*80)
    print("VERIFYING ANCHORED FEATURE PROTECTION AFTER LRP")
    print("="*80)
    
    try:
        from datasets.domain_expert_dataset import AnchoredMicrobialGNNDataset
        from pipelines.case_implementations import CaseImplementations
        
        # Get Case 3 anchored features
        case_impl = CaseImplementations()
        case3_full_paths = case_impl.get_case_features('case3')
        
        print(f"\n[1] Case 3 has {len(case3_full_paths)} anchored features (full paths)")
        
        # Extract genus names from full paths
        case3_genus_names = []
        for path in case3_full_paths:
            genus = path.split(';g__')[-1]
            case3_genus_names.append(genus)
        
        print(f"[2] Extracted {len(case3_genus_names)} genus names")
        print(f"[3] Unique genus names: {len(set(case3_genus_names))}")
        
        # Check for duplicates
        from collections import Counter
        genus_counts = Counter(case3_genus_names)
        duplicates = {k: v for k, v in genus_counts.items() if v > 1}
        
        if duplicates:
            print(f"\n⚠️  Found duplicate genus names:")
            for genus, count in duplicates.items():
                print(f"   - '{genus}' appears {count} times")
                # Find which full paths have this genus
                matching_paths = [p for p in case3_full_paths if p.endswith(f'g__{genus}')]
                for i, path in enumerate(matching_paths, 1):
                    print(f"     {i}. {path}")
        
        print(f"\n[4] Creating dataset with LRP for Case 3...")
        print("-" * 60)
        
        # Create dataset with LRP
        dataset = AnchoredMicrobialGNNDataset(
            data_path='../Data/New_Data.csv',
            anchored_features=case3_full_paths,
            case_type='case3',
            k_neighbors=10,
            graph_mode='genus',
            family_filter_mode='strict',
            lrp_feature_selection=True,
            n_lrp_features=20,
            target_for_lrp='first'
        )
        
        print(f"\n[5] Dataset created:")
        print(f"   Pre-LRP features: {len(dataset.pre_lrp_node_names)}")
        print(f"   Protected nodes: {len(dataset.protected_nodes)}")
        print(f"   Unique protected nodes: {len(set(dataset.protected_nodes))}")
        
        # Apply LRP for ACE-km
        print(f"\n[6] Applying LRP for ACE-km...")
        print("-" * 60)
        dataset.apply_lrp_for_specific_target('ACE-km')
        
        final_features = set(dataset.node_feature_names)
        protected_features = set(dataset.protected_nodes)
        
        print(f"\n[7] After LRP:")
        print(f"   Total features: {len(final_features)}")
        print(f"   Protected features: {len(protected_features)}")
        
        # Check which protected features are in final set
        protected_in_final = protected_features & final_features
        protected_missing = protected_features - final_features
        
        print(f"\n[8] Protection verification:")
        print(f"   Protected features in final set: {len(protected_in_final)}/{len(protected_features)}")
        
        if protected_missing:
            print(f"\n   ❌ MISSING protected features: {len(protected_missing)}")
            for feat in sorted(protected_missing):
                print(f"      - {feat}")
        else:
            print(f"\n   ✅ ALL protected features are in final set!")
        
        # Final verdict
        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)
        
        unique_protected = len(set(dataset.protected_nodes))
        unique_in_final = len(protected_in_final)
        
        print(f"Anchored features defined: {len(case3_full_paths)} (17 full paths)")
        print(f"Unique genus names: {len(set(case3_genus_names))}")
        print(f"Protected nodes set: {len(dataset.protected_nodes)}")
        print(f"Unique protected nodes: {unique_protected}")
        print(f"Protected nodes in final feature set: {unique_in_final}/{unique_protected}")
        
        if unique_in_final == unique_protected:
            print(f"\n✅ SUCCESS: All {unique_protected} unique protected features are in final set!")
            return True
        else:
            print(f"\n❌ PROBLEM: Only {unique_in_final}/{unique_protected} protected features in final set")
            return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_anchored_protection()
    sys.exit(0 if success else 1)

