"""
Domain Expert Dataset Module for Microbial GNN Analysis.

This module provides the AnchoredMicrobialGNNDataset class which extends the base
MicrobialGNNDataset with anchored feature support for domain expert case studies.
"""

import os
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import base dataset class from same datasets directory
from datasets.dataset_regression import MicrobialGNNDataset

# Import utilities
from utils.taxonomy_utils import (
    extract_family_from_taxonomy,
    extract_family_from_column_name,
    aggregate_otus_to_families,
    convert_to_relative_abundance,
    apply_family_filtering
)
from utils.result_management import (
    create_results_directory_structure,
    save_fold_results
)


class AnchoredMicrobialGNNDataset(MicrobialGNNDataset):
    """
    Extended dataset class with anchored feature support for domain expert cases.
    
    This class extends the base MicrobialGNNDataset to support anchored features
    that are guaranteed to be included in the final feature set regardless of
    statistical filtering. This is useful for domain expert cases where specific
    microbial families are known to be important for particular metabolic pathways.
    
    Attributes:
        anchored_features (list): List of taxonomic strings for anchored features
        case_type (str): Type of case study (e.g., 'case1', 'case2', etc.)
    """
    
    def __init__(self, data_path, anchored_features=None, case_type=None,
                 k_neighbors=5, mantel_threshold=0.05, use_fast_correlation=False,
                 graph_mode='family', family_filter_mode='relaxed',
                 graph_construction_method='original'):
        """
        Initialize the anchored microbial GNN dataset.
        
        Args:
            data_path (str): Path to the CSV file containing microbial abundance data
            anchored_features (list, optional): List of taxonomic strings for anchored features
            case_type (str, optional): Type of case study
            k_neighbors (int): Number of neighbors for KNN graph construction
            mantel_threshold (float): P-value threshold for Mantel test
            use_fast_correlation (bool): If True, use fast correlation-based graph construction
            graph_mode (str): Mode for graph construction ('otu' or 'family')
            family_filter_mode (str): Mode for family filtering ('strict', 'relaxed', 'permissive')
        """
        # Store anchored features and case type
        self.anchored_features = anchored_features or []
        self.case_type = case_type
        
        # Initialize base class
        super().__init__(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode,
            family_filter_mode=family_filter_mode,
            graph_construction_method=graph_construction_method
        )
    
    def _process_families(self):
        """
        Extended family processing with anchored features support.
        
        This method aggregates OTUs to family level, applies standard filtering,
        and then adds anchored features based on the case type.
        
        Returns:
            tuple: (family_dataframe, feature_names_list)
        """
        print(f"Processing families for {self.case_type or 'standard'} analysis...")
        
        # Aggregate OTUs to families using utility function
        df_fam, family_to_cols = aggregate_otus_to_families(self.df, self.otu_cols)
        
        # Convert to relative abundance
        df_fam_rel = convert_to_relative_abundance(df_fam)
        
        print(f"Total families before filtering: {df_fam_rel.shape[1]}")
        
        # Apply standard filtering first using utility function
        df_fam_rel_filtered, selected_families = apply_family_filtering(
            df_fam_rel, 
            filter_mode=self.family_filter_mode
        )
        
        print(f"Families after standard filtering: {df_fam_rel_filtered.shape[1]}")
        
        # Add anchored features based on case type
        if self.anchored_features and self.case_type:
            df_fam_rel_filtered = self._add_anchored_features(df_fam_rel, df_fam_rel_filtered)
        
        return df_fam_rel_filtered, list(df_fam_rel_filtered.columns)
    
    def _add_anchored_features(self, df_fam_rel, df_fam_rel_filtered):
        """
        Add case-specific anchored features to the Mantel-selected features.
        
        This method ensures that domain expert specified families are included
        in the final feature set even if they don't pass statistical filtering.
        
        Args:
            df_fam_rel (pd.DataFrame): Full family relative abundance data
            df_fam_rel_filtered (pd.DataFrame): Filtered family data
            
        Returns:
            pd.DataFrame: Enhanced dataset with anchored features
        """
        print(f"\nAdding case-specific anchored features for {self.case_type}...")
        
        # Get the anchored family names for this case
        anchored_family_names = []
        for taxonomy in self.anchored_features:
            family_name = extract_family_from_taxonomy(taxonomy)
            if family_name:
                anchored_family_names.append(family_name)
        
        print(f"Looking for anchored families: {anchored_family_names}")
        
        # Find matching families in the data
        matched_families = []
        for family_name in anchored_family_names:
            # Look for exact matches first
            if family_name in df_fam_rel.columns:
                matched_families.append(family_name)
                print(f"  Found exact match: {family_name}")
            else:
                # Look for partial matches
                partial_matches = [col for col in df_fam_rel.columns if family_name in col]
                if partial_matches:
                    matched_families.extend(partial_matches)
                    print(f"  Found partial matches for {family_name}: {partial_matches}")
                else:
                    print(f"  WARNING: No match found for {family_name}")
        
        print(f"Matched anchored families: {matched_families}")
        
        # Add anchored families to the existing Mantel-selected features
        # This preserves all Mantel-selected features and adds anchors
        anchors_added = 0
        for family in matched_families:
            if family not in df_fam_rel_filtered.columns:
                df_fam_rel_filtered[family] = df_fam_rel[family]
                print(f"  Added anchored family: {family}")
                anchors_added += 1
            else:
                print(f"  Anchored family already present in Mantel-selected features: {family}")
        
        print(f"Added {anchors_added} new anchored features to {df_fam_rel_filtered.shape[1] - anchors_added} Mantel-selected features")
        print(f"Final feature count: {df_fam_rel_filtered.shape[1]} families")
        print(f"Final feature set: Mantel-selected + Case-specific anchors")
        
        return df_fam_rel_filtered
    
    def get_feature_info(self):
        """
        Get detailed information about features in the dataset.
        
        Returns:
            dict: Information about features including anchored and filtered features
        """
        info = {
            'case_type': self.case_type,
            'total_features': len(self.node_feature_names),
            'feature_names': self.node_feature_names.copy(),
            'anchored_features_input': self.anchored_features.copy(),
            'graph_mode': self.graph_mode,
            'family_filter_mode': self.family_filter_mode,
            'k_neighbors': self.k_neighbors,
            'mantel_threshold': self.mantel_threshold
        }
        
        if hasattr(self, 'target_names'):
            info['target_names'] = self.target_names.copy()
        
        return info
    
    def save_dataset_info(self, save_dir):
        """
        Save dataset information to file.
        
        Args:
            save_dir (str): Directory to save the information
        """
        # Create directory structure
        dir_paths = create_results_directory_structure(save_dir, f"dataset_{self.case_type or 'standard'}")
        
        # Get feature info
        feature_info = self.get_feature_info()
        
        # Save as JSON
        import json
        with open(os.path.join(dir_paths['base'], 'dataset_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save feature names as CSV
        features_df = pd.DataFrame({
            'feature_name': self.node_feature_names,
            'feature_index': range(len(self.node_feature_names))
        })
        features_df.to_csv(os.path.join(dir_paths['base'], 'features.csv'), index=False)
        
        print(f"Dataset information saved to {dir_paths['base']}")
        
        return dir_paths
    
    def get_case_specific_summary(self):
        """
        Get a summary specific to the case type.
        
        Returns:
            dict: Case-specific summary information
        """
        summary = {
            'case_type': self.case_type,
            'dataset_size': len(self.data_list),
            'feature_count': len(self.node_feature_names),
            'target_count': len(self.target_names) if hasattr(self, 'target_names') else 0,
            'anchored_features_count': len(self.anchored_features),
            'graph_properties': {
                'k_neighbors': self.k_neighbors,
                'mantel_threshold': self.mantel_threshold,
                'graph_mode': self.graph_mode,
                'family_filter_mode': self.family_filter_mode
            }
        }
        
        # Add target information if available
        if hasattr(self, 'target_names'):
            summary['targets'] = self.target_names
        
        # Add anchored feature information
        if self.anchored_features:
            anchored_families = []
            for taxonomy in self.anchored_features:
                family_name = extract_family_from_taxonomy(taxonomy)
                if family_name:
                    anchored_families.append(family_name)
            summary['anchored_families'] = anchored_families
        
        return summary
    
    def __repr__(self):
        """String representation of the dataset."""
        return (f"AnchoredMicrobialGNNDataset(case_type='{self.case_type}', "
                f"features={len(self.node_feature_names) if hasattr(self, 'node_feature_names') else 0}, "
                f"samples={len(self.data_list) if hasattr(self, 'data_list') else 0}, "
                f"anchored_features={len(self.anchored_features)})")