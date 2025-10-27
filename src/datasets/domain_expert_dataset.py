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
    apply_family_filtering,
    extract_genus_from_taxonomy,
    extract_genus_from_column_name,
    aggregate_otus_to_genera,
    convert_to_relative_abundance_genus,
    apply_genus_filtering
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
                 graph_mode='genus', family_filter_mode='relaxed',
                 graph_construction_method='original', save_dir=None):
        """
        Initialize the anchored microbial GNN dataset.

        Args:
            data_path (str): Path to the CSV file containing microbial abundance data
            anchored_features (list, optional): List of taxonomic strings for anchored features
            case_type (str, optional): Type of case study
            k_neighbors (int): Number of neighbors for KNN graph construction
            mantel_threshold (float): P-value threshold for Mantel test
            use_fast_correlation (bool): If True, use fast correlation-based graph construction
            graph_mode (str): Mode for graph construction ('otu', 'family', or 'genus')
            family_filter_mode (str): Mode for taxonomic filtering ('strict', 'relaxed', 'permissive')
        """
        # Store anchored features, case type, and save directory
        self.anchored_features = anchored_features or []
        self.case_type = case_type
        self.save_dir = save_dir

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
        Extended taxonomic processing with anchored features support.

        This method aggregates OTUs to family/genus level, applies standard filtering,
        and then adds anchored features based on the case type.

        Returns:
            tuple: (taxonomic_dataframe, feature_names_list)
        """
        if self.graph_mode == 'genus':
            print(f"Processing genera for {self.case_type or 'standard'} analysis...")

            # Aggregate OTUs to genus level using utility function
            df_tax, tax_to_cols = aggregate_otus_to_genera(self.df, self.otu_cols)

            # Convert to relative abundance
            df_tax_rel = convert_to_relative_abundance_genus(df_tax)

            print(f"Total genera before filtering: {df_tax_rel.shape[1]}")

            # Apply standard filtering first using utility function
            df_tax_rel_filtered, selected_taxa = apply_genus_filtering(
                df_tax_rel,
                filter_mode=self.family_filter_mode
            )

            print(f"Genera after standard filtering: {df_tax_rel_filtered.shape[1]}")

        else:  # family mode
            print(f"Processing families for {self.case_type or 'standard'} analysis...")

            # Aggregate OTUs to families using utility function
            df_tax, tax_to_cols = aggregate_otus_to_families(self.df, self.otu_cols)

            # Convert to relative abundance
            df_tax_rel = convert_to_relative_abundance(df_tax)

            print(f"Total families before filtering: {df_tax_rel.shape[1]}")

            # Apply standard filtering first using utility function
            df_tax_rel_filtered, selected_taxa = apply_family_filtering(
                df_tax_rel,
                filter_mode=self.family_filter_mode
            )

            print(f"Families after standard filtering: {df_tax_rel_filtered.shape[1]}")

        # Add anchored features based on case type
        if self.anchored_features and self.case_type:
            df_tax_rel_filtered = self._add_anchored_features(df_tax_rel, df_tax_rel_filtered)

        return df_tax_rel_filtered, list(df_tax_rel_filtered.columns)

    def _process_genera(self):
        """
        Override genus processing to add anchored features support.
        This mirrors _process_families() but is called when graph_mode='genus'.

        Returns:
            tuple: (genus_dataframe, feature_names_list)
        """
        # Call the parent class method to do standard genus processing
        df_genus_rel_filtered, selected_genera = super()._process_genera()

        # Now add anchored features if specified
        if self.anchored_features and self.case_type:
            # Get the full unfiltered genus data to add anchored features
            # We need to reconstruct df_genus_rel from the raw data
            from utils.taxonomy_utils import (
                aggregate_otus_to_genera,
                convert_to_relative_abundance_genus
            )

            df_genus, _ = aggregate_otus_to_genera(self.df, self.otu_cols)
            df_genus_rel = convert_to_relative_abundance_genus(df_genus)

            # Add anchored features
            df_genus_rel_filtered = self._add_anchored_features(df_genus_rel, df_genus_rel_filtered)
            selected_genera = list(df_genus_rel_filtered.columns)

        return df_genus_rel_filtered, selected_genera

    def _add_anchored_features(self, df_tax_rel, df_tax_rel_filtered):
        """
        Add case-specific anchored features to the filtered features.

        This method ensures that domain expert specified taxa (genus/family) are included
        in the final feature set even if they don't pass statistical filtering.

        Args:
            df_tax_rel (pd.DataFrame): Full taxonomic relative abundance data
            df_tax_rel_filtered (pd.DataFrame): Filtered taxonomic data

        Returns:
            pd.DataFrame: Enhanced dataset with anchored features
        """
        print(f"\nAdding case-specific anchored features for {self.case_type}...")

        # Get the anchored taxonomic names for this case based on graph mode
        anchored_tax_names = []
        if self.graph_mode == 'genus':
            for taxonomy in self.anchored_features:
                genus_name = extract_genus_from_taxonomy(taxonomy)
                if genus_name:
                    anchored_tax_names.append(genus_name)
            tax_level = "genera"
        else:  # family mode
            for taxonomy in self.anchored_features:
                family_name = extract_family_from_taxonomy(taxonomy)
                if family_name:
                    anchored_tax_names.append(family_name)
            tax_level = "families"

        print(f"Looking for anchored {tax_level}: {anchored_tax_names}")

        # Find matching taxa in the data
        matched_taxa = []
        for tax_name in anchored_tax_names:
            # Look for exact matches first
            if tax_name in df_tax_rel.columns:
                matched_taxa.append(tax_name)
                print(f"  Found exact match: {tax_name}")
            else:
                # Look for partial matches
                partial_matches = [col for col in df_tax_rel.columns if tax_name in col]
                if partial_matches:
                    matched_taxa.extend(partial_matches)
                    print(f"  Found partial matches for {tax_name}: {partial_matches}")
                else:
                    print(f"  WARNING: No match found for {tax_name}")

        print(f"Matched anchored {tax_level}: {matched_taxa}")

        # Add anchored taxa to the existing filtered features
        # This preserves all filtered features and adds anchors
        anchors_added = 0
        for taxon in matched_taxa:
            if taxon not in df_tax_rel_filtered.columns:
                df_tax_rel_filtered[taxon] = df_tax_rel[taxon]
                print(f"  Added anchored {tax_level[:-1]}: {taxon}")
                anchors_added += 1
            else:
                print(f"  Anchored {tax_level[:-1]} already present in filtered features: {taxon}")

        print(f"Added {anchors_added} new anchored features to {df_tax_rel_filtered.shape[1] - anchors_added} filtered features")
        print(f"Final feature count: {df_tax_rel_filtered.shape[1]} {tax_level}")
        print(f"Final feature set: Filtered + Case-specific anchors")

        return df_tax_rel_filtered
    
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