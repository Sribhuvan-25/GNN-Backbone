"""
Case Implementation Module for Domain Expert Cases Pipeline

This module contains the case-specific logic extracted from the DomainExpertCasesPipeline
to improve modularity, maintainability, and reduce code duplication.

Cases Overview:
- Case 1: Hydrogenotrophic features only for H2-km prediction
- Case 2: Acetoclastic features only for ACE-km prediction
- Case 3: All feature groups for both ACE-km and H2-km predictions
- Case 4: Conditional ACE-km prediction (ACE-km < 10: acetoclastic only, ACE-km >= 10: all groups)
- Case 5: Conditional H2-km prediction (H2-km < 10: hydrogenotrophic only, H2-km >= 10: all groups)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class CaseImplementations:
    """
    Container class for all case-specific implementations and logic.
    
    This class provides modular, reusable methods for executing different
    domain expert cases with reduced code duplication.
    """
    
    def __init__(self):
        """Initialize case implementations with feature group definitions."""
        
        # Domain expert feature groups
        self.acetoclastic = [
            "d__Archaea;p__Halobacterota;c__Methanosarcinia;o__Methanosarciniales;f__Methanosaetaceae;g__Methanosaeta"
        ]
        
        self.hydrogenotrophic = [
            "d__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanoregulaceae;g__Methanolinea",
            "d__Archaea;p__Euryarchaeota;c__Methanobacteria;o__Methanobacteriales;f__Methanobacteriaceae;g__Methanobacterium",
            "d__Archaea;p__Halobacterota;c__Methanomicrobia;o__Methanomicrobiales;f__Methanospirillaceae;g__Methanospirillum"
        ]
        
        self.syntrophic = [
            "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Smithellaceae;g__Smithella",
            "d__Bacteria;p__Desulfobacterota;c__Syntrophorhabdia;o__Syntrophorhabdales;f__Syntrophorhabdaceae;g__Syntrophorhabdus",
            "d__Bacteria;p__Desulfobacterota;c__Syntrophobacteria;o__Syntrophobacterales;f__Syntrophobacteraceae;g__Syntrophobacter",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Syner-01",
            "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__uncultured;g__uncultured",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__uncultured",
            "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Rikenellaceae;g__DMER64",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Thermovirga",
            "d__Bacteria;p__Firmicutes;c__Syntrophomonadia;o__Syntrophomonadales;f__Syntrophomonadaceae;g__Syntrophomonas",
            "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Syntrophaceae;g__Syntrophus",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__JGI-0000079-D21",
            "d__Bacteria;p__Desulfobacterota;c__Desulfuromonadia;o__Geobacterales;f__Geobacteraceae;__",
            "d__Bacteria;p__Firmicutes;c__Desulfotomaculia;o__Desulfotomaculales;f__Desulfotomaculales;g__Pelotomaculum"
        ]
    
    def get_case_features(self, case_type: str) -> List[str]:
        """
        Get the appropriate anchored features for a given case type.
        
        Args:
            case_type: The case type ('case1', 'case2', 'case3', 'case4', 'case5')
            
        Returns:
            List of feature names for the specified case
            
        Raises:
            ValueError: If case_type is not recognized
        """
        if case_type == 'case1':
            return self.hydrogenotrophic.copy()
        elif case_type == 'case2':
            return self.acetoclastic.copy()
        elif case_type in ['case3', 'case4', 'case5']:
            return self.acetoclastic + self.hydrogenotrophic + self.syntrophic
        else:
            raise ValueError(f"Invalid case_type: {case_type}")
    
    def get_target_index(self, target_names: List[str], target_key: str) -> int:
        """
        Find the index of a target variable in the target names list.
        
        Args:
            target_names: List of target variable names
            target_key: Key to search for ('ACE' or 'H2')
            
        Returns:
            Index of the target variable
            
        Raises:
            ValueError: If target not found
        """
        for i, target in enumerate(target_names):
            if target_key in target:
                return i
        raise ValueError(f"{target_key}-km target not found in dataset")
    
    def run_case1(self, pipeline) -> Dict[str, Any]:
        """
        Case 1: Use only hydrogenotrophic features for H2-km prediction.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing case results
        """
        print("Case 1: Using only hydrogenotrophic features for H2 dataset")
        print("Target: H2-km only")
        print(f"Anchored features: {pipeline.anchored_features}")
        
        # Find H2 target index
        h2_target_idx = self.get_target_index(pipeline.target_names, 'H2')
        
        # Run pipeline for H2 target only
        return pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
    
    def run_case2(self, pipeline) -> Dict[str, Any]:
        """
        Case 2: Use only acetoclastic features for ACE-km prediction.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing case results
        """
        print("Case 2: Using only acetoclastic features for ACE dataset")
        print("Target: ACE-km only")
        print(f"Anchored features: {pipeline.anchored_features}")
        
        # Find ACE target index
        ace_target_idx = self.get_target_index(pipeline.target_names, 'ACE')
        
        # Run pipeline for ACE target only
        return pipeline._run_single_target_pipeline(ace_target_idx, "ACE-km")
    
    def run_case3(self, pipeline) -> Dict[str, Any]:
        """
        Case 3: Use all feature groups for both ACE-km and H2-km predictions.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing results for both targets
        """
        print("Case 3: Using acetoclastic + hydrogenotrophic + syntrophic features for both targets")
        print("Targets: ACE-km and H2-km")
        print(f"Anchored features: {len(pipeline.anchored_features)} features")
        
        # Find both target indices
        ace_target_idx = self.get_target_index(pipeline.target_names, 'ACE')
        h2_target_idx = self.get_target_index(pipeline.target_names, 'H2')
        
        # Run pipeline for both targets
        results = {}
        
        print(f"\n{'='*60}")
        print("CASE 3a: ACE-km with all feature groups")
        print(f"{'='*60}")
        results['ace_km'] = pipeline._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        print(f"\n{'='*60}")
        print("CASE 3b: H2-km with all feature groups")
        print(f"{'='*60}")
        results['h2_km'] = pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        # Save combined case 3 results
        pipeline._save_case3_combined_results(results)
        
        # Create combined visualization for Case 3
        pipeline._create_case3_combined_visualization(results, ace_target_idx, h2_target_idx)
        
        return results
    
    def run_case4(self, pipeline) -> Dict[str, Any]:
        """
        Case 4: Conditional ACE-km prediction based on ACE-km values.
        ACE-km < 10: acetoclastic features only
        ACE-km >= 10: acetoclastic + hydrogenotrophic features
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing results for both subsets
        """
        print("Case 4: Conditional feature selection based on ACE-km value")
        print("ACE-km < 10: acetoclastic only")
        print("ACE-km >= 10: acetoclastic + hydrogenotrophic")
        
        # Find ACE target index
        ace_target_idx = self.get_target_index(pipeline.target_names, 'ACE')
        
        # Get ACE-km values and split data
        ace_values = []
        for data in pipeline.dataset.data_list:
            ace_values.append(data.y[0, ace_target_idx].item())
        ace_values = np.array(ace_values)
        
        # Split indices based on ACE-km values
        low_ace_indices = np.where(ace_values < 10)[0]
        high_ace_indices = np.where(ace_values >= 10)[0]
        
        print(f"Data split: {len(low_ace_indices)} samples with ACE-km < 10, {len(high_ace_indices)} samples with ACE-km >= 10")
        
        # Check if we have enough samples in each subset
        if len(low_ace_indices) < 5:
            print(f"WARNING: Only {len(low_ace_indices)} samples with ACE-km < 10. Running combined analysis instead.")
            return pipeline._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        if len(high_ace_indices) < 5:
            print(f"WARNING: Only {len(high_ace_indices)} samples with ACE-km >= 10. Running combined analysis instead.")
            return pipeline._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        # Run conditional subsets
        return self._run_conditional_case(
            pipeline=pipeline,
            target_idx=ace_target_idx,
            target_name="ACE-km",
            low_indices=low_ace_indices,
            high_indices=high_ace_indices,
            low_features=self.acetoclastic,
            high_features=self.acetoclastic + self.hydrogenotrophic,
            case_name="case4",
            low_subset_name="case4a_low_ace",
            high_subset_name="case4b_high_ace",
            low_description="ACE-km < 10 with acetoclastic features",
            high_description="ACE-km >= 10 with acetoclastic + hydrogenotrophic features",
            threshold=10
        )
    
    def run_case5(self, pipeline) -> Dict[str, Any]:
        """
        Case 5: Conditional H2-km prediction based on H2-km values.
        H2-km < 10: hydrogenotrophic features only
        H2-km >= 10: hydrogenotrophic + acetoclastic features
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing results for both subsets
        """
        print("Case 5: Conditional feature selection based on H2-km value")
        print("H2-km < 10: hydrogenotrophic only")
        print("H2-km >= 10: hydrogenotrophic + acetoclastic")
        
        # Find H2 target index
        h2_target_idx = self.get_target_index(pipeline.target_names, 'H2')
        
        # Get H2-km values and split data
        h2_values = []
        for data in pipeline.dataset.data_list:
            h2_values.append(data.y[0, h2_target_idx].item())
        h2_values = np.array(h2_values)
        
        # Split indices based on H2-km values
        low_h2_indices = np.where(h2_values < 10)[0]
        high_h2_indices = np.where(h2_values >= 10)[0]
        
        print(f"Data split: {len(low_h2_indices)} samples with H2-km < 10, {len(high_h2_indices)} samples with H2-km >= 10")
        
        # Check if we have enough samples in each subset
        if len(low_h2_indices) < 5:
            print(f"WARNING: Only {len(low_h2_indices)} samples with H2-km < 10. Running combined analysis instead.")
            return pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        if len(high_h2_indices) < 5:
            print(f"WARNING: Only {len(high_h2_indices)} samples with H2-km >= 10. Running combined analysis instead.")
            return pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        # Run conditional subsets
        return self._run_conditional_case(
            pipeline=pipeline,
            target_idx=h2_target_idx,
            target_name="H2-km",
            low_indices=low_h2_indices,
            high_indices=high_h2_indices,
            low_features=self.hydrogenotrophic,
            high_features=self.acetoclastic + self.hydrogenotrophic,
            case_name="case5",
            low_subset_name="case5a_low_h2",
            high_subset_name="case5b_high_h2",
            low_description="H2-km < 10 with hydrogenotrophic features",
            high_description="H2-km >= 10 with all feature groups",
            threshold=10
        )
    
    def _run_conditional_case(self, pipeline, target_idx: int, target_name: str,
                            low_indices: np.ndarray, high_indices: np.ndarray,
                            low_features: List[str], high_features: List[str],
                            case_name: str, low_subset_name: str, high_subset_name: str,
                            low_description: str, high_description: str,
                            threshold: float) -> Dict[str, Any]:
        """
        Generic method to run conditional cases (Case 4 and Case 5).
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            target_idx: Index of the target variable
            target_name: Name of the target variable
            low_indices: Indices of samples below threshold
            high_indices: Indices of samples above/equal threshold
            low_features: Features to use for low subset
            high_features: Features to use for high subset
            case_name: Name of the case ('case4' or 'case5')
            low_subset_name: Name for low subset results
            high_subset_name: Name for high subset results
            low_description: Description for low subset
            high_description: Description for high subset
            threshold: Threshold value used for splitting
            
        Returns:
            Dictionary containing combined results for both subsets
        """
        # Run low subset
        print(f"\n{'='*60}")
        print(f"SUBSET A: {target_name} < {threshold} → {low_description.split(' with ')[1]}")
        print(f"{'='*60}")
        
        low_results = self._run_case_subset(
            pipeline=pipeline,
            subset_indices=low_indices,
            subset_name=low_subset_name,
            anchored_features=low_features,
            description=low_description,
            target_idx=target_idx,
            target_name=target_name
        )
        
        # Run high subset
        print(f"\n{'='*60}")
        print(f"SUBSET B: {target_name} >= {threshold} → {high_description.split(' with ')[1]}")
        print(f"{'='*60}")
        
        high_results = self._run_case_subset(
            pipeline=pipeline,
            subset_indices=high_indices,
            subset_name=high_subset_name,
            anchored_features=high_features,
            description=high_description,
            target_idx=target_idx,
            target_name=target_name
        )
        
        # Combine results
        combined_results = {
            low_subset_name: low_results,
            high_subset_name: high_results,
            'data_split': {
                'low_count': len(low_indices),
                'high_count': len(high_indices),
                'low_indices': low_indices.tolist(),
                'high_indices': high_indices.tolist(),
                'threshold': threshold
            }
        }
        
        # Save combined results using appropriate method
        if case_name == 'case4':
            pipeline._save_case4_combined_results(combined_results, target_idx)
            pipeline._create_case4_combined_visualization(combined_results, target_idx)
        elif case_name == 'case5':
            pipeline._save_case5_combined_results(combined_results, target_idx)
        
        return combined_results
    
    def _run_case_subset(self, pipeline, subset_indices: np.ndarray, subset_name: str,
                        anchored_features: List[str], description: str,
                        target_idx: int, target_name: str) -> Dict[str, Any]:
        """
        Run pipeline for a specific subset of data with given anchored features.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            subset_indices: Indices of samples to include in subset
            subset_name: Name identifier for the subset
            anchored_features: Features to anchor for this subset
            description: Description of the subset
            target_idx: Index of target variable
            target_name: Name of target variable
            
        Returns:
            Dictionary containing results for the subset
        """
        print(f"Running {subset_name}: {description}")
        print(f"Subset size: {len(subset_indices)} samples")
        print(f"Anchored features: {len(anchored_features)} features")
        
        # Create detailed_results folder for this subset
        subset_save_dir = f"{pipeline.save_dir}/{subset_name}"
        os.makedirs(f"{subset_save_dir}/detailed_results", exist_ok=True)
        
        # Create subset data from the ORIGINAL dataset to maintain consistent dimensions
        subset_data_list = [pipeline.dataset.data_list[i] for i in subset_indices]
        
        # Store original values
        original_data_list = pipeline.dataset.data_list
        original_anchored_features = pipeline.anchored_features
        original_save_dir = pipeline.save_dir
        
        # Temporarily modify pipeline for subset
        pipeline.dataset.data_list = subset_data_list
        pipeline.anchored_features = anchored_features
        pipeline.save_dir = subset_save_dir
        
        try:
            # Run the single target pipeline for this subset
            subset_results = pipeline._run_single_target_pipeline(target_idx, target_name)
            
            # Add subset-specific information to results
            subset_results['subset_info'] = {
                'subset_name': subset_name,
                'description': description,
                'sample_count': len(subset_indices),
                'sample_indices': subset_indices.tolist(),
                'anchored_features_count': len(anchored_features),
                'anchored_features': anchored_features,
                'final_features_count': len(pipeline.dataset.node_feature_names) if hasattr(pipeline.dataset, 'node_feature_names') else len(anchored_features)
            }
            
        finally:
            # Restore original values
            pipeline.dataset.data_list = original_data_list
            pipeline.anchored_features = original_anchored_features
            pipeline.save_dir = original_save_dir
        
        return subset_results
    
    def apply_case_specific_filtering(self, pipeline, case_type: str) -> None:
        """
        Apply case-specific filtering logic to the pipeline.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            case_type: The case type to apply filtering for
        """
        # For now, this is a placeholder for future case-specific filtering logic
        # Each case might have different filtering requirements
        if case_type == 'case1':
            # H2-specific filtering logic could go here
            pass
        elif case_type == 'case2':
            # ACE-specific filtering logic could go here
            pass
        elif case_type in ['case3', 'case4', 'case5']:
            # Multi-target or conditional filtering logic could go here
            pass
    
    def get_case_info(self, case_type: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific case.
        
        Args:
            case_type: The case type to get information for
            
        Returns:
            Dictionary containing case information
        """
        case_info = {
            'case1': {
                'description': 'Hydrogenotrophic features only for H2-km prediction',
                'target': 'H2-km',
                'features': 'hydrogenotrophic',
                'feature_count': len(self.hydrogenotrophic)
            },
            'case2': {
                'description': 'Acetoclastic features only for ACE-km prediction',
                'target': 'ACE-km',
                'features': 'acetoclastic',
                'feature_count': len(self.acetoclastic)
            },
            'case3': {
                'description': 'All feature groups for both ACE-km and H2-km predictions',
                'target': 'ACE-km and H2-km',
                'features': 'acetoclastic + hydrogenotrophic + syntrophic',
                'feature_count': len(self.acetoclastic + self.hydrogenotrophic + self.syntrophic)
            },
            'case4': {
                'description': 'Conditional ACE-km prediction (ACE < 10: acetoclastic, ACE >= 10: all)',
                'target': 'ACE-km',
                'features': 'conditional (acetoclastic or acetoclastic+hydrogenotrophic)',
                'feature_count': 'variable'
            },
            'case5': {
                'description': 'Conditional H2-km prediction (H2 < 10: hydrogenotrophic, H2 >= 10: all)',
                'target': 'H2-km',
                'features': 'conditional (hydrogenotrophic or acetoclastic+hydrogenotrophic)',
                'feature_count': 'variable'
            }
        }
        
        return case_info.get(case_type, {'description': 'Unknown case type'})