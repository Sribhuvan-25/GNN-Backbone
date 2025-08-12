"""
Case Implementation Module for Domain Expert Cases Pipeline

This module contains the case-specific logic extracted from the DomainExpertCasesPipeline
to improve modularity, maintainability, and reduce code duplication.

Cases Overview:
- Case 1: Hydrogenotrophic features for both ACE-km and H2-km predictions
- Case 2: Acetoclastic features for both ACE-km and H2-km predictions  
- Case 3: All feature groups (acetoclastic + hydrogenotrophic + syntrophic) for both ACE-km and H2-km predictions
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
            case_type: The case type ('case1', 'case2', 'case3')
            
        Returns:
            List of feature names for the specified case
            
        Raises:
            ValueError: If case_type is not recognized
        """
        if case_type == 'case1':
            return self.hydrogenotrophic.copy()
        elif case_type == 'case2':
            return self.acetoclastic.copy()
        elif case_type == 'case3':
            return self.acetoclastic + self.hydrogenotrophic + self.syntrophic
        else:
            raise ValueError(f"Invalid case_type: {case_type}. Valid options are: case1, case2, case3")
    
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
        Case 1: Use only hydrogenotrophic features for both ACE-km and H2-km predictions.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing results for both targets
        """
        print("Case 1: Using only hydrogenotrophic features for both targets")
        print("Targets: ACE-km and H2-km")
        print(f"Anchored features: {pipeline.anchored_features}")
        
        # Find both target indices
        ace_target_idx = self.get_target_index(pipeline.target_names, 'ACE')
        h2_target_idx = self.get_target_index(pipeline.target_names, 'H2')
        
        # Run pipeline for both targets
        results = {}
        
        print(f"\n{'='*60}")
        print("CASE 1a: ACE-km with hydrogenotrophic features")
        print(f"{'='*60}")
        results['ace_km'] = pipeline._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        print(f"\n{'='*60}")
        print("CASE 1b: H2-km with hydrogenotrophic features")
        print(f"{'='*60}")
        results['h2_km'] = pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        return results
    
    def run_case2(self, pipeline) -> Dict[str, Any]:
        """
        Case 2: Use only acetoclastic features for both ACE-km and H2-km predictions.
        
        Args:
            pipeline: The DomainExpertCasesPipeline instance
            
        Returns:
            Dictionary containing results for both targets
        """
        print("Case 2: Using only acetoclastic features for both targets")
        print("Targets: ACE-km and H2-km")
        print(f"Anchored features: {pipeline.anchored_features}")
        
        # Find both target indices
        ace_target_idx = self.get_target_index(pipeline.target_names, 'ACE')
        h2_target_idx = self.get_target_index(pipeline.target_names, 'H2')
        
        # Run pipeline for both targets
        results = {}
        
        print(f"\n{'='*60}")
        print("CASE 2a: ACE-km with acetoclastic features")
        print(f"{'='*60}")
        results['ace_km'] = pipeline._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        print(f"\n{'='*60}")
        print("CASE 2b: H2-km with acetoclastic features")
        print(f"{'='*60}")
        results['h2_km'] = pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        return results
    
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
