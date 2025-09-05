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
        
        # Reset dataset to original state between targets for independent processing
        print(f"\n{'='*60}")
        print("RESETTING DATASET BETWEEN TARGETS")
        print(f"{'='*60}")
        pipeline.dataset.reset_to_original_state()
        
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
        
        # Reset dataset to original state between targets for independent processing
        print(f"\n{'='*60}")
        print("RESETTING DATASET BETWEEN TARGETS")
        print(f"{'='*60}")
        pipeline.dataset.reset_to_original_state()
        
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
        
        # Reset dataset to original state between targets for independent processing
        print(f"\n{'='*60}")
        print("RESETTING DATASET BETWEEN TARGETS")
        print(f"{'='*60}")
        pipeline.dataset.reset_to_original_state()
        
        print(f"\n{'='*60}")
        print("CASE 3b: H2-km with all feature groups")
        print(f"{'='*60}")
        results['h2_km'] = pipeline._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        # Save combined case 3 results
        self._save_case_combined_results(pipeline, results, 'case3')
        
        # Create combined visualization for Case 3
        self._create_case_combined_visualization(pipeline, results, 'case3')
        
        return results
    
    def get_case_info(self, case_type: str) -> Dict[str, str]:
        """
        Get information about a specific case.
        
        Args:
            case_type: The case type ('case1', 'case2', 'case3')
            
        Returns:
            Dictionary with case information
        """
        case_info = {
            'case1': {
                'description': 'Hydrogenotrophic features for both ACE-km and H2-km predictions',
                'target': 'ACE-km and H2-km',
                'features': 'Hydrogenotrophic organisms only'
            },
            'case2': {
                'description': 'Acetoclastic features for both ACE-km and H2-km predictions',
                'target': 'ACE-km and H2-km', 
                'features': 'Acetoclastic organisms only'
            },
            'case3': {
                'description': 'All feature groups for both ACE-km and H2-km predictions',
                'target': 'ACE-km and H2-km',
                'features': 'Acetoclastic + Hydrogenotrophic + Syntrophic organisms'
            }
        }
        
        return case_info.get(case_type, {'description': 'Unknown case', 'target': 'Unknown', 'features': 'Unknown'})
    
    def _save_case_combined_results(self, pipeline, results: Dict[str, Any], case_type: str):
        """Save combined results for a case with both targets."""
        combined_results_path = os.path.join(pipeline.save_dir, f'{case_type}_combined_results.pkl')
        
        with open(combined_results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Create combined summary CSV
        summary_data = []
        
        for target_key, target_results in results.items():
            target_name = 'ACE-km' if 'ace' in target_key else 'H2-km'
            
            # Add results from each phase
            for phase in ['knn_training', 'explainer_training']:
                if phase in target_results:
                    phase_results = target_results[phase]
                    if 'fold_results' in phase_results:
                        # Calculate average metrics across folds
                        avg_metrics = self._calculate_average_metrics(phase_results['fold_results'])
                        
                        summary_data.append({
                            'case': case_type,
                            'target': target_name,
                            'phase': phase.replace('_training', ''),
                            'model_category': 'GNN',
                            'r2': avg_metrics.get('r2', 0),
                            'rmse': avg_metrics.get('rmse', 0),
                            'mae': avg_metrics.get('mae', 0),
                            'mse': avg_metrics.get('mse', 0),
                            'num_features': len(pipeline.dataset.node_feature_names)
                        })
            
            # Add ML results if available
            if 'ml_training' in target_results:
                ml_results = target_results['ml_training']
                if 'fold_results' in ml_results:
                    avg_metrics = self._calculate_average_metrics(ml_results['fold_results'])
                    
                    summary_data.append({
                        'case': case_type,
                        'target': target_name,
                        'phase': 'ml_embeddings',
                        'model_category': 'ML',
                        'r2': avg_metrics.get('r2', 0),
                        'rmse': avg_metrics.get('rmse', 0), 
                        'mae': avg_metrics.get('mae', 0),
                        'mse': avg_metrics.get('mse', 0),
                        'num_features': len(pipeline.dataset.node_feature_names)
                    })
        
        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(pipeline.save_dir, f'{case_type}_combined_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Combined results saved: {combined_results_path}")
            print(f"Combined summary saved: {summary_path}")
    
    def _create_case_combined_visualization(self, pipeline, results: Dict[str, Any], case_type: str):
        """Create combined visualization for a case with both targets."""
        from utils.visualization_utils import create_prediction_vs_actual_plots, generate_feature_importance_report
        
        # Create prediction vs actual plots for each target
        for target_key, target_results in results.items():
            target_name = 'ACE-km' if 'ace' in target_key else 'H2-km'
            
            # Collect predictions from different phases
            predictions_dict = {}
            
            for phase in ['knn_training', 'explainer_training']:
                if phase in target_results and 'fold_results' in target_results[phase]:
                    fold_results = target_results[phase]['fold_results']
                    
                    # Extract predictions from each fold
                    fold_predictions = []
                    for fold_result in fold_results:
                        if 'predictions' in fold_result:
                            fold_predictions.append({
                                'actual': fold_result['predictions']['actual'],
                                'predicted': fold_result['predictions']['predicted']
                            })
                    
                    if fold_predictions:
                        predictions_dict[phase.replace('_training', '')] = {
                            'fold_predictions': fold_predictions
                        }
            
            # ML predictions
            if 'ml_training' in target_results and 'fold_results' in target_results['ml_training']:
                ml_fold_results = target_results['ml_training']['fold_results']
                ml_fold_predictions = []
                for fold_result in ml_fold_results:
                    if 'predictions' in fold_result:
                        ml_fold_predictions.append({
                            'actual': fold_result['predictions']['actual'],
                            'predicted': fold_result['predictions']['predicted']
                        })
                
                if ml_fold_predictions:
                    predictions_dict['ml'] = {
                        'fold_predictions': ml_fold_predictions
                    }
            
            # Create plots
            if predictions_dict:
                plots_dir = os.path.join(pipeline.save_dir, f'{case_type}_{target_name}_plots')
                create_prediction_vs_actual_plots(predictions_dict, plots_dir, [target_name])
        
        # Generate feature importance reports if available
        if hasattr(pipeline.dataset, 'explainer_sparsified_graph_data'):
            explainer_data = pipeline.dataset.explainer_sparsified_graph_data
            if 'attention_scores' in explainer_data:
                importance_path = os.path.join(pipeline.save_dir, f'{case_type}_feature_importance.png')
                generate_feature_importance_report(
                    explainer_data['attention_scores'],
                    pipeline.dataset.node_feature_names,
                    importance_path,
                    top_n=20
                )
    
    def _calculate_average_metrics(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics across folds."""
        metrics = ['r2', 'rmse', 'mae', 'mse']
        avg_metrics = {}
        
        for metric in metrics:
            values = []
            for fold_result in fold_results:
                if 'metrics' in fold_result and metric in fold_result['metrics']:
                    values.append(fold_result['metrics'][metric])
                elif metric in fold_result:
                    values.append(fold_result[metric])
            
            if values:
                avg_metrics[metric] = np.mean(values)
            else:
                avg_metrics[metric] = 0.0
        
        return avg_metrics
