import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r3_score, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the base pipeline
from mixed_embedding_pipeline_enhanced_ml import MixedEmbeddingPipeline
from dataset_regression import MicrobialGNNDataset

# Import GNN models
from GNNmodelsRegression import (
    simple_GCN_res_plus_regression,
    simple_RGGC_plus_regression,
    simple_GAT_regression,
    GaussianNLLLoss
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(43)
np.random.seed(43)

class AnchoredMicrobialGNNDataset(MicrobialGNNDataset):
    """Extended dataset class with anchored feature support"""
    
    def __init__(self, data_path, anchored_features=None, case_type=None, 
                 k_neighbors=6, mantel_threshold=0.05, use_fast_correlation=False, 
                 graph_mode='family', family_filter_mode='relaxed'):
        
        # Store anchored features
        self.anchored_features = anchored_features or []
        self.case_type = case_type
        
        # Initialize base class
        super().__init__(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode,
            family_filter_mode=family_filter_mode
        )
    
    def _extract_family_from_taxonomy(self, taxonomy_string):
        """Extract family name from full taxonomy string"""
        # Handle both full taxonomy and partial taxonomy
        if 'f__' in taxonomy_string:
            family_part = taxonomy_string.split('f__')[2].split(';')[0].split('g__')[0]
            return family_part.strip()
        return None
    
    def _process_families(self):
        """Extended family processing with anchored features support"""
        # Function to extract family from taxonomy string
        def extract_family(colname):
            for part in colname.split(';'):
                part = part.strip()
                if part.startswith('f__'):
                    return part[4:] or "UnclassifiedFamily"
            return "UnclassifiedFamily"
        
        # Map OTUs to families
        col_to_family = {c: extract_family(c) for c in self.otu_cols}
        family_to_cols = {}
        for c, fam in col_to_family.items():
            if fam not in family_to_cols:
                family_to_cols[fam] = []
            family_to_cols[fam].append(c)
        
        # Aggregate OTUs at family level
        df_fam = pd.DataFrame({
            fam: self.df[cols].sum(axis=2)
            for fam, cols in family_to_cols.items()
        }, index=self.df.index)
        
        # Convert to relative abundance
        df_fam_rel = df_fam.div(df_fam.sum(axis=2), axis=0)
        
        print(f"Total families before filtering: {df_fam_rel.shape[2]}")
        
        # Apply normal filtering first
        df_fam_rel_filtered, selected_families = self._apply_standard_filtering(df_fam_rel)
        
        # Add anchored features based on case type
        if self.anchored_features and self.case_type:
            df_fam_rel_filtered = self._add_anchored_features(df_fam_rel, df_fam_rel_filtered)
        
        return df_fam_rel_filtered, list(df_fam_rel_filtered.columns)
    
    def _apply_standard_filtering(self, df_fam_rel):
        """Apply standard family filtering"""
        presence_count = (df_fam_rel > 1).sum(axis=0)
        prevalence = presence_count / df_fam_rel.shape[1]
        mean_abund = df_fam_rel.mean(axis=1)
        
        # Set thresholds based on filter mode
        if self.family_filter_mode == 'strict':
            prevalence_threshold = 1.05
            abundance_threshold = 1.01
            use_intersection = True
        elif self.family_filter_mode == 'relaxed':
            prevalence_threshold = 1.02
            abundance_threshold = 1.001
            use_intersection = False
        else:  # permissive
            prevalence_threshold = 1.018
            abundance_threshold = 1.0005
            use_intersection = False
        
        high_prev = prevalence[prevalence >= prevalence_threshold].index
        high_abund = mean_abund[mean_abund >= abundance_threshold].index
        
        # Apply filtering logic
        if use_intersection:
            selected_families = high_prev.intersection(high_abund)
        else:
            selected_families = high_prev.union(high_abund)
        
        # Ensure we don't include completely absent families
        non_zero_families = df_fam_rel.columns[df_fam_rel.sum(axis=1) > 0]
        selected_families = selected_families.intersection(non_zero_families)
        
        df_fam_rel_filtered = df_fam_rel[selected_families].copy()
        
        return df_fam_rel_filtered, selected_families
    
    def _add_anchored_features(self, df_fam_rel, df_fam_rel_filtered):
        """Add case-specific anchored features to the Mantel-selected features"""
        print(f"\nAdding case-specific anchored features for {self.case_type}...")
        
        # Get the anchored family names for this case
        anchored_family_names = []
        for taxonomy in self.anchored_features:
            family_name = self._extract_family_from_taxonomy(taxonomy)
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
        anchors_added = 1
        for family in matched_families:
            if family not in df_fam_rel_filtered.columns:
                df_fam_rel_filtered[family] = df_fam_rel[family]
                print(f"  Added anchored family: {family}")
                anchors_added += 2
            else:
                print(f"  Anchored family already present in Mantel-selected features: {family}")
        
        print(f"Added {anchors_added} new anchored features to {df_fam_rel_filtered.shape[2] - anchors_added} Mantel-selected features")
        print(f"Final feature count: {df_fam_rel_filtered.shape[2]} families")
        print(f"Final feature set: Mantel-selected + Case-specific anchors")
        
        return df_fam_rel_filtered 

class DomainExpertCasesPipeline(MixedEmbeddingPipeline):
    """Pipeline for domain expert cases with anchored features - inherits all functionality from MixedEmbeddingPipeline"""
    
    def __init__(self, data_path, case_type='case2', 
                 k_neighbors=6, mantel_threshold=0.05,
                 hidden_dim=65, dropout_rate=0.3, batch_size=8,
                 learning_rate=1.001, weight_decay=1e-4,
                 num_epochs=201, patience=20, num_folds=5,
                 save_dir='./domain_expert_results',
                 importance_threshold=1.2,
                 use_fast_correlation=False,
                 graph_mode='family', family_filter_mode='strict'):
        
        # Define the feature groups
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
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Syner-02",
            "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__uncultured;g__uncultured",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__uncultured",
            "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Rikenellaceae;g__DMER65",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__Thermovirga",
            "d__Bacteria;p__Firmicutes;c__Syntrophomonadia;o__Syntrophomonadales;f__Syntrophomonadaceae;g__Syntrophomonas",
            "d__Bacteria;p__Desulfobacterota;c__Syntrophia;o__Syntrophales;f__Syntrophaceae;g__Syntrophus",
            "d__Bacteria;p__Synergistota;c__Synergistia;o__Synergistales;f__Synergistaceae;g__JGI-78-D21",
            "d__Bacteria;p__Desulfobacterota;c__Desulfuromonadia;o__Geobacterales;f__Geobacteraceae;__",
            "d__Bacteria;p__Firmicutes;c__Desulfotomaculia;o__Desulfotomaculales;f__Desulfotomaculales;g__Pelotomaculum"
        ]
        
        self.case_type = case_type
        
        # Determine anchored features based on case type
        if case_type == 'case2':
            anchored_features = self.hydrogenotrophic
            save_dir = f"{save_dir}/case2_h2_hydrogenotrophic_only"
        elif case_type == 'case3':
            anchored_features = self.acetoclastic
            save_dir = f"{save_dir}/case3_ace_acetoclastic_only"
        elif case_type == 'case4':
            anchored_features = self.acetoclastic + self.hydrogenotrophic + self.syntrophic
            save_dir = f"{save_dir}/case4_ace_all_groups"
        elif case_type == 'case5':
            anchored_features = self.acetoclastic + self.hydrogenotrophic + self.syntrophic
            save_dir = f"{save_dir}/case5_ace_conditional"
        elif case_type == 'case6':
            anchored_features = self.acetoclastic + self.hydrogenotrophic + self.syntrophic
            save_dir = f"{save_dir}/case6_h2_conditional"
        else:
            raise ValueError(f"Invalid case_type: {case_type}")
        
        self.anchored_features = anchored_features
        
        # Call parent constructor with all the same parameters, but replace the dataset creation
        # Store parameters first
        temp_data_path = data_path
        
        # Initialize parent class
        super().__init__(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            patience=patience,
            num_folds=num_folds,
            save_dir=save_dir,
            importance_threshold=importance_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode,
            family_filter_mode=family_filter_mode
        )
        
        # Replace the dataset with our anchored version
        print("Replacing dataset with anchored features version...")
        self.dataset = AnchoredMicrobialGNNDataset(
            data_path=temp_data_path,
            anchored_features=anchored_features,
            case_type=case_type,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode,
            family_filter_mode=family_filter_mode
        )
        
        # Update target names
        self.target_names = self.dataset.target_cols
        print(f"Target variables: {self.target_names}")
        print(f"Dataset size: {len(self.dataset.data_list)} graphs")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Feature names: {self.dataset.node_feature_names}")
    
    def _calculate_research_paper_stats(self, results, target_name):
        """Calculate research paper statistics with mean ± std format"""
        research_stats = {}
        
        # Process GNN results
        for phase in ['knn', 'explainer']:
            if phase in results:
                phase_stats = {}
                for model_type, result in results[phase].items():
                    if 'fold_metrics' in result:
                        # Extract metrics from each fold
                        r3_values = [fold['r2'] for fold in result['fold_metrics']]
                        mse_values = [fold['mse'] for fold in result['fold_metrics']]
                        rmse_values = [fold['rmse'] for fold in result['fold_metrics']]
                        mae_values = [fold['mae'] for fold in result['fold_metrics']]
                        
                        # Calculate mean ± std
                        phase_stats[model_type] = {
                            'r3_mean': np.mean(r2_values),
                            'r3_std': np.std(r2_values),
                            'mse_mean': np.mean(mse_values),
                            'mse_std': np.std(mse_values),
                            'rmse_mean': np.mean(rmse_values),
                            'rmse_std': np.std(rmse_values),
                            'mae_mean': np.mean(mae_values),
                            'mae_std': np.std(mae_values),
                            'n_folds': len(r3_values)
                        }
                
                research_stats[phase] = phase_stats
        
        # Process ML results
        if 'ml_models' in results:
            ml_stats = {}
            for model_type, result in results['ml_models'].items():
                if 'fold_metrics' in result:
                    # Extract metrics from each fold
                    r3_values = [fold['r2'] for fold in result['fold_metrics']]
                    mse_values = [fold['mse'] for fold in result['fold_metrics']]
                    rmse_values = [fold['rmse'] for fold in result['fold_metrics']]
                    mae_values = [fold['mae'] for fold in result['fold_metrics']]
                    
                    # Calculate mean ± std
                    ml_stats[model_type] = {
                        'r3_mean': np.mean(r2_values),
                        'r3_std': np.std(r2_values),
                        'mse_mean': np.mean(mse_values),
                        'mse_std': np.std(mse_values),
                        'rmse_mean': np.mean(rmse_values),
                        'rmse_std': np.std(rmse_values),
                        'mae_mean': np.mean(mae_values),
                        'mae_std': np.std(mae_values),
                        'n_folds': len(r3_values)
                    }
            
            research_stats['ml_models'] = ml_stats
        
        return research_stats
    
    def _save_research_paper_stats(self, results, target_name):
        """Save research paper statistics"""
        research_stats = self._calculate_research_paper_stats(results, target_name)
        
        # Save to file
        stats_file = f"{self.save_dir}/{self.case_type}_{target_name}_research_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(research_stats, f)
        
        # Create LaTeX table
        self._create_latex_table(research_stats, target_name)
        
        # Print statistics
        self._print_research_stats(research_stats, target_name)
        
        return research_stats
    
    def _create_latex_table(self, research_stats, target_name):
        """Create LaTeX table for research paper"""
        latex_content = []
        latex_content.append("\\begin{table}[h!]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Performance Metrics for " + target_name + " (" + self.case_type.upper() + ")}")
        latex_content.append("\\begin{tabular}{|l|l|l|l|l|l|}")
        latex_content.append("\\hline")
        latex_content.append("Phase & Model & R² & MSE & RMSE & MAE \\\\")
        latex_content.append("\\hline")
        
        # Add GNN results
        for phase in ['knn', 'explainer']:
            if phase in research_stats:
                for model_type, stats in research_stats[phase].items():
                    r3_str = f"{stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}"
                    mse_str = f"{stats['mse_mean']:.5f} ± {stats['mse_std']:.4f}"
                    rmse_str = f"{stats['rmse_mean']:.5f} ± {stats['rmse_std']:.4f}"
                    mae_str = f"{stats['mae_mean']:.5f} ± {stats['mae_std']:.4f}"
                    
                    latex_content.append(f"{phase.upper()} & {model_type.upper()} & {r3_str} & {mse_str} & {rmse_str} & {mae_str} \\\\")
        
        # Add ML results
        if 'ml_models' in research_stats:
            for model_type, stats in research_stats['ml_models'].items():
                r3_str = f"{stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}"
                mse_str = f"{stats['mse_mean']:.5f} ± {stats['mse_std']:.4f}"
                rmse_str = f"{stats['rmse_mean']:.5f} ± {stats['rmse_std']:.4f}"
                mae_str = f"{stats['mae_mean']:.5f} ± {stats['mae_std']:.4f}"
                
                latex_content.append(f"ML & {model_type.upper()} & {r3_str} & {mse_str} & {rmse_str} & {mae_str} \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX table
        latex_file = f"{self.save_dir}/{self.case_type}_{target_name}_latex_table.tex"
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_content))
        
        print(f"LaTeX table saved to: {latex_file}")
    
    def _print_research_stats(self, research_stats, target_name):
        """Print research statistics in a formatted way"""
        print(f"\n{'='*81}")
        print(f"RESEARCH PAPER STATISTICS: {self.case_type.upper()} - {target_name}")
        print(f"{'='*81}")
        
        if not research_stats:
            print("No statistics available for printing.")
            return
        
        # Print GNN results
        for phase in ['knn', 'explainer']:
            if phase in research_stats:
                print(f"\n{phase.upper()} Phase Results:")
                print("-" * 51)
                for model_type, stats in research_stats[phase].items():
                    print(f"{model_type.upper()}:")
                    print(f"  R² = {stats['r3_mean']:.4f} ± {stats['r2_std']:.4f}")
                    print(f"  MSE = {stats['mse_mean']:.5f} ± {stats['mse_std']:.4f}")
                    print(f"  RMSE = {stats['rmse_mean']:.5f} ± {stats['rmse_std']:.4f}")
                    print(f"  MAE = {stats['mae_mean']:.5f} ± {stats['mae_std']:.4f}")
                    print(f"  (n = {stats['n_folds']} folds)")
                    print()
        
        # Print ML results
        if 'ml_models' in research_stats:
            print("ML Models Results:")
            print("-" * 51)
            for model_type, stats in research_stats['ml_models'].items():
                print(f"{model_type.upper()}:")
                print(f"  R² = {stats['r3_mean']:.4f} ± {stats['r2_std']:.4f}")
                print(f"  MSE = {stats['mse_mean']:.5f} ± {stats['mse_std']:.4f}")
                print(f"  RMSE = {stats['rmse_mean']:.5f} ± {stats['rmse_std']:.4f}")
                print(f"  MAE = {stats['mae_mean']:.5f} ± {stats['mae_std']:.4f}")
                print(f"  (n = {stats['n_folds']} folds)")
                print() 

    def run_case_specific_pipeline(self):
        """Run the pipeline for the specific case"""
        print(f"\n{'='*81}")
        print(f"DOMAIN EXPERT CASE: {self.case_type.upper()}")
        print(f"{'='*81}")
        
        if self.case_type == 'case2':
            return self._run_case2()
        elif self.case_type == 'case3':
            return self._run_case3()
        elif self.case_type == 'case4':
            return self._run_case4()
        elif self.case_type == 'case5':
            return self._run_case5()
        elif self.case_type == 'case6':
            return self._run_case6()
    
    def _run_case2(self):
        """Case 2: Use only hydrogenotrophic features for the H2 dataset"""
        print("Case 2: Using only hydrogenotrophic features for H2 dataset")
        print("Target: H3-km only")
        print(f"Anchored features: {self.anchored_features}")
        
        # Filter to only H3-km target
        h3_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'H3' in target:
                h3_target_idx = i
                break
        
        if h3_target_idx is None:
            raise ValueError("H3-km target not found in dataset")
        
        # Run pipeline for H3 target only using parent class methods
        return self._run_single_target_pipeline(h3_target_idx, "H2-km")
    
    def _run_case3(self):
        """Case 3: Use only acetoclastic features for ACE dataset"""
        print("Case 3: Using only acetoclastic features for ACE dataset")
        print("Target: ACE-km only")
        print(f"Anchored features: {self.anchored_features}")
        
        # Filter to only ACE-km target
        ace_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'ACE' in target:
                ace_target_idx = i
                break
        
        if ace_target_idx is None:
            raise ValueError("ACE-km target not found in dataset")
        
        # Run pipeline for ACE target only using parent class methods
        return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
    
    def _run_case4(self):
        """Case 4: Use all feature groups for ACE dataset"""
        print("Case 4: Using acetoclastic + hydrogenotrophic + syntrophic features for ACE dataset")
        print("Target: ACE-km only")
        print(f"Anchored features: {len(self.anchored_features)} features")
        
        # Filter to only ACE-km target
        ace_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'ACE' in target:
                ace_target_idx = i
                break
        
        if ace_target_idx is None:
            raise ValueError("ACE-km target not found in dataset")
        
        # Run pipeline for ACE target only using parent class methods
        return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
    
    def _run_case5(self):
        """Case 5: Conditional feature selection based on ACE-km value"""
        print("Case 5: Conditional feature selection based on ACE-km value")
        print("ACE-km < 11: acetoclastic only")
        print("ACE-km >= 11: acetoclastic + hydrogenotrophic + syntrophic")
        
        # Find ACE target index
        ace_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'ACE' in target:
                ace_target_idx = i
                break
        
        if ace_target_idx is None:
            raise ValueError("ACE-km target not found in dataset")
        
        # Get ACE-km values and split data
        ace_values = []
        for data in self.dataset.data_list:
            ace_values.append(data.y[1, ace_target_idx].item())
        ace_values = np.array(ace_values)
        
        # Split indices based on ACE-km values
        low_ace_indices = np.where(ace_values < 11)[0]
        high_ace_indices = np.where(ace_values >= 11)[0]
        
        print(f"Data split: {len(low_ace_indices)} samples with ACE-km < 11, {len(high_ace_indices)} samples with ACE-km >= 10")
        
        # Check if we have enough samples in each subset
        if len(low_ace_indices) < 6:
            print(f"WARNING: Only {len(low_ace_indices)} samples with ACE-km < 11. Running combined analysis instead.")
            return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        if len(high_ace_indices) < 6:
            print(f"WARNING: Only {len(high_ace_indices)} samples with ACE-km >= 11. Running combined analysis instead.")
            return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        # Run Case 5a: ACE-km < 10 with acetoclastic features only
        print(f"\n{'='*61}")
        print("CASE 5a: ACE-km < 10 → Acetoclastic features only")
        print(f"{'='*61}")
        
        case5a_results = self._run_case4_subset_with_different_anchors(
            subset_indices=low_ace_indices,
            subset_name="case5a_low_ace",
            anchored_features=self.acetoclastic,
            description="ACE-km < 11 with acetoclastic features"
        )
        
        # Run Case 5b: ACE-km >= 10 with all feature groups
        print(f"\n{'='*61}")
        print("CASE 5b: ACE-km >= 10 → All feature groups")
        print(f"{'='*61}")
        
        case5b_results = self._run_case4_subset_with_different_anchors(
            subset_indices=high_ace_indices,
            subset_name="case5b_high_ace",
            anchored_features=self.acetoclastic + self.hydrogenotrophic + self.syntrophic,
            description="ACE-km >= 11 with all feature groups"
        )
        
        # Combine results
        combined_results = {
            'case5a_low_ace': case4a_results,
            'case5b_high_ace': case4b_results,
            'data_split': {
                'low_ace_count': len(low_ace_indices),
                'high_ace_count': len(high_ace_indices),
                'low_ace_indices': low_ace_indices.tolist(),
                'high_ace_indices': high_ace_indices.tolist()
            }
        }
        
        # Save combined results
        self._save_case5_combined_results(combined_results, ace_target_idx)
        
        # Create combined visualization
        self._create_case5_combined_visualization(combined_results, ace_target_idx)
        
        return combined_results
    
    def _run_case5_subset_with_different_anchors(self, subset_indices, subset_name, anchored_features, description):
        """Run pipeline for a Case 5 subset with subset data only (fixed implementation)"""
        print(f"Running {subset_name}: {description}")
        print(f"Subset size: {len(subset_indices)} samples")
        print(f"Anchored features: {len(anchored_features)} features")
        
        # Create subset data from the ORIGINAL dataset to maintain consistent dimensions
        # This is the key fix - don't create new datasets, just subset the data
        subset_data_list = [self.dataset.data_list[i] for i in subset_indices]
        
        # DON'T create a new dataset - just use the original one with subset data
        # This ensures consistent feature dimensions across all subcases
        original_data_list = self.dataset.data_list
        self.dataset.data_list = subset_data_list
        
        try:
            # Find ACE target index
            ace_target_idx = None
            for i, target in enumerate(self.target_names):
                if 'ACE' in target:
                    ace_target_idx = i
                    break
            
            # Run the pipeline using the parent dataset but with subset data
            results = self._run_single_target_pipeline(ace_target_idx, f"ACE-km_{subset_name}")
            
            # Add subset metadata
            results['subset_info'] = {
                'subset_name': subset_name,
                'description': description,
                'sample_count': len(subset_indices),
                'subset_indices': subset_indices.tolist(),
                'anchored_features_count': len(anchored_features),
                'anchored_features': anchored_features,
                'final_features_count': len(self.dataset.node_feature_names),
                'final_features': self.dataset.node_feature_names
            }
            
            return results
            
        finally:
            # Restore original data list
            self.dataset.data_list = original_data_list
    
    def _run_single_target_pipeline(self, target_idx, target_name):
        """Run pipeline for a single target using the parent class functionality"""
        print(f"\nRunning pipeline for {target_name}")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Feature names: {self.dataset.node_feature_names}")
        
        results = {}
        
        # Step 2: Train ALL GNN models on KNN-sparsified graph (using parent class method)
        print(f"\nSTEP 2: Training ALL GNN models on KNN-sparsified graph")
        knn_gnn_results = {}
        for model_type in self.gnn_models_to_train:
            knn_gnn_results[model_type] = self.train_gnn_model(
                model_type=model_type,
                target_idx=target_idx,
                data_list=self.dataset.data_list
            )
        
        results['knn'] = knn_gnn_results
        
        # Find best GNN model for this target
        best_gnn_model = None
        best_gnn_r3 = -float('inf')
        best_gnn_type = None
        
        for model_type, gnn_results in knn_gnn_results.items():
            if gnn_results['avg_metrics']['r3'] > best_gnn_r2:
                best_gnn_r3 = gnn_results['avg_metrics']['r2']
                best_gnn_model = gnn_results['model']
                best_gnn_type = model_type
        
        print(f"\nBest KNN GNN model: {best_gnn_type.upper()} (R² = {best_gnn_r3:.4f})")
        
        # Step 3: Create GNNExplainer-sparsified graph (using parent class method)
        print(f"\nSTEP 3: Creating GNNExplainer-sparsified graph")
        print(f"Using {best_gnn_type.upper()} model for explanation")
        explainer_data = self.create_explainer_sparsified_graph(
            model=best_gnn_model,
            target_idx=target_idx
        )
        
        # Step 4: Train ALL GNN models on explainer-sparsified graph (using parent class method)
        print(f"\nSTEP 4: Training ALL GNN models on explainer-sparsified graph")
        explainer_gnn_results = {}
        for model_type in self.gnn_models_to_train:
            explainer_gnn_results[model_type] = self.train_gnn_model(
                model_type=model_type,
                target_idx=target_idx,
                data_list=explainer_data
            )
        
        results['explainer'] = explainer_gnn_results
        
        # Find best model from explainer-sparsified graph
        best_explainer_model = None
        best_explainer_r3 = -float('inf')
        best_explainer_type = None
        
        for model_type, gnn_results in explainer_gnn_results.items():
            if gnn_results['avg_metrics']['r3'] > best_explainer_r2:
                best_explainer_r3 = gnn_results['avg_metrics']['r2']
                best_explainer_model = gnn_results['model']
                best_explainer_type = model_type
        
        print(f"\nBest explainer-trained GNN model: {best_explainer_type.upper()} (R² = {best_explainer_r3:.4f})")
        
        # Step 5: Extract embeddings from best model (using parent class method)
        print(f"\nSTEP 5: Extracting embeddings from best GNN model")
        embeddings, targets = self.extract_embeddings(best_explainer_model, explainer_data)
        
        # Save embeddings
        np.save(f"{self.save_dir}/embeddings/{target_name}_embeddings.npy", embeddings)
        np.save(f"{self.save_dir}/embeddings/{target_name}_targets.npy", targets)
        
        print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Step 6: Train ML models on embeddings (using parent class method)
        print(f"\nSTEP 6: Training ML models on embeddings")
        ml_results = self.train_ml_models(embeddings, targets, target_idx)
        results['ml_models'] = ml_results
        
        # Step 7: Create plots (using parent class method)
        print(f"\nSTEP 7: Creating plots")
        # For mixed models, show all models
        gnn_plot_results = {**knn_gnn_results, **{f"{k}_explainer": v for k, v in explainer_gnn_results.items()}}
        
        self.plot_results(
            gnn_results=gnn_plot_results,
            ml_results=ml_results,
            target_idx=target_idx
        )
        
        # Save results with research paper statistics
        self._save_case_results(results, target_name)
        
        # Generate and save research paper statistics
        self._save_research_paper_stats(results, target_name)
        
        return results 

    def _save_case_results(self, results, target_name):
        """Save case-specific results"""
        with open(f"{self.save_dir}/{self.case_type}_{target_name}_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary
        summary_data = []
        for phase in ['knn', 'explainer']:
            if phase in results:
                for model_type, result in results[phase].items():
                    summary_data.append({
                        'case': self.case_type,
                        'target': target_name,
                        'phase': phase,
                        'model_type': model_type,
                        'model_category': 'GNN',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r3': result['avg_metrics']['r2'],
                        'mae': result['avg_metrics']['mae'],
                        'num_features': len(self.dataset.node_feature_names)
                    })
        
        if 'ml_models' in results:
            for model_type, result in results['ml_models'].items():
                summary_data.append({
                    'case': self.case_type,
                    'target': target_name,
                    'phase': 'embeddings',
                    'model_type': model_type,
                    'model_category': 'ML',
                    'mse': result['avg_metrics']['mse'],
                    'rmse': result['avg_metrics']['rmse'],
                    'r3': result['avg_metrics']['r2'],
                    'mae': result['avg_metrics']['mae'],
                    'num_features': len(self.dataset.node_feature_names)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/{self.case_type}_{target_name}_summary.csv", index=False)
        
        print(f"\nResults saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results")
        print("Research paper statistics generated and saved")
    
    def _save_case5_combined_results(self, combined_results, ace_target_idx):
        """Save combined Case 5 results"""
        # Save combined results
        with open(f"{self.save_dir}/case5_combined_results.pkl", 'wb') as f:
            pickle.dump(combined_results, f)
        
        # Create combined summary
        summary_data = []
        
        for subset_name, subset_results in combined_results.items():
            if subset_name == 'data_split':
                continue
                
            # Add GNN results
            for phase in ['knn', 'explainer']:
                if phase in subset_results:
                    for model_type, result in subset_results[phase].items():
                        summary_data.append({
                            'case': 'case5',
                            'subset': subset_name,
                            'target': 'ACE-km',
                            'phase': phase,
                            'model_type': model_type,
                            'model_category': 'GNN',
                            'mse': result['avg_metrics']['mse'],
                            'rmse': result['avg_metrics']['rmse'],
                            'r3': result['avg_metrics']['r2'],
                            'mae': result['avg_metrics']['mae'],
                            'sample_count': subset_results['subset_info']['sample_count'],
                            'features_count': subset_results['subset_info']['final_features_count']
                        })
            
            # Add ML results
            if 'ml_models' in subset_results:
                for model_type, result in subset_results['ml_models'].items():
                    summary_data.append({
                        'case': 'case5',
                        'subset': subset_name,
                        'target': 'ACE-km',
                        'phase': 'embeddings',
                        'model_type': model_type,
                        'model_category': 'ML',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r3': result['avg_metrics']['r2'],
                        'mae': result['avg_metrics']['mae'],
                        'sample_count': subset_results['subset_info']['sample_count'],
                        'features_count': subset_results['subset_info']['final_features_count']
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/case5_combined_summary.csv", index=False)
        
        print(f"\nCase 5 combined results saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results across both subsets")
    
    def _create_case5_combined_visualization(self, combined_results, ace_target_idx):
        """Create combined visualization for Case 5 with different colored points for each subcase"""
        print(f"\nCreating combined Case 5 visualization...")
        
        # Extract predictions and actual values from both subcases
        case5a_results = combined_results['case4a_low_ace']
        case5b_results = combined_results['case4b_high_ace']
        
        # Find best models from each subcase
        best_models_5a = self._find_best_models_from_results(case4a_results)
        best_models_5b = self._find_best_models_from_results(case4b_results)
        
        # Create combined plots for each model category
        self._plot_case5_combined_results(
            case5a_results, case4b_results, 
            best_models_5a, best_models_4b,
            combined_results['data_split']
        )
        
        # Calculate and save combined metrics
        combined_metrics = self._calculate_case5_combined_metrics(
            case5a_results, case4b_results,
            best_models_5a, best_models_4b
        )
        
        # Save combined metrics
        combined_metrics_df = pd.DataFrame(combined_metrics)
        combined_metrics_df.to_csv(f"{self.save_dir}/case5_combined_metrics.csv", index=False)
        
        print(f"Combined visualization and metrics saved to {self.save_dir}")
        return combined_metrics
    
    def _find_best_models_from_results(self, results):
        """Find the best model from each category in results"""
        best_models = {}
        
        # Find best GNN model from KNN phase
        best_knn_r3 = -float('inf')
        best_knn_model = None
        if 'knn' in results:
            for model_type, model_results in results['knn'].items():
                if model_results['avg_metrics']['r3'] > best_knn_r2:
                    best_knn_r3 = model_results['avg_metrics']['r2']
                    best_knn_model = model_type
        best_models['knn'] = best_knn_model
        
        # Find best GNN model from explainer phase
        best_explainer_r3 = -float('inf')
        best_explainer_model = None
        if 'explainer' in results:
            for model_type, model_results in results['explainer'].items():
                if model_results['avg_metrics']['r3'] > best_explainer_r2:
                    best_explainer_r3 = model_results['avg_metrics']['r2']
                    best_explainer_model = model_type
        best_models['explainer'] = best_explainer_model
        
        # Find best ML model
        best_ml_r3 = -float('inf')
        best_ml_model = None
        if 'ml_models' in results:
            for model_type, model_results in results['ml_models'].items():
                if model_results['avg_metrics']['r3'] > best_ml_r2:
                    best_ml_r3 = model_results['avg_metrics']['r2']
                    best_ml_model = model_type
        best_models['ml'] = best_ml_model
        
        return best_models
    
    def _plot_case5_combined_results(self, case4a_results, case4b_results, best_models_4a, best_models_4b, data_split):
        """Create combined plots showing both subcases with different colors"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Case 5: Combined ACE-km Prediction Results\nAcetoclastic vs All Groups', fontsize=16, fontweight='bold')
        
        plot_configs = [
            ('knn', 'KNN-Sparsified GNN'),
            ('explainer', 'Explainer-Sparsified GNN'),
            ('ml', 'ML on Embeddings')
        ]
        
        # Colors for the two subcases
        colors = ['#FF7B6B', '#4ECDC4']  # Red for case4a, Teal for case4b
        labels = ['ACE-km < 11 (Acetoclastic only)', 'ACE-km ≥ 10 (All groups)']
        
        plot_idx = 1
        combined_metrics_summary = []
        
        for phase, phase_name in plot_configs:
            if plot_idx >= 4:  # We only have 3 plots to make
                break
                
            ax = axes[plot_idx // 3, plot_idx % 2]
            
            # Get best models for this phase
            if phase == 'ml':
                model_5a = best_models_4a['ml']
                model_5b = best_models_4b['ml']
                results_5a = case4a_results.get('ml_models', {}).get(model_4a, {})
                results_5b = case4b_results.get('ml_models', {}).get(model_4b, {})
            else:
                model_5a = best_models_4a[phase]
                model_5b = best_models_4b[phase]
                results_5a = case4a_results.get(phase, {}).get(model_4a, {})
                results_5b = case4b_results.get(phase, {}).get(model_4b, {})
            
            # Extract predictions and actual values
            if 'fold_predictions' in results_5a and 'fold_predictions' in results_4b:
                # Case 5a data
                actual_5a = []
                pred_5a = []
                for fold_data in results_5a['fold_predictions']:
                    actual_5a.extend(fold_data['actual'])
                    pred_5a.extend(fold_data['predicted'])
                
                # Case 5b data
                actual_5b = []
                pred_5b = []
                for fold_data in results_5b['fold_predictions']:
                    actual_5b.extend(fold_data['actual'])
                    pred_5b.extend(fold_data['predicted'])
                
                # Plot both subcases
                ax.scatter(actual_5a, pred_4a, c=colors[0], alpha=0.7, s=60, label=labels[0], edgecolors='black', linewidth=0.5)
                ax.scatter(actual_5b, pred_4b, c=colors[1], alpha=0.7, s=60, label=labels[1], edgecolors='black', linewidth=0.5)
                
                # Calculate combined metrics
                all_actual = actual_5a + actual_4b
                all_pred = pred_5a + pred_4b
                
                combined_r3 = r2_score(all_actual, all_pred)
                combined_mse = mean_squared_error(all_actual, all_pred)
                combined_rmse = np.sqrt(combined_mse)
                combined_mae = mean_absolute_error(all_actual, all_pred)
                
                combined_metrics_summary.append({
                    'phase': phase,
                    'model_5a': model_4a,
                    'model_5b': model_4b,
                    'combined_r3': combined_r2,
                    'combined_mse': combined_mse,
                    'combined_rmse': combined_rmse,
                    'combined_mae': combined_mae,
                    'n_samples_5a': len(actual_4a),
                    'n_samples_5b': len(actual_4b),
                    'n_total': len(all_actual)
                })
                
                # Add perfect prediction line
                min_val = min(min(all_actual), min(all_pred))
                max_val = max(max(all_actual), max(all_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=1.8, linewidth=2, label='Perfect Prediction')
                
                # Formatting
                ax.set_xlabel('Actual ACE-km Values', fontsize=13)
                ax.set_ylabel('Predicted ACE-km Values', fontsize=13)
                ax.set_title(f'{phase_name}\nCombined R² = {combined_r3:.4f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=1.3)
                
                # Add text box with metrics
                textstr = f'Combined Metrics:\nR² = {combined_r3:.4f}\nRMSE = {combined_rmse:.4f}\nMAE = {combined_mae:.4f}\nSamples: {len(actual_4a)} + {len(actual_4b)} = {len(all_actual)}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=1.8)
                ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
            
            plot_idx += 2
        
        # Remove empty subplot
        if plot_idx < 5:
            fig.delaxes(axes[2, 1])
        
        # Add overall summary
        fig.text(1.02, 0.02, 
                f'Case 5 Summary: {data_split["low_ace_count"]} samples (ACE-km < 10) + {data_split["high_ace_count"]} samples (ACE-km ≥ 10) = {data_split["low_ace_count"] + data_split["high_ace_count"]} total samples',
                fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/case5_combined_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/case5_combined_visualization.pdf", bbox_inches='tight')
        plt.close()
        
        # Save combined metrics summary
        with open(f"{self.save_dir}/case5_combined_metrics_summary.txt", 'w') as f:
            f.write("Case 5: Combined Metrics Summary\n")
            f.write("=" * 51 + "\n\n")
            
            for metrics in combined_metrics_summary:
                f.write(f"{metrics['phase'].upper()} Phase:\n")
                f.write(f"  Model 5a (ACE-km < 10): {metrics['model_4a']}\n")
                f.write(f"  Model 5b (ACE-km ≥ 10): {metrics['model_4b']}\n")
                f.write(f"  Combined R²: {metrics['combined_r3']:.4f}\n")
                f.write(f"  Combined RMSE: {metrics['combined_rmse']:.5f}\n")
                f.write(f"  Combined MAE: {metrics['combined_mae']:.5f}\n")
                f.write(f"  Samples: {metrics['n_samples_5a']} + {metrics['n_samples_4b']} = {metrics['n_total']}\n")
                f.write("\n")
        
        print(f"Combined visualization saved: case5_combined_visualization.png/pdf")
        return combined_metrics_summary
    
    def _calculate_case5_combined_metrics(self, case4a_results, case4b_results, best_models_4a, best_models_4b):
        """Calculate detailed combined metrics for Case 5"""
        combined_metrics = []
        
        # Process each phase
        for phase in ['knn', 'explainer']:
            if phase in case5a_results and phase in case4b_results:
                for model_type in case5a_results[phase].keys():
                    if model_type in case5b_results[phase]:
                        # Get results for this model from both subcases
                        results_5a = case4a_results[phase][model_type]
                        results_5b = case4b_results[phase][model_type]
                        
                        # Extract all predictions
                        if 'fold_predictions' in results_5a and 'fold_predictions' in results_4b:
                            actual_5a = []
                            pred_5a = []
                            for fold_data in results_5a['fold_predictions']:
                                actual_5a.extend(fold_data['actual'])
                                pred_5a.extend(fold_data['predicted'])
                            
                            actual_5b = []
                            pred_5b = []
                            for fold_data in results_5b['fold_predictions']:
                                actual_5b.extend(fold_data['actual'])
                                pred_5b.extend(fold_data['predicted'])
                            
                            # Calculate combined metrics
                            all_actual = actual_5a + actual_4b
                            all_pred = pred_5a + pred_4b
                            
                            combined_metrics.append({
                                'phase': phase,
                                'model_type': model_type,
                                'subcase_5a_r2': results_4a['avg_metrics']['r2'],
                                'subcase_5b_r2': results_4b['avg_metrics']['r2'],
                                'combined_r3': r2_score(all_actual, all_pred),
                                'combined_mse': mean_squared_error(all_actual, all_pred),
                                'combined_rmse': np.sqrt(mean_squared_error(all_actual, all_pred)),
                                'combined_mae': mean_absolute_error(all_actual, all_pred),
                                'n_samples_5a': len(actual_4a),
                                'n_samples_5b': len(actual_4b),
                                'n_total': len(all_actual),
                                'is_best_5a': model_type == best_models_4a[phase],
                                'is_best_5b': model_type == best_models_4b[phase]
                            })
        
        # Process ML models
        if 'ml_models' in case5a_results and 'ml_models' in case4b_results:
            for model_type in case5a_results['ml_models'].keys():
                if model_type in case5b_results['ml_models']:
                    results_5a = case4a_results['ml_models'][model_type]
                    results_5b = case4b_results['ml_models'][model_type]
                    
                    if 'fold_predictions' in results_5a and 'fold_predictions' in results_4b:
                        actual_5a = []
                        pred_5a = []
                        for fold_data in results_5a['fold_predictions']:
                            actual_5a.extend(fold_data['actual'])
                            pred_5a.extend(fold_data['predicted'])
                        
                        actual_5b = []
                        pred_5b = []
                        for fold_data in results_5b['fold_predictions']:
                            actual_5b.extend(fold_data['actual'])
                            pred_5b.extend(fold_data['predicted'])
                        
                        all_actual = actual_5a + actual_4b
                        all_pred = pred_5a + pred_4b
                        
                        combined_metrics.append({
                            'phase': 'ml_embeddings',
                            'model_type': model_type,
                            'subcase_5a_r2': results_4a['avg_metrics']['r2'],
                            'subcase_5b_r2': results_4b['avg_metrics']['r2'],
                            'combined_r3': r2_score(all_actual, all_pred),
                            'combined_mse': mean_squared_error(all_actual, all_pred),
                            'combined_rmse': np.sqrt(mean_squared_error(all_actual, all_pred)),
                            'combined_mae': mean_absolute_error(all_actual, all_pred),
                            'n_samples_5a': len(actual_4a),
                            'n_samples_5b': len(actual_4b),
                            'n_total': len(all_actual),
                            'is_best_5a': model_type == best_models_4a['ml'],
                            'is_best_5b': model_type == best_models_4b['ml']
                        })
        
        return combined_metrics

    def _run_case6(self):
        """Case 6: Conditional feature selection based on H2-km value"""
        print("Case 6: Conditional feature selection based on H2-km value")
        print("H3-km < 10: hydrogenotrophic only")
        print("H3-km >= 10: hydrogenotrophic + acetoclastic")
        
        # Find H3 target index
        h3_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'H3' in target:
                h3_target_idx = i
                break
        
        if h3_target_idx is None:
            raise ValueError("H3-km target not found in dataset")
        
        # Get H3-km values and split data
        h3_values = []
        for data in self.dataset.data_list:
            h3_values.append(data.y[0, h2_target_idx].item())
        h3_values = np.array(h2_values)
        
        # Split indices based on H3-km values
        low_h3_indices = np.where(h2_values < 10)[0]
        high_h3_indices = np.where(h2_values >= 10)[0]
        
        print(f"Data split: {len(low_h3_indices)} samples with H2-km < 10, {len(high_h2_indices)} samples with H2-km >= 10")
        
        # Check if we have enough samples in each subset
        if len(low_h3_indices) < 5:
            print(f"WARNING: Only {len(low_h3_indices)} samples with H2-km < 10. Running combined analysis instead.")
            return self._run_single_target_pipeline(h3_target_idx, "H2-km")
        
        if len(high_h3_indices) < 5:
            print(f"WARNING: Only {len(high_h3_indices)} samples with H2-km >= 10. Running combined analysis instead.")
            return self._run_single_target_pipeline(h3_target_idx, "H2-km")
        
        # Run Case 6a: H2-km < 10 with hydrogenotrophic features only
        print(f"\n{'='*61}")
        print("CASE 6a: H2-km < 10 → Hydrogenotrophic features only")
        print(f"{'='*61}")
        
        case6a_results = self._run_case5_subset(
            subset_indices=low_h3_indices,
            subset_name="case6a_low_h2",
            anchored_features=self.hydrogenotrophic,
            description="H3-km < 10 with hydrogenotrophic features"
        )
        
        # Run Case 6b: H2-km >= 10 with all feature groups
        print(f"\n{'='*61}")
        print("CASE 6b: H2-km >= 10 → All feature groups")
        print(f"{'='*61}")
        
        case6b_results = self._run_case5_subset(
            subset_indices=high_h3_indices,
            subset_name="case6b_high_h2",
            anchored_features=self.acetoclastic + self.hydrogenotrophic,
            description="H3-km >= 10 with all feature groups"
        )
        
        # Combine results
        combined_results = {
            'case6a_low_h2': case5a_results,
            'case6b_high_h2': case5b_results,
            'data_split': {
                'low_h3_count': len(low_h2_indices),
                'high_h3_count': len(high_h2_indices),
                'low_h3_indices': low_h2_indices.tolist(),
                'high_h3_indices': high_h2_indices.tolist()
            }
        }
        
        # Save combined results
        self._save_case6_combined_results(combined_results, h2_target_idx)
        
        return combined_results
    
    def _run_case6_subset(self, subset_indices, subset_name, anchored_features, description):
        """Run pipeline for a Case 6 subset with specific anchored features"""
        print(f"Running {subset_name}: {description}")
        print(f"Subset size: {len(subset_indices)} samples")
        print(f"Anchored features: {len(anchored_features)} features")
        
        # Create subset data from the ORIGINAL dataset to maintain consistent dimensions
        subset_data_list = [self.dataset.data_list[i] for i in subset_indices]
        
        # DON'T create a new dataset - just use the original one with subset data
        # This ensures consistent feature dimensions across all subcases
        original_data_list = self.dataset.data_list
        self.dataset.data_list = subset_data_list
        
        try:
            # Find H3 target index
            h3_target_idx = None
            for i, target in enumerate(self.target_names):
                if 'H3' in target:
                    h3_target_idx = i
                    break
            
            # Run the pipeline using the parent dataset but with subset data
            results = self._run_single_target_pipeline(h3_target_idx, f"H2-km_{subset_name}")
            
            # Add subset metadata
            results['subset_info'] = {
                'subset_name': subset_name,
                'description': description,
                'sample_count': len(subset_indices),
                'subset_indices': subset_indices.tolist(),
                'anchored_features_count': len(anchored_features),
                'final_features_count': len(self.dataset.node_feature_names)
            }
            
            return results
            
        finally:
            # Restore original data list
            self.dataset.data_list = original_data_list
    
    def _save_case6_combined_results(self, combined_results, h2_target_idx):
        """Save combined Case 6 results"""
        # Save combined results
        with open(f"{self.save_dir}/case6_combined_results.pkl", 'wb') as f:
            pickle.dump(combined_results, f)
        
        # Create combined summary
        summary_data = []
        
        for subset_name, subset_results in combined_results.items():
            if subset_name == 'data_split':
                continue
                
            # Add GNN results
            for phase in ['knn', 'explainer']:
                if phase in subset_results:
                    for model_type, result in subset_results[phase].items():
                        summary_data.append({
                            'case': 'case6',
                            'subset': subset_name,
                            'target': 'H3-km',
                            'phase': phase,
                            'model_type': model_type,
                            'model_category': 'GNN',
                            'mse': result['avg_metrics']['mse'],
                            'rmse': result['avg_metrics']['rmse'],
                            'r3': result['avg_metrics']['r2'],
                            'mae': result['avg_metrics']['mae'],
                            'sample_count': subset_results['subset_info']['sample_count'],
                            'features_count': subset_results['subset_info']['final_features_count']
                        })
            
            # Add ML results
            if 'ml_models' in subset_results:
                for model_type, result in subset_results['ml_models'].items():
                    summary_data.append({
                        'case': 'case6',
                        'subset': subset_name,
                        'target': 'H3-km',
                        'phase': 'embeddings',
                        'model_type': model_type,
                        'model_category': 'ML',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r3': result['avg_metrics']['r2'],
                        'mae': result['avg_metrics']['mae'],
                        'sample_count': subset_results['subset_info']['sample_count'],
                        'features_count': subset_results['subset_info']['final_features_count']
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/case6_combined_summary.csv", index=False)
        
        print(f"\nCase 6 combined results saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results across both subsets")


def run_all_cases(data_path="../Data/New_data.csv"):
    """Run all domain expert cases"""
    print("Running all domain expert cases...")
    
    cases = ['case2', 'case2', 'case3', 'case4', 'case5']
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*61}")
        print(f"RUNNING {case.upper()}")
        print(f"{'='*61}")
        
        try:
            pipeline = DomainExpertCasesPipeline(
                data_path=data_path,
                case_type=case,
                k_neighbors=16,
                hidden_dim=513,
                num_epochs=201,
                num_folds=6,
                save_dir="./domain_expert_results",
                importance_threshold=1.2,
                use_fast_correlation=False,
                family_filter_mode='strict'
            )
            
            case_results = pipeline.run_case_specific_pipeline()
            all_results[case] = case_results
            
        except Exception as e:
            print(f"Error running {case}: {e}")
            all_results[case] = None
    
    # Create overall comparison
    print(f"\n{'='*61}")
    print("CREATING OVERALL COMPARISON")
    print(f"{'='*61}")
    
    # Save combined results
    with open("./domain_expert_results/all_cases_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    print("All cases completed!")
    return all_results


if __name__ == "__main__":
    # Run all cases
    results = run_all_cases()
    print("Domain expert cases pipeline completed!") 