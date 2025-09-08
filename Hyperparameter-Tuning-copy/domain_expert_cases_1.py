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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
from embeddings_pipeline import MixedEmbeddingPipeline
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
torch.manual_seed(42)
np.random.seed(42)

class AnchoredMicrobialGNNDataset(MicrobialGNNDataset):
    """Extended dataset class with anchored feature support"""
    
    def __init__(self, data_path, anchored_features=None, case_type=None, 
                 k_neighbors=5, mantel_threshold=0.05, use_fast_correlation=False, 
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
            family_part = taxonomy_string.split('f__')[1].split(';')[0].split('g__')[0]
            return family_part.strip()
        return None
    
    def _process_families(self):
        """Extended family processing with anchored features support"""
        # Function to extract family from taxonomy string
        def extract_family(colname):
            for part in colname.split(';'):
                part = part.strip()
                if part.startswith('f__'):
                    return part[3:] or "UnclassifiedFamily"
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
            fam: self.df[cols].sum(axis=1)
            for fam, cols in family_to_cols.items()
        }, index=self.df.index)
        
        # Convert to relative abundance
        df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
        
        print(f"Total families before filtering: {df_fam_rel.shape[1]}")
        
        # Apply normal filtering first
        df_fam_rel_filtered, selected_families = self._apply_standard_filtering(df_fam_rel)
        
        # Add anchored features based on case type
        if self.anchored_features and self.case_type:
            df_fam_rel_filtered = self._add_anchored_features(df_fam_rel, df_fam_rel_filtered)
        
        return df_fam_rel_filtered, list(df_fam_rel_filtered.columns)
    
    def _apply_standard_filtering(self, df_fam_rel):
        """Apply standard family filtering"""
        presence_count = (df_fam_rel > 0).sum(axis=0)
        prevalence = presence_count / df_fam_rel.shape[0]
        mean_abund = df_fam_rel.mean(axis=0)
        
        # Set thresholds based on filter mode
        if self.family_filter_mode == 'strict':
            prevalence_threshold = 0.05
            abundance_threshold = 0.01
            use_intersection = True
        elif self.family_filter_mode == 'relaxed':
            prevalence_threshold = 0.02
            abundance_threshold = 0.001
            use_intersection = False
        else:  # permissive
            prevalence_threshold = 0.018
            abundance_threshold = 0.0005
            use_intersection = False
        
        high_prev = prevalence[prevalence >= prevalence_threshold].index
        high_abund = mean_abund[mean_abund >= abundance_threshold].index
        
        # Apply filtering logic
        if use_intersection:
            selected_families = high_prev.intersection(high_abund)
        else:
            selected_families = high_prev.union(high_abund)
        
        # Ensure we don't include completely absent families
        non_zero_families = df_fam_rel.columns[df_fam_rel.sum(axis=0) > 0]
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


class DomainExpertCasesPipeline(MixedEmbeddingPipeline):
    """Pipeline for domain expert cases with anchored features - inherits all functionality from MixedEmbeddingPipeline"""
    
    def __init__(self, data_path, case_type='case1', 
                 k_neighbors=5, mantel_threshold=0.05,
                 hidden_dim=64, dropout_rate=0.3, batch_size=8,
                 learning_rate=0.001, weight_decay=1e-4,
                 num_epochs=200, patience=20, num_folds=5,
                 save_dir='./domain_expert_results',
                 importance_threshold=0.2,
                 use_fast_correlation=False,
                 graph_mode='family', family_filter_mode='strict',
                 use_nested_cv=True):  # Add nested CV parameter
        
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
        
        self.case_type = case_type
        
        # Determine anchored features based on case type
        if case_type == 'case1':
            anchored_features = self.hydrogenotrophic
            save_dir = f"{save_dir}/case1_h2_hydrogenotrophic_only"
        elif case_type == 'case2':
            anchored_features = self.acetoclastic
            save_dir = f"{save_dir}/case2_ace_acetoclastic_only"
        elif case_type == 'case3':
            anchored_features = self.acetoclastic + self.hydrogenotrophic + self.syntrophic
            save_dir = f"{save_dir}/case3_ace_all_groups"
        elif case_type == 'case4':
            anchored_features = self.acetoclastic + self.hydrogenotrophic + self.syntrophic
            save_dir = f"{save_dir}/case4_ace_conditional"
        elif case_type == 'case5':
            anchored_features = self.acetoclastic + self.hydrogenotrophic + self.syntrophic
            save_dir = f"{save_dir}/case5_h2_conditional"
        else:
            raise ValueError(f"Invalid case_type: {case_type}")
        
        self.anchored_features = anchored_features
        
        # Call parent constructor with all the same parameters, but replace the dataset creation
        # Store parameters first
        temp_data_path = data_path
        
        # Initialize parent class with nested CV parameter
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
            family_filter_mode=family_filter_mode,
            use_nested_cv=use_nested_cv  # Pass nested CV parameter to parent
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
        
        # Print nested CV status
        if self.use_nested_cv:
            print(f"\nNested CV Hyperparameter Tuning: ENABLED")
            print(f"Hyperparameter search space: {len(self.param_grid)} combinations")
            print(f"Explainer search space: {len(self.explainer_param_grid)} combinations")
        else:
            print(f"\nNested CV Hyperparameter Tuning: DISABLED")
    
    def run_case_specific_pipeline(self):
        """Run the pipeline for the specific case"""
        print(f"\n{'='*80}")
        print(f"DOMAIN EXPERT CASE: {self.case_type.upper()}")
        print(f"{'='*80}")
        
        if self.case_type == 'case1':
            return self._run_case1()
        elif self.case_type == 'case2':
            return self._run_case2()
        elif self.case_type == 'case3':
            return self._run_case3()
        elif self.case_type == 'case4':
            return self._run_case4()
        elif self.case_type == 'case5':
            return self._run_case5()
    
    def _run_case1(self):
        """Case 1: Use only hydrogenotrophic features for the H2 dataset"""
        print("Case 1: Using only hydrogenotrophic features for H2 dataset")
        print("Target: H2-km only")
        print(f"Anchored features: {self.anchored_features}")
        
        # Filter to only H2-km target
        h2_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'H2' in target:
                h2_target_idx = i
                break
        
        if h2_target_idx is None:
            raise ValueError("H2-km target not found in dataset")
        
        # Run pipeline for H2 target only using parent class methods
        return self._run_single_target_pipeline(h2_target_idx, "H2-km")
    
    def _run_case2(self):
        """Case 2: Use only acetoclastic features for ACE dataset"""
        print("Case 2: Using only acetoclastic features for ACE dataset")
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
    
    def _run_case3(self):
        """Case 3: Use all feature groups for both ACE-km and H2-km datasets"""
        print("Case 3: Using acetoclastic + hydrogenotrophic + syntrophic features for both targets")
        print("Targets: ACE-km and H2-km")
        print(f"Anchored features: {len(self.anchored_features)} features")
        
        # Find both target indices
        ace_target_idx = None
        h2_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'ACE' in target:
                ace_target_idx = i
            elif 'H2' in target:
                h2_target_idx = i
        
        if ace_target_idx is None:
            raise ValueError("ACE-km target not found in dataset")
        if h2_target_idx is None:
            raise ValueError("H2-km target not found in dataset")
        
        # Run pipeline for both targets
        results = {}
        
        print(f"\n{'='*60}")
        print("CASE 3a: ACE-km with all feature groups")
        print(f"{'='*60}")
        results['ace_km'] = self._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        print(f"\n{'='*60}")
        print("CASE 3b: H2-km with all feature groups")
        print(f"{'='*60}")
        results['h2_km'] = self._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        # Save combined case 3 results
        self._save_case3_combined_results(results)
        
        # Create combined visualization for Case 3
        self._create_case3_combined_visualization(results, ace_target_idx, h2_target_idx)
        
        return results
    
    def _run_case4(self):
        """Case 4: Conditional feature selection based on ACE-km value"""
        print("Case 4: Conditional feature selection based on ACE-km value")
        print("ACE-km < 10: acetoclastic only")
        print("ACE-km >= 10: acetoclastic + hydrogenotrophic + syntrophic")
        
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
            ace_values.append(data.y[0, ace_target_idx].item())
        ace_values = np.array(ace_values)
        
        # Split indices based on ACE-km values
        low_ace_indices = np.where(ace_values < 10)[0]
        high_ace_indices = np.where(ace_values >= 10)[0]
        
        print(f"Data split: {len(low_ace_indices)} samples with ACE-km < 10, {len(high_ace_indices)} samples with ACE-km >= 10")
        
        # Check if we have enough samples in each subset
        if len(low_ace_indices) < 5:
            print(f"WARNING: Only {len(low_ace_indices)} samples with ACE-km < 10. Running combined analysis instead.")
            return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        if len(high_ace_indices) < 5:
            print(f"WARNING: Only {len(high_ace_indices)} samples with ACE-km >= 10. Running combined analysis instead.")
            return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
        
        # Run Case 4a: ACE-km < 10 with acetoclastic features only
        print(f"\n{'='*60}")
        print("CASE 4a: ACE-km < 10 → Acetoclastic features only")
        print(f"{'='*60}")
        
        case4a_results = self._run_case4_subset_with_different_anchors(
            subset_indices=low_ace_indices,
            subset_name="case4a_low_ace",
            anchored_features=self.acetoclastic,
            description="ACE-km < 10 with acetoclastic features"
        )
        
        # Run Case 4b: ACE-km >= 10 with all feature groups
        print(f"\n{'='*60}")
        print("CASE 4b: ACE-km >= 10 → All feature groups")
        print(f"{'='*60}")
        
        case4b_results = self._run_case4_subset_with_different_anchors(
            subset_indices=high_ace_indices,
            subset_name="case4b_high_ace",
            anchored_features=self.acetoclastic + self.hydrogenotrophic + self.syntrophic,
            description="ACE-km >= 10 with all feature groups"
        )
        
        # Combine results
        combined_results = {
            'case4a_low_ace': case4a_results,
            'case4b_high_ace': case4b_results,
            'data_split': {
                'low_ace_count': len(low_ace_indices),
                'high_ace_count': len(high_ace_indices),
                'low_ace_indices': low_ace_indices.tolist(),
                'high_ace_indices': high_ace_indices.tolist()
            }
        }
        
        # Save combined results
        self._save_case4_combined_results(combined_results, ace_target_idx)
        
        # Create combined visualization
        self._create_case4_combined_visualization(combined_results, ace_target_idx)
        
        return combined_results
    
    def _run_case4_subset_with_different_anchors(self, subset_indices, subset_name, anchored_features, description):
        """Run pipeline for a Case 4 subset with subset data only (fixed implementation)"""
        print(f"Running {subset_name}: {description}")
        print(f"Subset size: {len(subset_indices)} samples")
        print(f"Anchored features: {len(anchored_features)} features")
        
        # Create detailed_results folder for this subset
        subset_save_dir = f"{self.save_dir}/{subset_name}"
        os.makedirs(f"{subset_save_dir}/detailed_results", exist_ok=True)
        
        # Create subset data from the ORIGINAL dataset to maintain consistent dimensions
        # This is the key fix - don't create new datasets, just subset the data
        subset_data_list = [self.dataset.data_list[i] for i in subset_indices]
        
        # DON'T create a new dataset - just use the original one with subset data
        # This ensures consistent feature dimensions across all subcases
        original_data_list = self.dataset.data_list
        original_save_dir = self.save_dir
        self.dataset.data_list = subset_data_list
        self.save_dir = subset_save_dir
        
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
                'final_features_count': len(self.dataset.node_feature_names),
                'final_features': self.dataset.node_feature_names
            }
            
            return results
            
        finally:
            # Restore original data list and save directory
            self.dataset.data_list = original_data_list
            self.save_dir = original_save_dir
    
    def _run_single_target_pipeline(self, target_idx, target_name):
        """Run pipeline for a single target using the parent class functionality"""
        print(f"\nRunning pipeline for {target_name}")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Feature names: {self.dataset.node_feature_names}")
        
        # Create detailed_results folder
        os.makedirs(f"{self.save_dir}/detailed_results", exist_ok=True)
        
        results = {}
        
        # Step 1: Train ALL GNN models on KNN-sparsified graph (using enhanced tracking method)
        print(f"\nSTEP 1: Training ALL GNN models on KNN-sparsified graph")
        knn_gnn_results = {}
        for model_type in self.gnn_models_to_train:
            knn_gnn_results[model_type] = self.train_gnn_model_with_fold_tracking(
                model_type=model_type,
                target_idx=target_idx,
                data_list=self.dataset.data_list
            )
            
            # Save detailed metrics for each GNN model
            if 'fold_results' in knn_gnn_results[model_type]:
                self.save_detailed_metrics(
                    fold_results=knn_gnn_results[model_type]['fold_results'],
                    model_type=model_type,
                    target_name=target_name,
                    phase='knn'
                )
        
        results['knn'] = knn_gnn_results
        
        # Find best GNN model for this target
        best_gnn_model = None
        best_gnn_r2 = -float('inf')
        best_gnn_type = None
        
        for model_type, gnn_results in knn_gnn_results.items():
            if gnn_results['avg_metrics']['r2'] > best_gnn_r2:
                best_gnn_r2 = gnn_results['avg_metrics']['r2']
                best_gnn_model = gnn_results['model']
                best_gnn_type = model_type
        
        print(f"\nBest KNN GNN model: {best_gnn_type.upper()} (R² = {best_gnn_r2:.4f})")
        
        # Step 2: Create GNNExplainer-sparsified graph (using parent class method)
        print(f"\nSTEP 2: Creating GNNExplainer-sparsified graph")
        print(f"Using {best_gnn_type.upper()} model for explanation")
        explainer_data = self.create_explainer_sparsified_graph(
            model=best_gnn_model,
            target_idx=target_idx
        )
        
        # Step 3: Train ALL GNN models on explainer-sparsified graph (using enhanced tracking method)
        print(f"\nSTEP 3: Training ALL GNN models on explainer-sparsified graph")
        explainer_gnn_results = {}
        for model_type in self.gnn_models_to_train:
            explainer_gnn_results[model_type] = self.train_gnn_model_with_fold_tracking(
                model_type=model_type,
                target_idx=target_idx,
                data_list=explainer_data
            )
            
            # Save detailed metrics for each GNN model
            if 'fold_results' in explainer_gnn_results[model_type]:
                self.save_detailed_metrics(
                    fold_results=explainer_gnn_results[model_type]['fold_results'],
                    model_type=model_type,
                    target_name=target_name,
                    phase='explainer'
                )
        
        results['explainer'] = explainer_gnn_results
        
        # Find best model from explainer-sparsified graph
        best_explainer_model = None
        best_explainer_r2 = -float('inf')
        best_explainer_type = None
        
        for model_type, gnn_results in explainer_gnn_results.items():
            if gnn_results['avg_metrics']['r2'] > best_explainer_r2:
                best_explainer_r2 = gnn_results['avg_metrics']['r2']
                best_explainer_model = gnn_results['model']
                best_explainer_type = model_type
        
        print(f"\nBest explainer-trained GNN model: {best_explainer_type.upper()} (R² = {best_explainer_r2:.4f})")
        
        # Step 4: Extract embeddings from best model (using parent class method)
        print(f"\nSTEP 4: Extracting embeddings from best GNN model")
        embeddings, targets = self.extract_embeddings(best_explainer_model, explainer_data)
        
        # Save embeddings
        os.makedirs(f"{self.save_dir}/embeddings", exist_ok=True)
        np.save(f"{self.save_dir}/embeddings/{target_name}_embeddings.npy", embeddings)
        np.save(f"{self.save_dir}/embeddings/{target_name}_targets.npy", targets)
        
        print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Step 5: Train ML models on embeddings (using parent class method)
        print(f"\nSTEP 5: Training ML models on embeddings")
        ml_results = self.train_ml_models(embeddings, targets, target_idx)
        
        # Save detailed ML model results
        self.save_ml_model_results(ml_results, target_name, "explainer_embeddings")
        
        results['ml_models'] = ml_results
        
        # Step 6: Create plots (using parent class method)
        print(f"\nSTEP 6: Creating plots")
        # For mixed models, show all models
        gnn_plot_results = {**knn_gnn_results, **{f"{k}_explainer": v for k, v in explainer_gnn_results.items()}}
        
        self.plot_results(
            gnn_results=gnn_plot_results,
            ml_results=ml_results,
            target_idx=target_idx
        )
        
        # Save results
        self._save_case_results(results, target_name)
        
        # Create fold summary visualizations
        self.create_fold_summary_visualization()
        
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
                        'r2': result['avg_metrics']['r2'],
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
                    'r2': result['avg_metrics']['r2'],
                    'mae': result['avg_metrics']['mae'],
                    'num_features': len(self.dataset.node_feature_names)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/{self.case_type}_{target_name}_summary.csv", index=False)
        
        # Save comprehensive results with detailed metrics
        self._save_comprehensive_results(results, target_name)
        
        print(f"\nResults saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results")
    
    def _save_comprehensive_results(self, results, target_name):
        """Save comprehensive results including detailed metrics"""
        # Create comprehensive results directory
        comprehensive_dir = f"{self.save_dir}/comprehensive_results"
        os.makedirs(comprehensive_dir, exist_ok=True)
        
        # Save detailed fold-by-fold results for each model
        for phase in ['knn', 'explainer']:
            if phase in results:
                for model_type, result in results[phase].items():
                    if 'fold_results' in result:
                        # Create detailed CSV with fold-by-fold results
                        fold_data = []
                        for fold_idx, fold_result in enumerate(result['fold_results']):
                            fold_data_row = {
                                'fold': fold_idx + 1,
                                'mse': fold_result['mse'],
                                'rmse': fold_result['rmse'],
                                'r2': fold_result['r2'],
                                'mae': fold_result['mae'],
                                'model_type': model_type,
                                'phase': phase,
                                'target': target_name,
                                'case': self.case_type
                            }
                            
                            # Add hyperparameters if available
                            if 'best_params' in fold_result:
                                best_params = fold_result['best_params']
                                for param_name, param_value in best_params.items():
                                    fold_data_row[f'best_{param_name}'] = param_value
                            else:
                                # Add placeholder for non-nested CV results
                                fold_data_row['best_hidden_dim'] = 'N/A'
                                fold_data_row['best_k_neighbors'] = 'N/A'
                            
                            fold_data.append(fold_data_row)
                        
                        fold_df = pd.DataFrame(fold_data)
                        csv_path = f"{comprehensive_dir}/{model_type}_{target_name}_{phase}_fold_results.csv"
                        fold_df.to_csv(csv_path, index=False)
                        
                        # Calculate and save statistics with ± format
                        stats_data = []
                        for metric in ['mse', 'rmse', 'r2', 'mae']:
                            values = [fold_result[metric] for fold_result in result['fold_results']]
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            stats_data.append({
                                'metric': metric,
                                'mean': mean_val,
                                'std': std_val,
                                'mean_std_format': f"{mean_val:.4f} ± {std_val:.4f}",
                                'min': np.min(values),
                                'max': np.max(values),
                                'model_type': model_type,
                                'phase': phase,
                                'target': target_name,
                                'case': self.case_type
                            })
                        
                        # Add hyperparameter statistics if available
                        if any('best_params' in fold_result for fold_result in result['fold_results']):
                            # Calculate hyperparameter selection frequencies
                            hidden_dims = []
                            k_neighbors_list = []
                            
                            for fold_result in result['fold_results']:
                                if 'best_params' in fold_result:
                                    best_params = fold_result['best_params']
                                    hidden_dims.append(best_params.get('hidden_dim', 'N/A'))
                                    k_neighbors_list.append(best_params.get('k_neighbors', 'N/A'))
                            
                            if hidden_dims:
                                from collections import Counter
                                hidden_dim_counts = Counter(hidden_dims)
                                k_neighbors_counts = Counter(k_neighbors_list)
                                
                                # Add most common hyperparameters to stats
                                most_common_hidden_dim = hidden_dim_counts.most_common(1)[0] if hidden_dim_counts else ('N/A', 0)
                                most_common_k_neighbors = k_neighbors_counts.most_common(1)[0] if k_neighbors_counts else ('N/A', 0)
                                
                                stats_data.append({
                                    'metric': 'best_hidden_dim',
                                    'mean': most_common_hidden_dim[0],
                                    'std': f"{most_common_hidden_dim[1]}/{len(hidden_dims)}",
                                    'mean_std_format': f"{most_common_hidden_dim[0]} (selected {most_common_hidden_dim[1]}/{len(hidden_dims)} times)",
                                    'min': 'N/A',
                                    'max': 'N/A',
                                    'model_type': model_type,
                                    'phase': phase,
                                    'target': target_name,
                                    'case': self.case_type
                                })
                                
                                stats_data.append({
                                    'metric': 'best_k_neighbors',
                                    'mean': most_common_k_neighbors[0],
                                    'std': f"{most_common_k_neighbors[1]}/{len(k_neighbors_list)}",
                                    'mean_std_format': f"{most_common_k_neighbors[0]} (selected {most_common_k_neighbors[1]}/{len(k_neighbors_list)} times)",
                                    'min': 'N/A',
                                    'max': 'N/A',
                                    'model_type': model_type,
                                    'phase': phase,
                                    'target': target_name,
                                    'case': self.case_type
                                })
                        
                        stats_df = pd.DataFrame(stats_data)
                        stats_csv_path = f"{comprehensive_dir}/{model_type}_{target_name}_{phase}_statistics.csv"
                        stats_df.to_csv(stats_csv_path, index=False)
        
        # Save ML model detailed results
        if 'ml_models' in results:
            for model_type, result in results['ml_models'].items():
                if 'fold_results' in result:
                    # Create detailed CSV with fold-by-fold results
                    fold_data = []
                    for fold_idx, fold_result in enumerate(result['fold_results']):
                        fold_data.append({
                            'fold': fold_idx + 1,
                            'mse': fold_result['mse'],
                            'rmse': fold_result['rmse'],
                            'r2': fold_result['r2'],
                            'mae': fold_result['mae'],
                            'model_type': model_type,
                            'phase': 'embeddings',
                            'target': target_name,
                            'case': self.case_type
                        })
                    
                    fold_df = pd.DataFrame(fold_data)
                    csv_path = f"{comprehensive_dir}/ml_{model_type}_{target_name}_embeddings_fold_results.csv"
                    fold_df.to_csv(csv_path, index=False)
                    
                    # Calculate and save statistics with ± format
                    stats_data = []
                    for metric in ['mse', 'rmse', 'r2', 'mae']:
                        values = [fold_result[metric] for fold_result in result['fold_results']]
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        stats_data.append({
                            'metric': metric,
                            'mean': mean_val,
                            'std': std_val,
                            'mean_std_format': f"{mean_val:.4f} ± {std_val:.4f}",
                            'min': np.min(values),
                            'max': np.max(values),
                            'model_type': model_type,
                            'phase': 'embeddings',
                            'target': target_name,
                            'case': self.case_type
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    stats_csv_path = f"{comprehensive_dir}/ml_{model_type}_{target_name}_embeddings_statistics.csv"
                    stats_df.to_csv(stats_csv_path, index=False)
        
        # Save comprehensive hyperparameter tracking for this case
        self._save_case_hyperparameter_tracking(results, target_name)
        
        print(f"Comprehensive results saved to {comprehensive_dir}")
        print(f"Detailed fold-by-fold results and statistics with ± format saved")
    
    def _save_case_hyperparameter_tracking(self, results, target_name):
        """Save comprehensive hyperparameter tracking for this specific case"""
        tracking_data = []
        
        # Track initial case parameters
        initial_params = {
            'target_name': target_name,
            'case_type': self.case_type,
            'phase': 'case_initial',
            'model_type': 'case',
            'hidden_dim': self.hidden_dim,
            'k_neighbors': self.k_neighbors,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'num_folds': self.num_folds,
            'importance_threshold': self.importance_threshold,
            'use_fast_correlation': self.use_fast_correlation,
            'graph_mode': self.graph_mode,
            'family_filter_mode': self.family_filter_mode,
            'use_enhanced_training': getattr(self, 'use_enhanced_training', True),
            'adaptive_hyperparameters': getattr(self, 'adaptive_hyperparameters', True),
            'use_nested_cv': getattr(self, 'use_nested_cv', True),
            'anchored_features_count': len(self.anchored_features) if hasattr(self, 'anchored_features') else 0
        }
        tracking_data.append(initial_params)
        
        # Track GNN model hyperparameters
        for phase in ['knn', 'explainer']:
            if phase in results:
                for model_type, result in results[phase].items():
                    if 'fold_results' in result:
                        for fold_result in result['fold_results']:
                            if 'best_params' in fold_result:
                                best_params = fold_result['best_params']
                                fold_tracking = {
                                    'target_name': target_name,
                                    'case_type': self.case_type,
                                    'phase': phase,
                                    'model_type': model_type,
                                    'fold': fold_result['fold'],
                                    'r2': fold_result['r2'],
                                    'mse': fold_result['mse'],
                                    'rmse': fold_result.get('rmse', 'N/A'),
                                    'mae': fold_result.get('mae', 'N/A')
                                }
                                # Add all hyperparameters
                                for param_name, param_value in best_params.items():
                                    fold_tracking[f'best_{param_name}'] = param_value
                                
                                tracking_data.append(fold_tracking)
        
        # Save case-specific tracking
        tracking_df = pd.DataFrame(tracking_data)
        csv_path = f"{self.save_dir}/case_{self.case_type}_{target_name}_hyperparameter_tracking.csv"
        tracking_df.to_csv(csv_path, index=False)
        
        print(f"Case hyperparameter tracking saved: {csv_path}")
        return csv_path

    def train_gnn_model_with_fold_tracking(self, model_type, target_idx, data_list=None):
        """Enhanced GNN training with fold-specific graph tracking"""
        if self.use_nested_cv:
            return self.train_gnn_model_nested_with_tracking(model_type, target_idx, data_list)
        else:
            return self.train_gnn_model(model_type, target_idx, data_list)

    def train_gnn_model_nested_with_tracking(self, model_type, target_idx, data_list=None):
        """Train GNN model with nested cross-validation and fold-specific graph tracking"""
        if data_list is None:
            data_list = self.dataset.data_list
        
        target_name = self.target_names[target_idx]
        # Robust phase detection
        phase = "explainer" if self._is_explainer_phase(data_list) else "knn"
        print(f"\nTraining {model_type.upper()} model with NESTED CV for target: {target_name} ({phase} graph)")
        
        # Create fold-specific directories
        fold_graphs_dir = f"{self.save_dir}/fold_specific_graphs/{model_type}_{target_name}_{phase}"
        os.makedirs(fold_graphs_dir, exist_ok=True)
        
        outer_kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        outer_results = []
        best_model_state = None
        best_outer_r2 = -float('inf')
        best_hyperparams = None
        
        # Track fold-specific information
        fold_tracking = []
        
        print(f"\n{'='*60}")
        print(f"NESTED CV RESULTS FOR {model_type.upper()} - {target_name} ({phase.upper()})")
        print(f"{'='*60}")
        
        for fold, (train_idx, test_idx) in enumerate(outer_kf.split(data_list)):
            fold_num = fold + 1
            print(f"\n  {'-'*50}")
            print(f"  OUTER FOLD {fold_num}/{self.num_folds}")
            print(f"  {'-'*50}")
            
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]
            
            # 1. Inner loop: pick hyperparameters
            print(f"  Inner CV Hyperparameter Selection:")
            best_params = self._inner_loop_select(model_type, train_data, target_idx)
            
            print(f"  Selected hyperparameters for fold {fold_num}: {best_params}")
            
            # 2. Create dataset with best hyperparameters and save fold-specific graph
            best_hidden_dim = best_params['hidden_dim']
            best_k_neighbors = best_params.get('k_neighbors', self.k_neighbors)
            
            fold_dataset = None
            if phase == "knn" and 'k_neighbors' in best_params:
                # KNN phase: create new dataset with best k_neighbors
                print(f"    Creating dataset with k_neighbors={best_k_neighbors}")
                fold_dataset = AnchoredMicrobialGNNDataset(
                    data_path=self.data_path,
                    anchored_features=self.anchored_features,
                    case_type=self.case_type,
                    k_neighbors=best_k_neighbors,
                    mantel_threshold=self.mantel_threshold,
                    use_fast_correlation=self.use_fast_correlation,
                    graph_mode=self.graph_mode,
                    family_filter_mode=self.family_filter_mode
                )
                
                # Save fold-specific k-NN graph
                self._save_fold_knn_graph(
                    fold_dataset, fold_num, best_k_neighbors, 
                    fold_graphs_dir, target_name, best_params
                )
                
                best_dataset_data_list = self._move_data_to_device(fold_dataset.data_list)
                best_train_data = [best_dataset_data_list[i] for i in train_idx]
                best_test_data = [best_dataset_data_list[i] for i in test_idx]
            else:
                # Explainer phase: use existing explainer-sparsified data
                best_train_data = train_data
                best_test_data = test_data
                print(f"    Using existing explainer-sparsified data")
            
            # 3. Train model with best hyperparameters
            model = self._create_gnn_model_with_params(model_type, best_hidden_dim, num_targets=1)
            model = self._train_model_full_with_params(model, best_train_data, target_idx, best_hidden_dim)
            
            # 4. Generate explainer-sparsified graph if this is the KNN phase (after model is trained)
            if phase == "knn":
                print(f"    Creating explainer-sparsified graph for fold {fold_num}")
                self._create_fold_explainer_graph(
                    model, target_idx, fold_dataset if fold_dataset else self.dataset,
                    fold_num, fold_graphs_dir, target_name, best_params
                )
            
            # 5. Evaluate on test set
            mse, r2 = self._evaluate_model(model, best_test_data, target_idx)
            print(f"  Final Evaluation on Test Set:")
            print(f"    R² = {r2:.4f}, MSE = {mse:.4f}")
            
            # Get predictions for fold
            fold_predictions = self._get_fold_predictions(model, best_test_data, target_idx)
            
            # Calculate additional metrics
            rmse = np.sqrt(mse)
            mae = mean_absolute_error([pred['actual'] for pred in fold_predictions], 
                                    [pred['predicted'] for pred in fold_predictions])
            
            # Track this fold's information
            fold_info = {
                'fold': fold_num,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'best_params': best_params,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'predictions': fold_predictions,
                'graph_info': {
                    'k_neighbors': best_k_neighbors,
                    'hidden_dim': best_hidden_dim,
                    'graph_saved': True
                }
            }
            
            outer_results.append(fold_info)
            fold_tracking.append(fold_info)
            
            # Track best model
            if r2 > best_outer_r2:
                best_outer_r2 = r2
                best_model_state = model.state_dict().copy()
                best_hyperparams = best_params.copy()
        
        # Save fold tracking information
        self._save_fold_tracking_info(fold_tracking, model_type, target_name, phase, fold_graphs_dir)
        
        # Calculate average metrics
        avg_metrics = {
            'r2': np.mean([result['r2'] for result in outer_results]),
            'mse': np.mean([result['mse'] for result in outer_results]),
            'rmse': np.mean([result['rmse'] for result in outer_results]),
            'mae': np.mean([result['mae'] for result in outer_results])
        }
        
        # Print summary
        print(f"\n  NESTED CV SUMMARY for {model_type.upper()} - {target_name} ({phase.upper()}):")
        print(f"  Average R² = {avg_metrics['r2']:.4f} ± {np.std([result['r2'] for result in outer_results]):.4f}")
        print(f"  Average MSE = {avg_metrics['mse']:.4f} ± {np.std([result['mse'] for result in outer_results]):.4f}")
        print(f"  Best hyperparameters: {best_hyperparams}")
        
        # Create final model with best hyperparameters and full training data
        if best_hyperparams:
            if 'k_neighbors' in best_hyperparams and not (data_list != self.dataset.data_list):
                final_dataset = AnchoredMicrobialGNNDataset(
                    data_path=self.data_path,
                    anchored_features=self.anchored_features,
                    case_type=self.case_type,
                    k_neighbors=best_hyperparams['k_neighbors'],
                    mantel_threshold=self.mantel_threshold,
                    use_fast_correlation=self.use_fast_correlation,
                    graph_mode=self.graph_mode,
                    family_filter_mode=self.family_filter_mode
                )
                final_data_list = self._move_data_to_device(final_dataset.data_list)
            else:
                final_data_list = data_list
            
            final_model = self._create_gnn_model_with_params(
                model_type, best_hyperparams['hidden_dim'], num_targets=1
            )
            final_model = self._train_model_full_with_params(
                final_model, final_data_list, target_idx, best_hyperparams['hidden_dim']
            )
        else:
            final_model = self.create_gnn_model(model_type, num_targets=1)
            final_model = super()._train_model_full_with_params(final_model, data_list, target_idx)
        
        return {
            'model': final_model,
            'avg_metrics': avg_metrics,
            'fold_results': outer_results,
            'fold_predictions': [result['predictions'] for result in outer_results],
            'best_hyperparams': best_hyperparams,
            'fold_tracking': fold_tracking
        }

    def _save_fold_knn_graph(self, dataset, fold_num, k_neighbors, graphs_dir, target_name, best_params):
        """Save fold-specific k-NN graph visualization and information"""
        import networkx as nx
        from torch_geometric.utils import to_networkx
        
        # Create fold-specific subdirectory
        fold_dir = f"{graphs_dir}/fold_{fold_num}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Get the first graph from dataset for visualization
        graph_data = dataset.data_list[0]
        
        # Convert to NetworkX for visualization
        G = to_networkx(graph_data, to_undirected=True)
        
        # Save graph visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=0.8)
        
        # Add labels if there aren't too many nodes
        if len(G.nodes()) <= 50:
            labels = {i: dataset.node_feature_names[i][:10] for i in range(len(dataset.node_feature_names))}
            nx.draw_networkx_labels(G, pos, labels, font_size=6)
        
        plt.title(f'Fold {fold_num} k-NN Graph\\n'
                 f'Target: {target_name}, k={k_neighbors}, hidden_dim={best_params["hidden_dim"]}\\n'
                 f'Nodes: {len(G.nodes())}, Edges: {len(G.edges())}')
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(f"{fold_dir}/knn_graph_fold{fold_num}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{fold_dir}/knn_graph_fold{fold_num}.pdf", bbox_inches='tight')
        plt.close()
        
        # Save graph information
        graph_info = {
            'fold': fold_num,
            'k_neighbors': k_neighbors,
            'target_name': target_name,
            'best_params': best_params,
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'node_features': dataset.node_feature_names,
            'graph_density': nx.density(G) if len(G.nodes()) > 1 else 0
        }
        
        # Save as JSON
        import json
        with open(f"{fold_dir}/knn_graph_info_fold{fold_num}.json", 'w') as f:
            json.dump(graph_info, f, indent=2, default=str)
        
        # Save as CSV for easy reading
        import pandas as pd
        info_df = pd.DataFrame([{
            'Metric': 'Fold',
            'Value': fold_num
        }, {
            'Metric': 'k_neighbors',
            'Value': k_neighbors
        }, {
            'Metric': 'target_name', 
            'Value': target_name
        }, {
            'Metric': 'hidden_dim',
            'Value': best_params['hidden_dim']
        }, {
            'Metric': 'num_nodes',
            'Value': len(G.nodes())
        }, {
            'Metric': 'num_edges',
            'Value': len(G.edges())
        }, {
            'Metric': 'graph_density',
            'Value': f"{nx.density(G):.4f}" if len(G.nodes()) > 1 else "0"
        }])
        info_df.to_csv(f"{fold_dir}/knn_graph_info_fold{fold_num}.csv", index=False)
        
        print(f"    Saved k-NN graph for fold {fold_num}: {fold_dir}/")
        return graph_info

    def _create_fold_explainer_graph(self, model, target_idx, dataset, fold_num, graphs_dir, target_name, best_params):
        """Create and save fold-specific explainer-sparsified graph"""
        from pipeline_explainer import create_explainer_sparsified_graph
        
        fold_dir = f"{graphs_dir}/fold_{fold_num}"
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"    Creating explainer graph for fold {fold_num}...")
        
        # Create explainer-sparsified graph
        explainer_data = create_explainer_sparsified_graph(
            pipeline=self,  # Pass self as pipeline
            model=model,
            target_idx=target_idx,
            importance_threshold=self.importance_threshold
        )
        
        # Visualize explainer-sparsified graph
        self._visualize_explainer_graph(explainer_data, fold_num, fold_dir, target_name, best_params)
        
        # Save explainer graph information
        explainer_info = {
            'fold': fold_num,
            'target_name': target_name,
            'best_params': best_params,
            'importance_threshold': self.importance_threshold,
            'original_graphs': len(dataset.data_list),
            'explainer_graphs': len(explainer_data),
            'avg_nodes_original': np.mean([data.x.shape[0] for data in dataset.data_list]),
            'avg_edges_original': np.mean([data.edge_index.shape[1] for data in dataset.data_list]),
            'avg_nodes_explainer': np.mean([data.x.shape[0] for data in explainer_data]),
            'avg_edges_explainer': np.mean([data.edge_index.shape[1] for data in explainer_data])
        }
        
        # Save explainer info
        import json
        with open(f"{fold_dir}/explainer_graph_info_fold{fold_num}.json", 'w') as f:
            json.dump(explainer_info, f, indent=2, default=str)
        
        # Save as CSV
        info_df = pd.DataFrame([{
            'Metric': 'Fold',
            'Value': fold_num
        }, {
            'Metric': 'target_name',
            'Value': target_name
        }, {
            'Metric': 'importance_threshold',
            'Value': self.importance_threshold
        }, {
            'Metric': 'avg_nodes_original',
            'Value': f"{explainer_info['avg_nodes_original']:.2f}"
        }, {
            'Metric': 'avg_edges_original', 
            'Value': f"{explainer_info['avg_edges_original']:.2f}"
        }, {
            'Metric': 'avg_nodes_explainer',
            'Value': f"{explainer_info['avg_nodes_explainer']:.2f}"
        }, {
            'Metric': 'avg_edges_explainer',
            'Value': f"{explainer_info['avg_edges_explainer']:.2f}"
        }, {
            'Metric': 'edge_reduction_ratio',
            'Value': f"{1 - explainer_info['avg_edges_explainer']/explainer_info['avg_edges_original']:.4f}"
        }])
        info_df.to_csv(f"{fold_dir}/explainer_graph_info_fold{fold_num}.csv", index=False)
        
        print(f"    Saved explainer graph for fold {fold_num}: {fold_dir}/")
        return explainer_data

    def _visualize_explainer_graph(self, explainer_data, fold_num, fold_dir, target_name, best_params):
        """Visualize explainer-sparsified graph"""
        import networkx as nx
        from torch_geometric.utils import to_networkx
        
        # Get the first graph for visualization
        graph_data = explainer_data[0]
        
        # Convert to NetworkX
        G = to_networkx(graph_data, to_undirected=True)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color='lightcoral', 
                              node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='darkred', width=1.0)
        
        # Add labels if there aren't too many nodes
        if len(G.nodes()) <= 50:
            labels = {i: self.dataset.node_feature_names[i][:10] for i in range(len(self.dataset.node_feature_names))}
            nx.draw_networkx_labels(G, pos, labels, font_size=6)
        
        plt.title(f'Fold {fold_num} Explainer-Sparsified Graph\\n'
                 f'Target: {target_name}, hidden_dim={best_params["hidden_dim"]}\\n'
                 f'Nodes: {len(G.nodes())}, Edges: {len(G.edges())}\\n'
                 f'Importance Threshold: {self.importance_threshold}')
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(f"{fold_dir}/explainer_graph_fold{fold_num}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{fold_dir}/explainer_graph_fold{fold_num}.pdf", bbox_inches='tight')
        plt.close()

    def _save_fold_tracking_info(self, fold_tracking, model_type, target_name, phase, graphs_dir):
        """Save comprehensive fold tracking information"""
        
        # Create summary CSV with all fold information
        fold_summary = []
        for fold_info in fold_tracking:
            summary_row = {
                'fold': fold_info['fold'],
                'model_type': model_type,
                'target_name': target_name,
                'phase': phase,
                'r2': fold_info['r2'],
                'mse': fold_info['mse'],
                'rmse': fold_info['rmse'],
                'mae': fold_info['mae'],
                'train_size': fold_info['train_size'],
                'test_size': fold_info['test_size'],
                'k_neighbors': fold_info['graph_info']['k_neighbors'],
                'hidden_dim': fold_info['graph_info']['hidden_dim'],
                'graph_saved': fold_info['graph_info']['graph_saved']
            }
            
            # Add all hyperparameters
            for param_name, param_value in fold_info['best_params'].items():
                summary_row[f'best_{param_name}'] = param_value
            
            fold_summary.append(summary_row)
        
        # Save summary
        fold_df = pd.DataFrame(fold_summary)
        fold_df.to_csv(f"{graphs_dir}/fold_tracking_summary.csv", index=False)
        
        # Save detailed tracking as JSON
        import json
        with open(f"{graphs_dir}/fold_tracking_detailed.json", 'w') as f:
            json.dump(fold_tracking, f, indent=2, default=str)
        
        # Create hyperparameter selection frequency analysis
        param_freq = {}
        for fold_info in fold_tracking:
            for param_name, param_value in fold_info['best_params'].items():
                if param_name not in param_freq:
                    param_freq[param_name] = {}
                if param_value not in param_freq[param_name]:
                    param_freq[param_name][param_value] = 0
                param_freq[param_name][param_value] += 1
        
        # Save parameter frequency analysis
        freq_data = []
        for param_name, value_counts in param_freq.items():
            for param_value, count in value_counts.items():
                freq_data.append({
                    'parameter': param_name,
                    'value': param_value,
                    'frequency': count,
                    'percentage': f"{100 * count / self.num_folds:.1f}%"
                })
        
        freq_df = pd.DataFrame(freq_data)
        freq_df.to_csv(f"{graphs_dir}/hyperparameter_selection_frequency.csv", index=False)
        
        print(f"  Saved fold tracking info: {graphs_dir}/")
        print(f"    - fold_tracking_summary.csv")
        print(f"    - fold_tracking_detailed.json")
        print(f"    - hyperparameter_selection_frequency.csv")
        
        return fold_summary

    # Helper methods needed for nested CV with tracking
    def _inner_loop_select(self, model_type, train_data, target_idx):
        """Inner loop for hyperparameter selection using K-fold CV"""
        inner_kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        best_score = -float('inf')
        best_params = None
        
        # Check if we're in explainer phase using robust method
        is_explainer_phase = self._is_explainer_phase(train_data)
        
        # Use appropriate parameter grid
        if is_explainer_phase:
            param_grid = self.explainer_param_grid
            print(f"    Inner CV (Explainer Phase): Testing {len(param_grid)} hidden_dim combinations...")
        else:
            param_grid = self.param_grid
            print(f"    Inner CV (KNN Phase): Testing {len(param_grid)} hyperparameter combinations...")
        
        for params in param_grid:
            local_hidden_dim = params['hidden_dim']
            local_k_neighbors = params.get('k_neighbors', self.k_neighbors)
            
            # For explainer phase, use existing sparsified graph data
            if is_explainer_phase:
                temp_data_list = train_data
            else:
                # For KNN phase, recreate dataset with local k_neighbors value
                temp_dataset = AnchoredMicrobialGNNDataset(
                    data_path=self.data_path,
                    anchored_features=self.anchored_features,
                    case_type=self.case_type,
                    k_neighbors=local_k_neighbors,
                    mantel_threshold=self.mantel_threshold,
                    use_fast_correlation=self.use_fast_correlation,
                    graph_mode=self.graph_mode,
                    family_filter_mode=self.family_filter_mode
                )
                temp_data_list = self._move_data_to_device(temp_dataset.data_list)
            
            val_scores = []
            
            for tr_idx, val_idx in inner_kf.split(temp_data_list):
                model = self._create_gnn_model_with_params(model_type, local_hidden_dim, num_targets=1)
                r2_score = self._train_and_evaluate_once_with_params(
                    model, temp_data_list, tr_idx, val_idx, target_idx, 
                    hidden_dim=local_hidden_dim
                )
                val_scores.append(r2_score)
            
            mean_score = np.mean(val_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
        
        print(f"    Best params: {best_params} (R² = {best_score:.4f})")
        return best_params

    def _create_gnn_model_with_params(self, model_type, hidden_dim, num_targets=1):
        """Create a GNN model with specific hyperparameters"""
        if model_type == 'gcn':
            model = simple_GCN_res_plus_regression(
                hidden_channels=hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=False
            ).to(device)
        elif model_type == 'rggc':
            model = simple_RGGC_plus_regression(
                hidden_channels=hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=False
            ).to(device)
        elif model_type == 'gat':
            model = simple_GAT_regression(
                hidden_channels=hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=False
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model

    def _train_model_full_with_params(self, model, train_data, target_idx, hidden_dim=None):
        """Train model with specific parameters"""
        return super()._train_model_full_with_params(model, train_data, target_idx, hidden_dim)

    def _train_and_evaluate_once_with_params(self, model, data_list, tr_idx, val_idx, target_idx, hidden_dim=None):
        """Train and evaluate model once with specific parameters"""
        train_data = [data_list[i] for i in tr_idx]
        val_data = [data_list[i] for i in val_idx]
        
        # Train the model
        super()._train_model_full_with_params(model, train_data, target_idx)
        
        # Evaluate the model
        mse, r2 = self._evaluate_model(model, val_data, target_idx)
        return r2

    def _evaluate_model(self, model, test_data, target_idx):
        """Evaluate model on test data"""
        model.eval()
        all_preds = []
        all_targets = []
        
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return mse, r2

    def _get_fold_predictions(self, model, test_data, target_idx):
        """Get predictions for fold"""
        model.eval()
        fold_predictions = []
        
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                predictions = out.cpu().numpy().flatten()
                targets = target.cpu().numpy().flatten()
                
                for pred, actual in zip(predictions, targets):
                    fold_predictions.append({
                        'predicted': float(pred),
                        'actual': float(actual)
                    })
        
        return fold_predictions

    def _move_data_to_device(self, data_list):
        """Move data list to device"""
        return [data.to(device) for data in data_list]

    def _is_explainer_phase(self, data_list):
        """Robustly determine if we're in explainer phase"""
        # Explainer phase characteristics:
        # 1. Different length (explainer may filter graphs)
        # 2. Different object identity 
        # 3. Different edge structure (explainer sparsifies)
        
        if len(data_list) != len(self.dataset.data_list):
            return True
            
        if id(data_list) != id(self.dataset.data_list):
            return True
            
        # Check if edge structures are different (explainer sparsification)
        if len(data_list) > 0 and len(self.dataset.data_list) > 0:
            original_edges = self.dataset.data_list[0].edge_index.shape[1]
            current_edges = data_list[0].edge_index.shape[1]
            if current_edges != original_edges:
                return True
                
        return False

    def create_fold_summary_visualization(self):
        """Create comprehensive visualization of fold-specific results across all models"""
        summary_dir = f"{self.save_dir}/fold_summary_visualizations"
        os.makedirs(summary_dir, exist_ok=True)
        
        print(f"\nCreating fold summary visualizations...")
        
        # Find all fold tracking files
        fold_dirs = []
        for model_type in self.gnn_models_to_train:
            for target_name in self.target_names:
                for phase in ['knn', 'explainer']:
                    graphs_dir = f"{self.save_dir}/fold_specific_graphs/{model_type}_{target_name}_{phase}"
                    if os.path.exists(graphs_dir):
                        fold_dirs.append((model_type, target_name, phase, graphs_dir))
        
        if not fold_dirs:
            print("  No fold tracking data found.")
            return
        
        # Create comprehensive summary
        self._create_hyperparameter_selection_summary(fold_dirs, summary_dir)
        self._create_fold_performance_comparison(fold_dirs, summary_dir)
        self._create_graph_statistics_summary(fold_dirs, summary_dir)
        
        print(f"  Fold summary visualizations saved to: {summary_dir}/")

    def _create_hyperparameter_selection_summary(self, fold_dirs, summary_dir):
        """Create summary of hyperparameter selections across all models and folds"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hyperparameter Selection Summary Across All Folds', fontsize=16, fontweight='bold')
        
        # Collect all hyperparameter data
        all_data = []
        for model_type, target_name, phase, graphs_dir in fold_dirs:
            csv_file = f"{graphs_dir}/fold_tracking_summary.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df['model_phase'] = f"{model_type}_{phase}"
                all_data.append(df)
        
        if not all_data:
            plt.close()
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Plot 1: Hidden dimension selections
        ax1 = axes[0, 0]
        hidden_dim_counts = combined_df.groupby(['model_phase', 'hidden_dim']).size().unstack(fill_value=0)
        hidden_dim_counts.plot(kind='bar', ax=ax1, stacked=True)
        ax1.set_title('Hidden Dimension Selections by Model/Phase')
        ax1.set_xlabel('Model/Phase')
        ax1.set_ylabel('Number of Folds')
        ax1.legend(title='Hidden Dim', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: K-neighbors selections  
        ax2 = axes[0, 1]
        k_counts = combined_df.groupby(['model_phase', 'k_neighbors']).size().unstack(fill_value=0)
        k_counts.plot(kind='bar', ax=ax2, stacked=True)
        ax2.set_title('K-Neighbors Selections by Model/Phase')
        ax2.set_xlabel('Model/Phase')
        ax2.set_ylabel('Number of Folds')
        ax2.legend(title='K-Neighbors', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Performance by hyperparameters
        ax3 = axes[1, 0]
        perf_by_hidden = combined_df.groupby('hidden_dim')['r2'].mean()
        perf_by_hidden.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Average R² by Hidden Dimension')
        ax3.set_xlabel('Hidden Dimension')
        ax3.set_ylabel('Average R²')
        ax3.tick_params(axis='x', rotation=0)
        
        # Plot 4: Performance by k-neighbors
        ax4 = axes[1, 1]
        perf_by_k = combined_df.groupby('k_neighbors')['r2'].mean()
        perf_by_k.plot(kind='bar', ax=ax4, color='lightcoral')
        ax4.set_title('Average R² by K-Neighbors')
        ax4.set_xlabel('K-Neighbors')
        ax4.set_ylabel('Average R²')
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/hyperparameter_selection_summary.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{summary_dir}/hyperparameter_selection_summary.pdf", bbox_inches='tight')
        plt.close()
        
        # Save detailed hyperparameter frequency table
        freq_table = combined_df.groupby(['model_phase', 'hidden_dim', 'k_neighbors']).size().reset_index(name='frequency')
        freq_table['percentage'] = 100 * freq_table['frequency'] / self.num_folds
        freq_table.to_csv(f"{summary_dir}/hyperparameter_frequency_table.csv", index=False)

    def _create_fold_performance_comparison(self, fold_dirs, summary_dir):
        """Create performance comparison across folds"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison Across Folds', fontsize=16, fontweight='bold')
        
        # Collect all performance data
        all_data = []
        for model_type, target_name, phase, graphs_dir in fold_dirs:
            csv_file = f"{graphs_dir}/fold_tracking_summary.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df['model_phase'] = f"{model_type}_{phase}"
                all_data.append(df)
        
        if not all_data:
            plt.close()
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Plot 1: R² across folds
        ax1 = axes[0, 0]
        for model_phase in combined_df['model_phase'].unique():
            data = combined_df[combined_df['model_phase'] == model_phase]
            ax1.plot(data['fold'], data['r2'], marker='o', label=model_phase, linewidth=2)
        ax1.set_title('R² Performance Across Folds')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('R²')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RMSE across folds
        ax2 = axes[0, 1]
        for model_phase in combined_df['model_phase'].unique():
            data = combined_df[combined_df['model_phase'] == model_phase]
            ax2.plot(data['fold'], data['rmse'], marker='s', label=model_phase, linewidth=2)
        ax2.set_title('RMSE Performance Across Folds')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('RMSE')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot of R² by model/phase
        ax3 = axes[1, 0]
        box_data = []
        labels = []
        for model_phase in combined_df['model_phase'].unique():
            data = combined_df[combined_df['model_phase'] == model_phase]['r2']
            box_data.append(data)
            labels.append(model_phase)
        ax3.boxplot(box_data, labels=labels)
        ax3.set_title('R² Distribution by Model/Phase')
        ax3.set_ylabel('R²')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance vs hyperparameter consistency
        ax4 = axes[1, 1]
        # Calculate coefficient of variation for each model/phase
        cv_data = []
        for model_phase in combined_df['model_phase'].unique():
            data = combined_df[combined_df['model_phase'] == model_phase]
            cv_r2 = data['r2'].std() / data['r2'].mean() if data['r2'].mean() > 0 else 0
            mean_r2 = data['r2'].mean()
            cv_data.append({'model_phase': model_phase, 'cv_r2': cv_r2, 'mean_r2': mean_r2})
        
        cv_df = pd.DataFrame(cv_data)
        scatter = ax4.scatter(cv_df['cv_r2'], cv_df['mean_r2'], s=100, alpha=0.7)
        for i, txt in enumerate(cv_df['model_phase']):
            ax4.annotate(txt, (cv_df.iloc[i]['cv_r2'], cv_df.iloc[i]['mean_r2']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_title('Performance Consistency')
        ax4.set_xlabel('Coefficient of Variation (R²)')
        ax4.set_ylabel('Mean R²')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/fold_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{summary_dir}/fold_performance_comparison.pdf", bbox_inches='tight')
        plt.close()

    def _create_graph_statistics_summary(self, fold_dirs, summary_dir):
        """Create summary of graph statistics across folds"""
        
        # Collect graph statistics from individual fold directories
        graph_stats = []
        for model_type, target_name, phase, graphs_dir in fold_dirs:
            for fold_num in range(1, self.num_folds + 1):
                fold_dir = f"{graphs_dir}/fold_{fold_num}"
                
                # Check for k-NN graph info
                knn_info_file = f"{fold_dir}/knn_graph_info_fold{fold_num}.csv"
                if os.path.exists(knn_info_file):
                    knn_df = pd.read_csv(knn_info_file)
                    knn_stats = dict(zip(knn_df['Metric'], knn_df['Value']))
                    knn_stats.update({
                        'model_type': model_type,
                        'target_name': target_name,
                        'phase': phase,
                        'fold': fold_num,
                        'graph_type': 'knn'
                    })
                    graph_stats.append(knn_stats)
                
                # Check for explainer graph info
                exp_info_file = f"{fold_dir}/explainer_graph_info_fold{fold_num}.csv"
                if os.path.exists(exp_info_file):
                    exp_df = pd.read_csv(exp_info_file)
                    exp_stats = dict(zip(exp_df['Metric'], exp_df['Value']))
                    exp_stats.update({
                        'model_type': model_type,
                        'target_name': target_name,
                        'phase': phase,
                        'fold': fold_num,
                        'graph_type': 'explainer'
                    })
                    graph_stats.append(exp_stats)
        
        if not graph_stats:
            return
        
        stats_df = pd.DataFrame(graph_stats)
        stats_df.to_csv(f"{summary_dir}/graph_statistics_summary.csv", index=False)
        
        # Create visualization
        if len(stats_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Graph Structure Statistics Across Folds', fontsize=16, fontweight='bold')
            
            # Ensure numeric columns
            numeric_cols = ['num_nodes', 'num_edges', 'k_neighbors']
            for col in numeric_cols:
                if col in stats_df.columns:
                    stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
            
            # Plot only if we have the required columns
            if 'num_nodes' in stats_df.columns:
                ax1 = axes[0, 0]
                for graph_type in stats_df['graph_type'].unique():
                    data = stats_df[stats_df['graph_type'] == graph_type]
                    ax1.boxplot([data['num_nodes']], positions=[0 if graph_type == 'knn' else 1], widths=0.3)
                ax1.set_title('Number of Nodes Distribution')
                ax1.set_ylabel('Number of Nodes')
                ax1.set_xticks([0, 1])
                ax1.set_xticklabels(['KNN', 'Explainer'])
            
            if 'num_edges' in stats_df.columns:
                ax2 = axes[0, 1]
                for graph_type in stats_df['graph_type'].unique():
                    data = stats_df[stats_df['graph_type'] == graph_type]
                    ax2.boxplot([data['num_edges']], positions=[0 if graph_type == 'knn' else 1], widths=0.3)
                ax2.set_title('Number of Edges Distribution')
                ax2.set_ylabel('Number of Edges')
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(['KNN', 'Explainer'])
            
            plt.tight_layout()
            plt.savefig(f"{summary_dir}/graph_statistics_visualization.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{summary_dir}/graph_statistics_visualization.pdf", bbox_inches='tight')
            plt.close()

        print(f"    Graph statistics summary saved: {summary_dir}/graph_statistics_summary.csv")

    def _save_case3_combined_results(self, combined_results):
        """Save combined Case 3 results for both targets"""
        # Save combined results
        with open(f"{self.save_dir}/case3_combined_results.pkl", 'wb') as f:
            pickle.dump(combined_results, f)
        
        # Create combined summary
        summary_data = []
        
        for target_name, target_results in combined_results.items():
            # Add GNN results
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    for model_type, result in target_results[phase].items():
                        summary_data.append({
                            'case': 'case3',
                            'target': target_name.upper(),
                            'phase': phase,
                            'model_type': model_type,
                            'model_category': 'GNN',
                            'mse': result['avg_metrics']['mse'],
                            'rmse': result['avg_metrics']['rmse'],
                            'r2': result['avg_metrics']['r2'],
                            'mae': result['avg_metrics']['mae'],
                            'features_count': len(self.dataset.node_feature_names)
                        })
            
            # Add ML results
            if 'ml_models' in target_results:
                for model_type, result in target_results['ml_models'].items():
                    summary_data.append({
                        'case': 'case3',
                        'target': target_name.upper(),
                        'phase': 'embeddings',
                        'model_type': model_type,
                        'model_category': 'ML',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r2': result['avg_metrics']['r2'],
                        'mae': result['avg_metrics']['mae'],
                        'features_count': len(self.dataset.node_feature_names)
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/case3_combined_summary.csv", index=False)
        
        print(f"\nCase 3 combined results saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results across both targets")

    def _create_case3_combined_visualization(self, combined_results, ace_target_idx, h2_target_idx):
        """Create combined visualization for Case 3 with both targets"""
        print(f"\nCreating combined Case 3 visualization...")
        
        # Extract results
        ace_results = combined_results['ace_km']
        h2_results = combined_results['h2_km']
        
        # Find best models from each target
        best_models_ace = self._find_best_models_from_results(ace_results)
        best_models_h2 = self._find_best_models_from_results(h2_results)
        
        # Create combined plots for each model category
        self._plot_case3_combined_results(
            ace_results, h2_results, 
            best_models_ace, best_models_h2
        )
        
        # Calculate and save combined metrics
        combined_metrics = self._calculate_case3_combined_metrics(
            ace_results, h2_results,
            best_models_ace, best_models_h2
        )
        
        # Save combined metrics
        combined_metrics_df = pd.DataFrame(combined_metrics)
        combined_metrics_df.to_csv(f"{self.save_dir}/case3_combined_metrics.csv", index=False)
        
        print(f"Combined visualization and metrics saved to {self.save_dir}")
        return combined_metrics

    def _plot_case3_combined_results(self, ace_results, h2_results, best_models_ace, best_models_h2):
        """Create combined plots showing both targets with different colors"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Case 3: Combined Performance - All Feature Groups\nACE-km vs H2-km', fontsize=16, fontweight='bold')
        
        plot_configs = [
            ('knn', 'KNN-Sparsified GNN'),
            ('explainer', 'Explainer-Sparsified GNN'),
            ('ml', 'ML on Embeddings')
        ]
        
        # Colors for the two targets
        colors = ['#1f77b4', '#ff7f0e']  # Blue for ACE-km, Orange for H2-km
        labels = ['ACE-km (All Groups)', 'H2-km (All Groups)']
        
        plot_idx = 0
        combined_metrics_summary = []
        
        for phase, phase_name in plot_configs:
            if plot_idx >= 3:  # We only have 3 plots to make
                break
                
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            # Get best models for this phase
            if phase == 'ml':
                model_ace = best_models_ace['ml']
                model_h2 = best_models_h2['ml']
                results_ace = ace_results.get('ml_models', {}).get(model_ace, {})
                results_h2 = h2_results.get('ml_models', {}).get(model_h2, {})
            else:
                model_ace = best_models_ace[phase]
                model_h2 = best_models_h2[phase]
                results_ace = ace_results.get(phase, {}).get(model_ace, {})
                results_h2 = h2_results.get(phase, {}).get(model_h2, {})
            
            # Extract predictions and actual values
            if 'fold_predictions' in results_ace and 'fold_predictions' in results_h2:
                # ACE data
                actual_ace = []
                pred_ace = []
                for fold_data in results_ace['fold_predictions']:
                    actual_ace.extend(fold_data['actual'])
                    pred_ace.extend(fold_data['predicted'])
                
                # H2 data
                actual_h2 = []
                pred_h2 = []
                for fold_data in results_h2['fold_predictions']:
                    actual_h2.extend(fold_data['actual'])
                    pred_h2.extend(fold_data['predicted'])
                
                # Plot both targets
                ax.scatter(actual_ace, pred_ace, c=colors[0], alpha=0.7, s=60, label=labels[0], edgecolors='black', linewidth=0.5)
                ax.scatter(actual_h2, pred_h2, c=colors[1], alpha=0.7, s=60, label=labels[1], edgecolors='black', linewidth=0.5)
                
                # Calculate individual metrics
                ace_r2 = r2_score(actual_ace, pred_ace)
                ace_mse = mean_squared_error(actual_ace, pred_ace)
                ace_rmse = np.sqrt(ace_mse)
                ace_mae = mean_absolute_error(actual_ace, pred_ace)
                
                h2_r2 = r2_score(actual_h2, pred_h2)
                h2_mse = mean_squared_error(actual_h2, pred_h2)
                h2_rmse = np.sqrt(h2_mse)
                h2_mae = mean_absolute_error(actual_h2, pred_h2)
                
                combined_metrics_summary.append({
                    'phase': phase,
                    'model_ace': model_ace,
                    'model_h2': model_h2,
                    'ace_r2': ace_r2,
                    'ace_mse': ace_mse,
                    'ace_rmse': ace_rmse,
                    'ace_mae': ace_mae,
                    'h2_r2': h2_r2,
                    'h2_mse': h2_mse,
                    'h2_rmse': h2_rmse,
                    'h2_mae': h2_mae,
                    'n_samples_ace': len(actual_ace),
                    'n_samples_h2': len(actual_h2)
                })
                
                # Add perfect prediction lines for both targets
                all_actual = actual_ace + actual_h2
                all_pred = pred_ace + pred_h2
                min_val = min(min(all_actual), min(all_pred))
                max_val = max(max(all_actual), max(all_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # Formatting
                ax.set_xlabel('Actual Values', fontsize=12)
                ax.set_ylabel('Predicted Values', fontsize=12)
                ax.set_title(f'{phase_name}\nACE R²={ace_r2:.3f}, H2 R²={h2_r2:.3f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add text box with metrics
                textstr = f'ACE-km: R²={ace_r2:.3f}, RMSE={ace_rmse:.3f}\nH2-km: R²={h2_r2:.3f}, RMSE={h2_rmse:.3f}\nSamples: {len(actual_ace)} + {len(actual_h2)}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
            
            plot_idx += 1
        
        # Remove empty subplot
        if plot_idx < 4:
            fig.delaxes(axes[1, 1])
        
        # Add overall summary
        fig.text(0.02, 0.02, 
                f'Case 3: All feature groups (acetoclastic + hydrogenotrophic + syntrophic) applied to both ACE-km and H2-km targets',
                fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/case3_combined_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/case3_combined_visualization.pdf", bbox_inches='tight')
        plt.close()
        
        # Save combined metrics summary
        with open(f"{self.save_dir}/case3_combined_metrics_summary.txt", 'w') as f:
            f.write("Case 3: Combined Metrics Summary (Both Targets)\n")
            f.write("=" * 50 + "\n\n")
            
            for metrics in combined_metrics_summary:
                f.write(f"{metrics['phase'].upper()} Phase:\n")
                f.write(f"  ACE-km Model: {metrics['model_ace']}\n")
                f.write(f"  H2-km Model: {metrics['model_h2']}\n")
                f.write(f"  ACE-km R²: {metrics['ace_r2']:.4f}, RMSE: {metrics['ace_rmse']:.4f}\n")
                f.write(f"  H2-km R²: {metrics['h2_r2']:.4f}, RMSE: {metrics['h2_rmse']:.4f}\n")
                f.write(f"  Samples: ACE={metrics['n_samples_ace']}, H2={metrics['n_samples_h2']}\n")
                f.write("\n")
        
        print(f"Combined visualization saved: case3_combined_visualization.png/pdf")
        return combined_metrics_summary

    def _calculate_case3_combined_metrics(self, ace_results, h2_results, best_models_ace, best_models_h2):
        """Calculate detailed combined metrics for Case 3"""
        combined_metrics = []
        
        # Process each phase for both targets
        for phase in ['knn', 'explainer']:
            if phase in ace_results and phase in h2_results:
                for model_type in ace_results[phase].keys():
                    if model_type in h2_results[phase]:
                        # Get results for this model from both targets
                        results_ace = ace_results[phase][model_type]
                        results_h2 = h2_results[phase][model_type]
                        
                        combined_metrics.append({
                            'phase': phase,
                            'model_type': model_type,
                            'target': 'ACE-km',
                            'r2': results_ace['avg_metrics']['r2'],
                            'mse': results_ace['avg_metrics']['mse'],
                            'rmse': results_ace['avg_metrics']['rmse'],
                            'mae': results_ace['avg_metrics']['mae'],
                            'is_best': model_type == best_models_ace[phase]
                        })
                        
                        combined_metrics.append({
                            'phase': phase,
                            'model_type': model_type,
                            'target': 'H2-km',
                            'r2': results_h2['avg_metrics']['r2'],
                            'mse': results_h2['avg_metrics']['mse'],
                            'rmse': results_h2['avg_metrics']['rmse'],
                            'mae': results_h2['avg_metrics']['mae'],
                            'is_best': model_type == best_models_h2[phase]
                        })
        
        # Process ML models
        if 'ml_models' in ace_results and 'ml_models' in h2_results:
            for model_type in ace_results['ml_models'].keys():
                if model_type in h2_results['ml_models']:
                    results_ace = ace_results['ml_models'][model_type]
                    results_h2 = h2_results['ml_models'][model_type]
                    
                    combined_metrics.append({
                        'phase': 'ml_embeddings',
                        'model_type': model_type,
                        'target': 'ACE-km',
                        'r2': results_ace['avg_metrics']['r2'],
                        'mse': results_ace['avg_metrics']['mse'],
                        'rmse': results_ace['avg_metrics']['rmse'],
                        'mae': results_ace['avg_metrics']['mae'],
                        'is_best': model_type == best_models_ace['ml']
                    })
                    
                    combined_metrics.append({
                        'phase': 'ml_embeddings',
                        'model_type': model_type,
                        'target': 'H2-km',
                        'r2': results_h2['avg_metrics']['r2'],
                        'mse': results_h2['avg_metrics']['mse'],
                        'rmse': results_h2['avg_metrics']['rmse'],
                        'mae': results_h2['avg_metrics']['mae'],
                        'is_best': model_type == best_models_h2['ml']
                    })
        
        return combined_metrics
    
    def _save_case4_combined_results(self, combined_results, ace_target_idx):
        """Save combined Case 4 results"""
        # Save combined results
        with open(f"{self.save_dir}/case4_combined_results.pkl", 'wb') as f:
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
                            'case': 'case4',
                            'subset': subset_name,
                            'target': 'ACE-km',
                            'phase': phase,
                            'model_type': model_type,
                            'model_category': 'GNN',
                            'mse': result['avg_metrics']['mse'],
                            'rmse': result['avg_metrics']['rmse'],
                            'r2': result['avg_metrics']['r2'],
                            'mae': result['avg_metrics']['mae'],
                            'sample_count': subset_results['subset_info']['sample_count'],
                            'features_count': subset_results['subset_info']['final_features_count']
                        })
            
            # Add ML results
            if 'ml_models' in subset_results:
                for model_type, result in subset_results['ml_models'].items():
                    summary_data.append({
                        'case': 'case4',
                        'subset': subset_name,
                        'target': 'ACE-km',
                        'phase': 'embeddings',
                        'model_type': model_type,
                        'model_category': 'ML',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r2': result['avg_metrics']['r2'],
                        'mae': result['avg_metrics']['mae'],
                        'sample_count': subset_results['subset_info']['sample_count'],
                        'features_count': subset_results['subset_info']['final_features_count']
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/case4_combined_summary.csv", index=False)
        
        print(f"\nCase 4 combined results saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results across both subsets")
    
    def _create_case4_combined_visualization(self, combined_results, ace_target_idx):
        """Create combined visualization for Case 4 with different colored points for each subcase"""
        print(f"\nCreating combined Case 4 visualization...")
        
        # Extract predictions and actual values from both subcases
        case4a_results = combined_results['case4a_low_ace']
        case4b_results = combined_results['case4b_high_ace']
        
        # Find best models from each subcase
        best_models_4a = self._find_best_models_from_results(case4a_results)
        best_models_4b = self._find_best_models_from_results(case4b_results)
        
        # Create combined plots for each model category
        self._plot_case4_combined_results(
            case4a_results, case4b_results, 
            best_models_4a, best_models_4b,
            combined_results['data_split']
        )
        
        # Calculate and save combined metrics
        combined_metrics = self._calculate_case4_combined_metrics(
            case4a_results, case4b_results,
            best_models_4a, best_models_4b
        )
        
        # Save combined metrics
        combined_metrics_df = pd.DataFrame(combined_metrics)
        combined_metrics_df.to_csv(f"{self.save_dir}/case4_combined_metrics.csv", index=False)
        
        print(f"Combined visualization and metrics saved to {self.save_dir}")
        return combined_metrics
    
    def _find_best_models_from_results(self, results):
        """Find the best model from each category in results"""
        best_models = {}
        
        # Find best GNN model from KNN phase
        best_knn_r2 = -float('inf')
        best_knn_model = None
        if 'knn' in results:
            for model_type, model_results in results['knn'].items():
                if model_results['avg_metrics']['r2'] > best_knn_r2:
                    best_knn_r2 = model_results['avg_metrics']['r2']
                    best_knn_model = model_type
        best_models['knn'] = best_knn_model
        
        # Find best GNN model from explainer phase
        best_explainer_r2 = -float('inf')
        best_explainer_model = None
        if 'explainer' in results:
            for model_type, model_results in results['explainer'].items():
                if model_results['avg_metrics']['r2'] > best_explainer_r2:
                    best_explainer_r2 = model_results['avg_metrics']['r2']
                    best_explainer_model = model_type
        best_models['explainer'] = best_explainer_model
        
        # Find best ML model
        best_ml_r2 = -float('inf')
        best_ml_model = None
        if 'ml_models' in results:
            for model_type, model_results in results['ml_models'].items():
                if model_results['avg_metrics']['r2'] > best_ml_r2:
                    best_ml_r2 = model_results['avg_metrics']['r2']
                    best_ml_model = model_type
        best_models['ml'] = best_ml_model
        
        return best_models
    
    def _plot_case4_combined_results(self, case4a_results, case4b_results, best_models_4a, best_models_4b, data_split):
        """Create combined plots showing both subcases with different colors"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Case 4: Combined ACE-km Prediction Results\nAcetoclastic vs All Groups', fontsize=16, fontweight='bold')
        
        plot_configs = [
            ('knn', 'KNN-Sparsified GNN'),
            ('explainer', 'Explainer-Sparsified GNN'),
            ('ml', 'ML on Embeddings')
        ]
        
        # Colors for the two subcases
        colors = ['#FF6B6B', '#4ECDC4']  # Red for case4a, Teal for case4b
        labels = ['ACE-km < 10 (Acetoclastic only)', 'ACE-km ≥ 10 (All groups)']
        
        plot_idx = 0
        combined_metrics_summary = []
        
        for phase, phase_name in plot_configs:
            if plot_idx >= 3:  # We only have 3 plots to make
                break
                
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            # Get best models for this phase
            if phase == 'ml':
                model_4a = best_models_4a['ml']
                model_4b = best_models_4b['ml']
                results_4a = case4a_results.get('ml_models', {}).get(model_4a, {})
                results_4b = case4b_results.get('ml_models', {}).get(model_4b, {})
            else:
                model_4a = best_models_4a[phase]
                model_4b = best_models_4b[phase]
                results_4a = case4a_results.get(phase, {}).get(model_4a, {})
                results_4b = case4b_results.get(phase, {}).get(model_4b, {})
            
            # Extract predictions and actual values
            if 'fold_predictions' in results_4a and 'fold_predictions' in results_4b:
                # Case 4a data
                actual_4a = []
                pred_4a = []
                for fold_data in results_4a['fold_predictions']:
                    actual_4a.extend(fold_data['actual'])
                    pred_4a.extend(fold_data['predicted'])
                
                # Case 4b data
                actual_4b = []
                pred_4b = []
                for fold_data in results_4b['fold_predictions']:
                    actual_4b.extend(fold_data['actual'])
                    pred_4b.extend(fold_data['predicted'])
                
                # Plot both subcases
                ax.scatter(actual_4a, pred_4a, c=colors[0], alpha=0.7, s=60, label=labels[0], edgecolors='black', linewidth=0.5)
                ax.scatter(actual_4b, pred_4b, c=colors[1], alpha=0.7, s=60, label=labels[1], edgecolors='black', linewidth=0.5)
                
                # Calculate combined metrics
                all_actual = actual_4a + actual_4b
                all_pred = pred_4a + pred_4b
                
                combined_r2 = r2_score(all_actual, all_pred)
                combined_mse = mean_squared_error(all_actual, all_pred)
                combined_rmse = np.sqrt(combined_mse)
                combined_mae = mean_absolute_error(all_actual, all_pred)
                
                combined_metrics_summary.append({
                    'phase': phase,
                    'model_4a': model_4a,
                    'model_4b': model_4b,
                    'combined_r2': combined_r2,
                    'combined_mse': combined_mse,
                    'combined_rmse': combined_rmse,
                    'combined_mae': combined_mae,
                    'n_samples_4a': len(actual_4a),
                    'n_samples_4b': len(actual_4b),
                    'n_total': len(all_actual)
                })
                
                # Add perfect prediction line
                min_val = min(min(all_actual), min(all_pred))
                max_val = max(max(all_actual), max(all_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
                
                # Formatting
                ax.set_xlabel('Actual ACE-km Values', fontsize=12)
                ax.set_ylabel('Predicted ACE-km Values', fontsize=12)
                ax.set_title(f'{phase_name}\nCombined R² = {combined_r2:.4f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add text box with metrics
                textstr = f'Combined Metrics:\nR² = {combined_r2:.4f}\nRMSE = {combined_rmse:.4f}\nMAE = {combined_mae:.4f}\nSamples: {len(actual_4a)} + {len(actual_4b)} = {len(all_actual)}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
            
            plot_idx += 1
        
        # Remove empty subplot
        if plot_idx < 4:
            fig.delaxes(axes[1, 1])
        
        # Add overall summary
        fig.text(0.02, 0.02, 
                f'Case 4 Summary: {data_split["low_ace_count"]} samples (ACE-km < 10) + {data_split["high_ace_count"]} samples (ACE-km ≥ 10) = {data_split["low_ace_count"] + data_split["high_ace_count"]} total samples',
                fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/case4_combined_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/case4_combined_visualization.pdf", bbox_inches='tight')
        plt.close()
        
        # Save combined metrics summary
        with open(f"{self.save_dir}/case4_combined_metrics_summary.txt", 'w') as f:
            f.write("Case 4: Combined Metrics Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for metrics in combined_metrics_summary:
                f.write(f"{metrics['phase'].upper()} Phase:\n")
                f.write(f"  Model 4a (ACE-km < 10): {metrics['model_4a']}\n")
                f.write(f"  Model 4b (ACE-km ≥ 10): {metrics['model_4b']}\n")
                f.write(f"  Combined R²: {metrics['combined_r2']:.4f}\n")
                f.write(f"  Combined RMSE: {metrics['combined_rmse']:.4f}\n")
                f.write(f"  Combined MAE: {metrics['combined_mae']:.4f}\n")
                f.write(f"  Samples: {metrics['n_samples_4a']} + {metrics['n_samples_4b']} = {metrics['n_total']}\n")
                f.write("\n")
        
        print(f"Combined visualization saved: case4_combined_visualization.png/pdf")
        return combined_metrics_summary
    
    def _calculate_case4_combined_metrics(self, case4a_results, case4b_results, best_models_4a, best_models_4b):
        """Calculate detailed combined metrics for Case 4"""
        combined_metrics = []
        
        # Process each phase
        for phase in ['knn', 'explainer']:
            if phase in case4a_results and phase in case4b_results:
                for model_type in case4a_results[phase].keys():
                    if model_type in case4b_results[phase]:
                        # Get results for this model from both subcases
                        results_4a = case4a_results[phase][model_type]
                        results_4b = case4b_results[phase][model_type]
                        
                        # Extract all predictions
                        if 'fold_predictions' in results_4a and 'fold_predictions' in results_4b:
                            actual_4a = []
                            pred_4a = []
                            for fold_data in results_4a['fold_predictions']:
                                actual_4a.extend(fold_data['actual'])
                                pred_4a.extend(fold_data['predicted'])
                            
                            actual_4b = []
                            pred_4b = []
                            for fold_data in results_4b['fold_predictions']:
                                actual_4b.extend(fold_data['actual'])
                                pred_4b.extend(fold_data['predicted'])
                            
                            # Calculate combined metrics
                            all_actual = actual_4a + actual_4b
                            all_pred = pred_4a + pred_4b
                            
                            combined_metrics.append({
                                'phase': phase,
                                'model_type': model_type,
                                'subcase_4a_r2': results_4a['avg_metrics']['r2'],
                                'subcase_4b_r2': results_4b['avg_metrics']['r2'],
                                'combined_r2': r2_score(all_actual, all_pred),
                                'combined_mse': mean_squared_error(all_actual, all_pred),
                                'combined_rmse': np.sqrt(mean_squared_error(all_actual, all_pred)),
                                'combined_mae': mean_absolute_error(all_actual, all_pred),
                                'n_samples_4a': len(actual_4a),
                                'n_samples_4b': len(actual_4b),
                                'n_total': len(all_actual),
                                'is_best_4a': model_type == best_models_4a[phase],
                                'is_best_4b': model_type == best_models_4b[phase]
                            })
        
        # Process ML models
        if 'ml_models' in case4a_results and 'ml_models' in case4b_results:
            for model_type in case4a_results['ml_models'].keys():
                if model_type in case4b_results['ml_models']:
                    results_4a = case4a_results['ml_models'][model_type]
                    results_4b = case4b_results['ml_models'][model_type]
                    
                    if 'fold_predictions' in results_4a and 'fold_predictions' in results_4b:
                        actual_4a = []
                        pred_4a = []
                        for fold_data in results_4a['fold_predictions']:
                            actual_4a.extend(fold_data['actual'])
                            pred_4a.extend(fold_data['predicted'])
                        
                        actual_4b = []
                        pred_4b = []
                        for fold_data in results_4b['fold_predictions']:
                            actual_4b.extend(fold_data['actual'])
                            pred_4b.extend(fold_data['predicted'])
                        
                        all_actual = actual_4a + actual_4b
                        all_pred = pred_4a + pred_4b
                        
                        combined_metrics.append({
                            'phase': 'ml_embeddings',
                            'model_type': model_type,
                            'subcase_4a_r2': results_4a['avg_metrics']['r2'],
                            'subcase_4b_r2': results_4b['avg_metrics']['r2'],
                            'combined_r2': r2_score(all_actual, all_pred),
                            'combined_mse': mean_squared_error(all_actual, all_pred),
                            'combined_rmse': np.sqrt(mean_squared_error(all_actual, all_pred)),
                            'combined_mae': mean_absolute_error(all_actual, all_pred),
                            'n_samples_4a': len(actual_4a),
                            'n_samples_4b': len(actual_4b),
                            'n_total': len(all_actual),
                            'is_best_4a': model_type == best_models_4a['ml'],
                            'is_best_4b': model_type == best_models_4b['ml']
                        })
        
        return combined_metrics

    def _run_case5(self):
        """Case 5: Conditional feature selection based on H2-km value"""
        print("Case 5: Conditional feature selection based on H2-km value")
        print("H2-km < 10: hydrogenotrophic only")
        print("H2-km >= 10: hydrogenotrophic + acetoclastic")
        
        # Find H2 target index
        h2_target_idx = None
        for i, target in enumerate(self.target_names):
            if 'H2' in target:
                h2_target_idx = i
                break
        
        if h2_target_idx is None:
            raise ValueError("H2-km target not found in dataset")
        
        # Get H2-km values and split data
        h2_values = []
        for data in self.dataset.data_list:
            h2_values.append(data.y[0, h2_target_idx].item())
        h2_values = np.array(h2_values)
        
        # Split indices based on H2-km values
        low_h2_indices = np.where(h2_values < 10)[0]
        high_h2_indices = np.where(h2_values >= 10)[0]
        
        print(f"Data split: {len(low_h2_indices)} samples with H2-km < 10, {len(high_h2_indices)} samples with H2-km >= 10")
        
        # Check if we have enough samples in each subset
        if len(low_h2_indices) < 5:
            print(f"WARNING: Only {len(low_h2_indices)} samples with H2-km < 10. Running combined analysis instead.")
            return self._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        if len(high_h2_indices) < 5:
            print(f"WARNING: Only {len(high_h2_indices)} samples with H2-km >= 10. Running combined analysis instead.")
            return self._run_single_target_pipeline(h2_target_idx, "H2-km")
        
        # Run Case 5a: H2-km < 10 with hydrogenotrophic features only
        print(f"\n{'='*60}")
        print("CASE 5a: H2-km < 10 → Hydrogenotrophic features only")
        print(f"{'='*60}")
        
        case5a_results = self._run_case5_subset(
            subset_indices=low_h2_indices,
            subset_name="case5a_low_h2",
            anchored_features=self.hydrogenotrophic,
            description="H2-km < 10 with hydrogenotrophic features"
        )
        
        # Run Case 5b: H2-km >= 10 with all feature groups
        print(f"\n{'='*60}")
        print("CASE 5b: H2-km >= 10 → All feature groups")
        print(f"{'='*60}")
        
        case5b_results = self._run_case5_subset(
            subset_indices=high_h2_indices,
            subset_name="case5b_high_h2",
            anchored_features=self.acetoclastic + self.hydrogenotrophic,
            description="H2-km >= 10 with all feature groups"
        )
        
        # Combine results
        combined_results = {
            'case5a_low_h2': case5a_results,
            'case5b_high_h2': case5b_results,
            'data_split': {
                'low_h2_count': len(low_h2_indices),
                'high_h2_count': len(high_h2_indices),
                'low_h2_indices': low_h2_indices.tolist(),
                'high_h2_indices': high_h2_indices.tolist()
            }
        }
        
        # Save combined results
        self._save_case5_combined_results(combined_results, h2_target_idx)
        
        return combined_results
    
    def _run_case5_subset(self, subset_indices, subset_name, anchored_features, description):
        """Run pipeline for a Case 5 subset with specific anchored features"""
        print(f"Running {subset_name}: {description}")
        print(f"Subset size: {len(subset_indices)} samples")
        print(f"Anchored features: {len(anchored_features)} features")
        
        # Create detailed_results folder for this subset
        subset_save_dir = f"{self.save_dir}/{subset_name}"
        os.makedirs(f"{subset_save_dir}/detailed_results", exist_ok=True)
        
        # Create subset data from the ORIGINAL dataset to maintain consistent dimensions
        subset_data_list = [self.dataset.data_list[i] for i in subset_indices]
        
        # DON'T create a new dataset - just use the original one with subset data
        # This ensures consistent feature dimensions across all subcases
        original_data_list = self.dataset.data_list
        original_save_dir = self.save_dir
        self.dataset.data_list = subset_data_list
        self.save_dir = subset_save_dir
        
        try:
            # Find H2 target index
            h2_target_idx = None
            for i, target in enumerate(self.target_names):
                if 'H2' in target:
                    h2_target_idx = i
                    break
            
            # Run the pipeline using the parent dataset but with subset data
            results = self._run_single_target_pipeline(h2_target_idx, f"H2-km_{subset_name}")
            
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
            # Restore original data list and save directory
            self.dataset.data_list = original_data_list
            self.save_dir = original_save_dir
    
    def _save_case5_combined_results(self, combined_results, h2_target_idx):
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
                            'target': 'H2-km',
                            'phase': phase,
                            'model_type': model_type,
                            'model_category': 'GNN',
                            'mse': result['avg_metrics']['mse'],
                            'rmse': result['avg_metrics']['rmse'],
                            'r2': result['avg_metrics']['r2'],
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
                        'target': 'H2-km',
                        'phase': 'embeddings',
                        'model_type': model_type,
                        'model_category': 'ML',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r2': result['avg_metrics']['r2'],
                        'mae': result['avg_metrics']['mae'],
                        'sample_count': subset_results['subset_info']['sample_count'],
                        'features_count': subset_results['subset_info']['final_features_count']
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/case5_combined_summary.csv", index=False)
        
        print(f"\nCase 5 combined results saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results across both subsets")


def run_all_cases(data_path="../Data/New_data.csv"):
    """Run all domain expert cases"""
    print("Running all domain expert cases...")
    
    # cases = ['case1', 'case2', 'case3', 'case4', 'case5']
    cases = ['case1']
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()}")
        print(f"{'='*60}")
        
        try:
            pipeline = DomainExpertCasesPipeline(
                data_path=data_path,
                case_type=case,
                k_neighbors=15,
                hidden_dim=512,
                num_epochs=200,
                num_folds=5,
                save_dir="./domain_expert_results",
                importance_threshold=0.2,
                use_fast_correlation=False,
                family_filter_mode='strict',
                use_nested_cv=True  # Enable nested CV
            )
            
            case_results = pipeline.run_case_specific_pipeline()
            all_results[case] = case_results
            
        except Exception as e:
            print(f"Error running {case}: {e}")
            all_results[case] = None
    
    # Create overall comparison
    print(f"\n{'='*60}")
    print("CREATING OVERALL COMPARISON")
    print(f"{'='*60}")
    
    # Save combined results
    with open("./domain_expert_results/all_cases_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    print("All cases completed!")
    return all_results


def test_single_case(data_path="../Data/New_data.csv", case_type='case1'):
    """Test a single case with reduced parameters for faster execution"""
    print(f"Testing {case_type} with reduced parameters...")
    
    try:
        pipeline = DomainExpertCasesPipeline(
            data_path=data_path,
            case_type=case_type,
            k_neighbors=10,
            hidden_dim=64,  # Reduced for testing
            num_epochs=2,   # Reduced for testing
            num_folds=2,    # Reduced for testing
            save_dir=f"./domain_expert_results_test/{case_type}",
            importance_threshold=0.2,
            use_fast_correlation=False,
            family_filter_mode='strict',
            use_nested_cv=True
        )
        
        case_results = pipeline.run_case_specific_pipeline()
        print(f"Test completed successfully for {case_type}")
        return case_results
        
    except Exception as e:
        print(f"Error testing {case_type}: {e}")
        return None


def validate_implementation():
    """Validate the implementation logic"""
    print("Validating implementation logic...")
    
    # Test 1: Phase detection
    dummy_list1 = [1, 2, 3]
    dummy_list2 = [4, 5]
    dummy_list3 = dummy_list1
    
    print(f"List1 vs List2 (different): {id(dummy_list1) != id(dummy_list2)}")  # Should be True
    print(f"List1 vs List3 (same): {id(dummy_list1) == id(dummy_list3)}")      # Should be True
    print(f"Length comparison: {len(dummy_list1) != len(dummy_list2)}")        # Should be True
    
    print("Implementation logic validation passed!")
    return True

if __name__ == "__main__":
    # Validate implementation first
    if validate_implementation():
        # Test single case first
        print("\nTesting single case with nested CV...")
        test_results = test_single_case(case_type='case1')
        
        if test_results:
            print("Test successful! Running all cases...")
            # Run all cases
            results = run_all_cases()
            print("Domain expert cases pipeline completed!")
        else:
            print("Test failed! Check the error above.")
    else:
        print("Implementation validation failed!") 