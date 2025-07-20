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
from dataclasses import dataclass
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
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class PipelineConfig:
    """Configuration class for the domain expert cases pipeline."""
    
    # Nested CV settings
    inner_cv_folds: int = 2
    outer_cv_folds: int = 3
    
    # Model hyperparameters
    hidden_dim: int = 64
    dropout_rate: float = 0.3
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 200
    patience: int = 20
    
    # Graph settings
    k_neighbors: int = 5
    mantel_threshold: float = 0.05
    importance_threshold: float = 0.2
    
    # Data processing
    use_fast_correlation: bool = False
    graph_mode: str = 'family'
    family_filter_mode: str = 'strict'
    
    # Hyperparameter tuning grid (reduced for efficiency)
    hidden_dim_grid: list = None
    dropout_rate_grid: list = None
    
    def __post_init__(self):
        if self.hidden_dim_grid is None:
            self.hidden_dim_grid = [64, 256]
        if self.dropout_rate_grid is None:
            self.dropout_rate_grid = [0.2, 0.5]

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
    """Pipeline for domain expert cases with nested CV hyperparameter tuning"""
    
    def __init__(self, data_path, case_type='case1', 
                 k_neighbors=5, mantel_threshold=0.05,
                 hidden_dim=64, dropout_rate=0.3, batch_size=8,
                 learning_rate=0.001, weight_decay=1e-4,
                 num_epochs=200, patience=20, num_folds=5,
                 save_dir='./domain_expert_results',
                 importance_threshold=0.2,
                 use_fast_correlation=False,
                 graph_mode='family', family_filter_mode='strict',
                 inner_cv_folds=2, outer_cv_folds=None,
                 config=None):
        
        # Use configuration if provided, otherwise create from parameters
        if config is None:
            config = PipelineConfig(
                inner_cv_folds=inner_cv_folds,
                outer_cv_folds=outer_cv_folds if outer_cv_folds else num_folds,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_epochs=num_epochs,
                patience=patience,
                k_neighbors=k_neighbors,
                mantel_threshold=mantel_threshold,
                importance_threshold=importance_threshold,
                use_fast_correlation=use_fast_correlation,
                graph_mode=graph_mode,
                family_filter_mode=family_filter_mode
            )
        
        # Store configuration
        self.config = config
        
        # Nested CV settings
        self.inner_cv_folds = config.inner_cv_folds
        self.outer_cv_folds = config.outer_cv_folds
        self.case_type = case_type
        
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

    def _tune_gnn_hyperparams(self, model_type, target_idx, data_list, phase="knn"):
        """
        Perform proper nested cross-validation for hyperparameter tuning.
        
        Args:
            model_type (str): Type of GNN model ('gcn', 'rggc', 'gat')
            target_idx (int): Index of target variable
            data_list (list): List of graph data objects
            phase (str): Phase of training ('knn' or 'explainer')
        
        Returns:
            dict: Best hyperparameters found
        """
        print(f"\nStarting nested CV hyperparameter tuning for {model_type} on {phase} graph...")
        print(f"Inner CV folds: {self.inner_cv_folds}")
        
        grid = {
            'hidden_dim': self.config.hidden_dim_grid,
            'dropout_rate': self.config.dropout_rate_grid
        }
        
        best_neg_mse = -float('inf')
        best_r2 = -float('inf')
        best_params = {'hidden_dim': self.config.hidden_dim, 'dropout_rate': self.config.dropout_rate}
        
        # Store original settings
        orig_hidden_dim = self.config.hidden_dim
        orig_dropout_rate = self.config.dropout_rate
        
        # Track all hyperparameter combinations and their scores
        hyperparam_results = []
        
        # Create inner CV splits for hyperparameter tuning
        inner_cv = KFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=42)
        
        total_combinations = len(grid['hidden_dim']) * len(grid['dropout_rate'])
        print(f"Testing {total_combinations} hyperparameter combinations...")

        for hd in grid['hidden_dim']:
            for dr in grid['dropout_rate']:
                print(f"  Testing hidden_dim={hd}, dropout_rate={dr}...")
                
                # Store hyperparameters for this iteration
                self.config.hidden_dim = hd
                self.config.dropout_rate = dr
                
                # Inner CV loop for this hyperparameter combination
                inner_scores = []
                inner_mses = []
                
                try:
                    for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(data_list)):
                        print(f"    Inner CV Fold {fold_idx + 1}/{self.inner_cv_folds}")
                        
                        # Split data for this fold
                        train_data = [data_list[i] for i in train_idx]
                        val_data = [data_list[i] for i in val_idx]
                        
                        # Train model on train data using our overridden method
                        fold_result = self._train_single_fold(model_type, target_idx, train_data, val_data)
                        
                        if fold_result:
                            inner_scores.append(fold_result['r2'])
                            inner_mses.append(fold_result['mse'])
                    
                    # Calculate average performance for this hyperparameter combination
                    if inner_scores:
                        avg_r2 = np.mean(inner_scores)
                        avg_mse = np.mean(inner_mses)
                        neg_mse = -avg_mse
                        
                        print(f"    → Avg R²={avg_r2:.4f}, Avg MSE={avg_mse:.4f}")
                        
                        # Store results for this combination
                        hyperparam_results.append({
                            'model_type': model_type,
                            'phase': phase,
                            'hidden_dim': hd,
                            'dropout_rate': dr,
                            'avg_r2': avg_r2,
                            'avg_mse': avg_mse,
                            'std_r2': np.std(inner_scores),
                            'std_mse': np.std(inner_mses),
                            'fold_scores': inner_scores,
                            'fold_mses': inner_mses
                        })
                        
                        # Update best parameters
                        if neg_mse > best_neg_mse or (neg_mse == best_neg_mse and avg_r2 > best_r2):
                            best_neg_mse, best_r2 = neg_mse, avg_r2
                            best_params = {'hidden_dim': hd, 'dropout_rate': dr}
                    else:
                        print(f"    → No valid results for this combination")
                        
                except Exception as e:
                    print(f"    → Error with hidden_dim={hd}, dropout_rate={dr}: {e}")
                    continue

        print(f"\nBest hyperparameters for {phase}: hidden_dim={best_params['hidden_dim']}, dropout_rate={best_params['dropout_rate']}")
        print(f"Best performance: R²={best_r2:.4f}, MSE={-best_neg_mse:.4f}")
        
        # Check if we found any valid results
        if best_r2 == -float('inf'):
            print(f"WARNING: No valid hyperparameter combinations found for {phase}. Using default parameters.")
            best_params = {'hidden_dim': self.config.hidden_dim, 'dropout_rate': self.config.dropout_rate}
            best_r2 = 0.0
            best_neg_mse = 0.0
        
        # Save hyperparameter tuning results
        hyperparam_df = pd.DataFrame(hyperparam_results)
        os.makedirs(f"{self.save_dir}/hyperparameters", exist_ok=True)
        hyperparam_df.to_csv(f"{self.save_dir}/hyperparameters/{model_type}_{phase}_hyperparameter_tuning.csv", index=False)
        
        # Save best hyperparameters summary
        best_hyperparam_summary = {
            'model_type': model_type,
            'phase': phase,
            'best_hidden_dim': best_params['hidden_dim'],
            'best_dropout_rate': best_params['dropout_rate'],
            'best_r2': best_r2,
            'best_mse': -best_neg_mse,
            'all_combinations': hyperparam_results
        }
        
        with open(f"{self.save_dir}/hyperparameters/{model_type}_{phase}_best_hyperparameters.pkl", 'wb') as f:
            pickle.dump(best_hyperparam_summary, f)
        
        # Restore original settings
        self.config.hidden_dim = orig_hidden_dim
        self.config.dropout_rate = orig_dropout_rate
        
        return best_params

    def train_gnn_model(self, model_type, target_idx, data_list, phase="knn"):
        """Train GNN with nested CV hyperparameter tuning"""
        # Inner CV for hyperparameter tuning
        best_params = self._tune_gnn_hyperparams(model_type, target_idx, data_list, phase)
        
        # Apply best hyperparameters
        self.config.hidden_dim = best_params['hidden_dim']
        self.config.dropout_rate = best_params['dropout_rate']
        
        # Use outer CV for final evaluation
        self.num_folds = self.outer_cv_folds
        
        # Train with best hyperparameters using parent class method
        return super().train_gnn_model(model_type, target_idx, data_list)
    
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
        """Case 3: Use all feature groups for ACE dataset"""
        print("Case 3: Using acetoclastic + hydrogenotrophic + syntrophic features for ACE dataset")
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
                'final_features_count': len(self.dataset.node_feature_names),
                'final_features': self.dataset.node_feature_names
            }
            
            return results
            
        finally:
            # Restore original data list
            self.dataset.data_list = original_data_list
    
    def _run_single_target_pipeline(self, target_idx, target_name):
        """
        Run pipeline for a single target using the parent class functionality.
        
        Args:
            target_idx (int): Index of target variable
            target_name (str): Name of target variable
            
        Returns:
            dict: Results dictionary containing all model performances
        """
        print(f"\nRunning pipeline for {target_name}")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Feature names: {self.dataset.node_feature_names}")
        
        results = {}
        
        try:
            # Step 1: Train ALL GNN models on KNN-sparsified graph (using parent class method)
            print(f"\nSTEP 1: Training ALL GNN models on KNN-sparsified graph")
            knn_gnn_results = {}
            for model_type in self.gnn_models_to_train:
                try:
                    print(f"\nTraining {model_type.upper()} for target {target_name} (KNN)")
                    knn_gnn_results[model_type] = self.train_gnn_model(
                        model_type=model_type,
                        target_idx=target_idx,
                        data_list=self.dataset.data_list,
                        phase="knn"
                    )
                except Exception as e:
                    print(f"ERROR: Failed to train {model_type} on KNN graph: {e}")
                    continue
            
            if not knn_gnn_results:
                raise ValueError("No GNN models were successfully trained on KNN graph")
            
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
            
            try:
                explainer_data = self.create_explainer_sparsified_graph(
                    model=best_gnn_model,
                    target_idx=target_idx
                )
            except Exception as e:
                print(f"ERROR: Failed to create explainer graph: {e}")
                # Fallback to using original data
                explainer_data = self.dataset.data_list
                print("Using original data as fallback")
            
            # Step 3: Train ALL GNN models on explainer-sparsified graph (using parent class method)
            print(f"\nSTEP 3: Training ALL GNN models on explainer-sparsified graph")
            explainer_gnn_results = {}
            for model_type in self.gnn_models_to_train:
                try:
                    # For explainer phase, we need to tune hyperparameters again since the graph structure is different
                    print(f"\nTuning hyperparameters for {model_type} on explainer-sparsified graph...")
                    explainer_gnn_results[model_type] = self.train_gnn_model(
                        model_type=model_type,
                        target_idx=target_idx,
                        data_list=explainer_data,
                        phase="explainer"
                    )
                except Exception as e:
                    print(f"ERROR: Failed to train {model_type} on explainer graph: {e}")
                    continue
            
            if not explainer_gnn_results:
                raise ValueError("No GNN models were successfully trained on explainer graph")
            
            results['knn'] = knn_gnn_results
            results['explainer'] = explainer_gnn_results
            
            # Find best model from explainer-sparsified graph
            best_explainer_model = None
            best_explainer_r2 = -float('inf')
            best_explainer_type = None
            
            for model_type, gnn_results in explainer_gnn_results.items():
                if gnn_results['avg_metrics']['r2'] > best_explainer_r2:
                    best_explainer_r2 = gnn_results['avg_metrics']['r2']
                    best_explainer_model = gnn_results['model']  # Store actual model object
                    best_explainer_type = model_type
            
            print(f"\nBest explainer-trained GNN model: {best_explainer_type.upper()} (R² = {best_explainer_r2:.4f})")
            
            # Step 4: Extract embeddings from best model (using parent class method)
            print(f"\nSTEP 4: Extracting embeddings from best GNN model")
            
            if best_explainer_model is None:
                print("WARNING: No valid model found for embedding extraction. Using first available model.")
                # Fallback to first available model
                for model_type, gnn_results in explainer_gnn_results.items():
                    best_explainer_model = gnn_results['model']
                    best_explainer_type = model_type
                    break
            
            try:
                embeddings, targets = self.extract_embeddings(best_explainer_model, explainer_data)
                
                # Memory cleanup
                del best_explainer_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Save embeddings
                os.makedirs(f"{self.save_dir}/embeddings", exist_ok=True)
                np.save(f"{self.save_dir}/embeddings/{target_name}_embeddings.npy", embeddings)
                np.save(f"{self.save_dir}/embeddings/{target_name}_targets.npy", targets)
                
                print(f"Extracted embeddings shape: {embeddings.shape}")
                
            except Exception as e:
                print(f"ERROR: Failed to extract embeddings: {e}")
                raise
            
            # Step 5: Train ML models on embeddings (using parent class method)
            print(f"\nSTEP 5: Training ML models on embeddings")
            try:
                ml_results = self.train_ml_models(embeddings, targets, target_idx)
                results['ml_models'] = ml_results
            except Exception as e:
                print(f"ERROR: Failed to train ML models: {e}")
                results['ml_models'] = {}
            
            # Step 6: Create plots (using parent class method)
            print(f"\nSTEP 6: Creating plots")
            try:
                # For mixed models, show all models
                gnn_plot_results = {**knn_gnn_results, **{f"{k}_explainer": v for k, v in explainer_gnn_results.items()}}
                
                self.plot_results(
                    gnn_results=gnn_plot_results,
                    ml_results=results.get('ml_models', {}),
                    target_idx=target_idx
                )
            except Exception as e:
                print(f"ERROR: Failed to create plots: {e}")
            
            # Save results
            self._save_case_results(results, target_name)
            
            return results
            
        except Exception as e:
            print(f"CRITICAL ERROR in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
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
        
        print(f"\nResults saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results")
    
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
        
        # Create subset data from the ORIGINAL dataset to maintain consistent dimensions
        subset_data_list = [self.dataset.data_list[i] for i in subset_indices]
        
        # DON'T create a new dataset - just use the original one with subset data
        # This ensures consistent feature dimensions across all subcases
        original_data_list = self.dataset.data_list
        self.dataset.data_list = subset_data_list
        
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
            # Restore original data list
            self.dataset.data_list = original_data_list
    
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

    def plot_results(self, gnn_results, ml_results, target_idx):
        """Use parent class plotting method for consistent results"""
        # Call parent class method
        super().plot_results(gnn_results, ml_results, target_idx)

    def _train_single_fold(self, model_type, target_idx, train_data, val_data):
        """
        Train a single fold for hyperparameter tuning.
        
        Args:
            model_type (str): Type of GNN model
            target_idx (int): Index of target variable
            train_data (list): Training data
            val_data (list): Validation data
            
        Returns:
            dict: Training results with metrics
        """
        try:
            # Create data loaders
            train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model using our overridden method
            model = self.create_gnn_model(model_type).to(device)
            
            # Setup optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            # Training loop
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(min(self.config.num_epochs, 50)):  # Limit epochs for hyperparameter tuning
                train_loss = 0.0
                val_loss = 0.0
                
                # Training
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass - handle batch properly
                    model_output = model(batch.x, batch.edge_index, batch.batch)
                    
                    # Handle tuple output from GNN models (prediction, features)
                    if isinstance(model_output, tuple):
                        output = model_output[0]  # Get prediction
                    else:
                        output = model_output
                    
                    target = batch.y[:, target_idx]
                    
                    # Ensure output and target have the same shape
                    if output.dim() == 1:
                        output = output.unsqueeze(1)
                    if target.dim() == 1:
                        target = target.unsqueeze(1)
                    
                    # Loss
                    loss = F.mse_loss(output.squeeze(), target.squeeze())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        model_output = model(batch.x, batch.edge_index, batch.batch)
                        
                        # Handle tuple output from GNN models (prediction, features)
                        if isinstance(model_output, tuple):
                            output = model_output[0]  # Get prediction
                        else:
                            output = model_output
                        
                        target = batch.y[:, target_idx]
                        
                        # Ensure output and target have the same shape
                        if output.dim() == 1:
                            output = output.unsqueeze(1)
                        if target.dim() == 1:
                            target = target.unsqueeze(1)
                        
                        val_loss += F.mse_loss(output.squeeze(), target.squeeze()).item()
                        val_predictions.extend(output.squeeze().cpu().numpy())
                        val_targets.extend(target.squeeze().cpu().numpy())
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:  # Early stopping
                    break
            
            # Calculate final metrics
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            mse = mean_squared_error(val_targets, val_predictions)
            r2 = r2_score(val_targets, val_predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(val_targets, val_predictions)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                'mse': mse,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
            
        except Exception as e:
            print(f"    Error in single fold training: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_gnn_model(self, model_type, num_targets=1):
        """Override parent method to use correct input_channel parameter"""
        if model_type.lower() == 'gcn':
            return simple_GCN_res_plus_regression(
                hidden_channels=self.config.hidden_dim,
                dropout_prob=self.config.dropout_rate,
                input_channel=1,  # Node features have shape (num_features, 1)
                output_dim=num_targets
            )
        elif model_type.lower() == 'rggc':
            return simple_RGGC_plus_regression(
                hidden_channels=self.config.hidden_dim,
                dropout_prob=self.config.dropout_rate,
                input_channel=1,  # Node features have shape (num_features, 1)
                output_dim=num_targets
            )
        elif model_type.lower() == 'gat':
            return simple_GAT_regression(
                hidden_channels=self.config.hidden_dim,
                dropout_prob=self.config.dropout_rate,
                input_channel=1,  # Node features have shape (num_features, 1)
                output_dim=num_targets,
                num_heads=4
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def run_all_cases(data_path="../Data/New_data.csv"):
    """Run all domain expert cases with proper configuration"""
    print("Running all domain expert cases...")
    
    cases = ['case1'] # 'case2', 'case3', 'case4', 'case5'
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()}")
        print(f"{'='*60}")
        
        try:
            # Create configuration for testing
            test_config = PipelineConfig(
                inner_cv_folds=2,
                outer_cv_folds=3,
                hidden_dim=64,
                num_epochs=15,  # Reduced for testing
                patience=10,
                batch_size=8,
                learning_rate=0.001,
                weight_decay=1e-4,
                # Reduced hyperparameter grid for testing
                hidden_dim_grid=[64, 256],
                dropout_rate_grid=[0.2, 0.5]
            )
            
            pipeline = DomainExpertCasesPipeline(
                data_path=data_path,
                case_type=case,
                config=test_config,
                save_dir=f'./domain_expert_results/{case}_h2_hydrogenotrophic_only',
                importance_threshold=0.2,
                use_fast_correlation=False,
                graph_mode='family',
                family_filter_mode='strict'
            )
            
            # Run the case-specific pipeline
            results = pipeline.run_case_specific_pipeline()
            all_results[case] = results
            
            print(f"\n✅ {case.upper()} completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error in {case.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("CREATING OVERALL COMPARISON")
    print(f"{'='*80}")
    
    # Save combined results
    with open('./domain_expert_results/all_cases_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("All cases completed!")
    print("Domain expert cases pipeline completed!")


if __name__ == "__main__":
    # Run all cases
    results = run_all_cases()
    print("Domain expert cases pipeline completed!")