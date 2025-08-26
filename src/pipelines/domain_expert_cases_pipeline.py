import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
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
from .embeddings_pipeline import MixedEmbeddingPipeline
from ..datasets.dataset_regression import MicrobialGNNDataset

# Import GNN models
from ..models.GNNmodelsRegression import (
    simple_GCN_res_plus_regression,
    simple_RGGC_plus_regression,
    simple_GAT_regression,
    GaussianNLLLoss
)

# Import the actual GNNExplainer function
from ..explainers.pipeline_explainer import create_explainer_sparsified_graph

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
        
        # CRITICAL FIX: Override the hyperparameter grid to use fixed values
        # PRODUCTION: Full hyperparameter grid for comprehensive search
        hidden_dim_options = [512, 128, 64]  # PRODUCTION: Full grid
        k_neighbors_options = [8, 10, 12]    # PRODUCTION: Full grid
        
        # Override the parent's hyperparameter grids
        self.gnn_hyperparams = {
            'hidden_dim': hidden_dim_options,
            'k_neighbors': k_neighbors_options
        }
        self.param_grid = list(ParameterGrid(self.gnn_hyperparams))
        
        # For explainer phase, only tune hidden_dim
        self.explainer_hyperparams = {
            'hidden_dim': hidden_dim_options
        }
        self.explainer_param_grid = list(ParameterGrid(self.explainer_hyperparams))
        
        print(f"DOMAIN EXPERT: Production hyperparameter grid:")
        print(f"  hidden_dim options: {hidden_dim_options}")
        print(f"  k_neighbors options: {k_neighbors_options}")
        print(f"  Total combinations: {len(self.param_grid)} (3 x 3 = 9 combinations)")
        print(f"  Nested CV will select the BEST combination for each fold based on MSE")
        print(f"  PRODUCTION PARAMETERS: {num_epochs} epochs, {num_folds} folds")
        
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
    
    def _run_single_target_pipeline(self, target_idx, target_name):
        """Run pipeline for a single target using the parent class functionality"""
        print(f"\nRunning pipeline for {target_name}")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Feature names: {self.dataset.node_feature_names}")
        
        # Create detailed_results folder and graphs folder
        os.makedirs(f"{self.save_dir}/detailed_results", exist_ok=True)
        os.makedirs(f"{self.save_dir}/graphs", exist_ok=True)
        
        results = {}
        
        # Step 1: Train ALL GNN models on KNN-sparsified graph (using parent class method)
        print(f"\nSTEP 1: Training ALL GNN models on KNN-sparsified graph")
        knn_gnn_results = {}
        for model_type in self.gnn_models_to_train:
            # Run the existing pipeline and save fold graphs
            knn_gnn_results[model_type] = self.train_gnn_model_with_visualization(
                model_type=model_type,
                target_idx=target_idx,
                data_list=self.dataset.data_list,
                phase="knn",
                target_name=target_name
            )
            # Save fold-specific graphs after training
            self._save_fold_graphs_post_training(model_type, target_name, "knn", knn_gnn_results[model_type])
            
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
        explainer_data = self.create_explainer_sparsified_graph_with_visualization(
            model=best_gnn_model,
            target_idx=target_idx,
            target_name=target_name
        )
        
        # Step 3: Train ALL GNN models on explainer-sparsified graph (using parent class method)
        print(f"\nSTEP 3: Training ALL GNN models on explainer-sparsified graph")
        explainer_gnn_results = {}
        for model_type in self.gnn_models_to_train:
            explainer_gnn_results[model_type] = self.train_gnn_model_with_visualization(
                model_type=model_type,
                target_idx=target_idx,
                data_list=explainer_data,
                phase="explainer",
                target_name=target_name
            )
            # Save fold-specific graphs after training
            self._save_fold_graphs_post_training(model_type, target_name, "explainer", explainer_gnn_results[model_type])
            
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
        
        # Generate comprehensive graph visualization summary
        self._create_graph_visualization_summary(target_name)
        
        print(f"\nResults saved to {self.save_dir}")
        print(f"Summary: {len(summary_data)} model results")
        print(f"Graph visualizations saved to {self.save_dir}/graphs/")

    def _create_graph_visualization_summary(self, target_name):
        """Create a comprehensive summary of all generated graph visualizations"""
        try:
            import matplotlib.pyplot as plt
            
            graphs_dir = f"{self.save_dir}/graphs"
            if not os.path.exists(graphs_dir):
                print("      No graphs directory found - skipping visualization summary")
                return
            
            print(f"      Creating graph visualization summary for {target_name}...")
            
            # Count visualizations by type
            fold_dirs = [d for d in os.listdir(graphs_dir) if d.startswith('fold_')]
            explainer_dirs = [d for d in os.listdir(graphs_dir) if d.startswith('explainer_')]
            
            # Organize by model and phase
            visualization_counts = {}
            
            for fold_dir in fold_dirs:
                parts = fold_dir.split('_')
                if len(parts) >= 4:  # fold_X_modeltype_phase
                    fold_num = parts[1]
                    model_type = parts[2]
                    phase = parts[3]
                    
                    key = f"{model_type}_{phase}"
                    if key not in visualization_counts:
                        visualization_counts[key] = 0
                    visualization_counts[key] += 1
            
            # Create summary visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Visualization counts by model and phase
            if visualization_counts:
                models_phases = list(visualization_counts.keys())
                counts = list(visualization_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(models_phases)))
                
                bars = ax1.bar(models_phases, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                ax1.set_ylabel('Number of Fold Visualizations', fontsize=12)
                ax1.set_title(f'Graph Visualizations Generated\n{target_name} - {self.case_type.upper()}', 
                             fontsize=14, fontweight='bold')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                            f'{count}', ha='center', va='bottom', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No fold visualizations found', 
                        transform=ax1.transAxes, ha='center', va='center', fontsize=12)
                ax1.set_title('Graph Visualizations Generated', fontsize=14, fontweight='bold')
            
            # Plot 2: Summary statistics
            stats_text = []
            stats_text.append(f"Case: {self.case_type.upper()}")
            stats_text.append(f"Target: {target_name}")
            stats_text.append(f"Total fold visualizations: {len(fold_dirs)}")
            stats_text.append(f"Explainer summaries: {len(explainer_dirs)}")
            stats_text.append(f"Models trained: {', '.join(self.gnn_models_to_train)}")
            stats_text.append(f"Number of features: {len(self.dataset.node_feature_names)}")
            stats_text.append(f"Cross-validation folds: {self.num_folds}")
            
            # Add anchored features info
            if hasattr(self, 'anchored_features') and self.anchored_features:
                stats_text.append(f"Anchored features: {len(self.anchored_features)}")
            
            ax2.text(0.1, 0.9, '\n'.join(stats_text), transform=ax2.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Pipeline Summary', fontsize=14, fontweight='bold')
            
            # Add file structure summary
            file_structure = []
            file_structure.append("Generated Files Structure:")
            file_structure.append("├── graphs/")
            for fold_dir in sorted(fold_dirs)[:5]:  # Show first 5
                file_structure.append(f"│   ├── {fold_dir}/")
            if len(fold_dirs) > 5:
                file_structure.append(f"│   ├── ... ({len(fold_dirs)-5} more)")
            for exp_dir in explainer_dirs:
                file_structure.append(f"│   ├── {exp_dir}/")
            
            ax2.text(0.1, 0.4, '\n'.join(file_structure), transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/graph_visualization_summary_{target_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(f"{self.save_dir}/graph_visualization_summary_{target_name}.pdf", 
                       bbox_inches='tight')
            plt.close()
            
            print(f"      Saved: graph_visualization_summary_{target_name}.png/pdf")
            
        except Exception as e:
            print(f"      WARNING: Could not create graph visualization summary: {e}")

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

    # Simple hook to save fold-specific graphs - will be added to existing pipeline

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

    def _save_fold_graphs_post_training(self, model_type, target_name, phase, model_results):
        """Save fold-specific graphs after training is complete"""
        if not self.use_nested_cv or 'fold_results' not in model_results:
            print(f"  Skipping fold graph saving (no nested CV or fold results)")
            return
            
        print(f"  Saving fold-specific graphs for {model_type.upper()} - {target_name} ({phase})")
        
        # Create fold-specific directories
        fold_graphs_dir = f"{self.save_dir}/fold_specific_graphs/{model_type}_{target_name}_{phase}"
        os.makedirs(fold_graphs_dir, exist_ok=True)
        
        fold_results = model_results['fold_results']
        
        # Save graph information for each fold
        fold_summaries = []
        for fold_result in fold_results:
            fold_num = fold_result['fold']
            
            # Create fold directory
            fold_dir = f"{fold_graphs_dir}/fold_{fold_num}"
            os.makedirs(fold_dir, exist_ok=True)
            
            # Get hyperparameters for this fold
            if 'best_params' in fold_result:
                best_params = fold_result['best_params']
                k_neighbors = best_params.get('k_neighbors', self.k_neighbors)
                hidden_dim = best_params.get('hidden_dim', self.hidden_dim)
            else:
                k_neighbors = self.k_neighbors
                hidden_dim = self.hidden_dim
            
            # Save fold summary
            fold_summary = {
                'fold': fold_num,
                'model_type': model_type,
                'target_name': target_name,
                'phase': phase,
                'k_neighbors': k_neighbors,
                'hidden_dim': hidden_dim,
                'r2': fold_result['r2'],
                'mse': fold_result['mse'],
                'rmse': fold_result.get('rmse', np.sqrt(fold_result['mse'])),
                'mae': fold_result.get('mae', 'N/A')
            }
            fold_summaries.append(fold_summary)
            
            # Save individual fold info
            fold_info_df = pd.DataFrame([fold_summary])
            fold_info_df.to_csv(f"{fold_dir}/fold_{fold_num}_info.csv", index=False)
            
            # Generate and save actual graph visualizations for this fold
            if phase == "knn":
                self._generate_fold_knn_graph_visualization(fold_num, k_neighbors, hidden_dim, fold_dir, target_name)
            elif phase == "explainer":
                self._generate_fold_explainer_graph_visualization(fold_num, k_neighbors, hidden_dim, fold_dir, target_name, fold_result)
            
            print(f"    Saved fold {fold_num}: k={k_neighbors}, h={hidden_dim}, R²={fold_result['r2']:.4f}")
        
        # Save combined fold summary
        if fold_summaries:
            summary_df = pd.DataFrame(fold_summaries)
            summary_df.to_csv(f"{fold_graphs_dir}/all_folds_summary.csv", index=False)
            
            # Create simple visualization of hyperparameter selections
            self._create_simple_fold_visualization(fold_summaries, fold_graphs_dir, model_type, target_name, phase)
        
        print(f"  Fold graphs saved to: {fold_graphs_dir}/")

    def _create_simple_fold_visualization(self, fold_summaries, graphs_dir, model_type, target_name, phase):
        """Create simple visualization of fold results"""
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(fold_summaries)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Fold Results: {model_type.upper()} - {target_name} ({phase.upper()})', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: R² across folds
        ax1 = axes[0, 0]
        ax1.plot(df['fold'], df['r2'], marker='o', linewidth=2, markersize=8)
        ax1.set_title('R² Across Folds')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('R²')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Hyperparameter selections
        ax2 = axes[0, 1]
        if phase == "knn":  # Only k-neighbors varies in KNN phase
            k_counts = df['k_neighbors'].value_counts()
            ax2.bar(k_counts.index.astype(str), k_counts.values)
            ax2.set_title('K-Neighbors Selection Frequency')
            ax2.set_xlabel('K-Neighbors')
        else:  # Only hidden_dim varies in explainer phase
            h_counts = df['hidden_dim'].value_counts()
            ax2.bar(h_counts.index.astype(str), h_counts.values)
            ax2.set_title('Hidden Dim Selection Frequency')
            ax2.set_xlabel('Hidden Dimension')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Performance vs Hyperparameters
        ax3 = axes[1, 0]
        if phase == "knn":
            ax3.scatter(df['k_neighbors'], df['r2'], s=100, alpha=0.7)
            ax3.set_xlabel('K-Neighbors')
        else:
            ax3.scatter(df['hidden_dim'], df['r2'], s=100, alpha=0.7)
            ax3.set_xlabel('Hidden Dimension')
        ax3.set_ylabel('R²')
        ax3.set_title('Performance vs Hyperparameters')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Fold statistics
        ax4 = axes[1, 1]
        metrics = ['r2', 'rmse']
        values = [df['r2'].tolist()]
        if 'rmse' in df.columns:
            values.append(df['rmse'].tolist())
        ax4.boxplot(values, labels=metrics)
        ax4.set_title('Performance Distribution')
        ax4.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/fold_summary_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Fold visualization saved: fold_summary_visualization.png")

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

    def _generate_fold_knn_graph_visualization(self, fold_num, k_neighbors, hidden_dim, fold_dir, target_name):
        """Generate and save k-NN graph visualization for a specific fold with professional styling"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
            
            print(f"      Generating REAL k-NN graph visualization (k={k_neighbors}, h={hidden_dim}) for fold {fold_num}...")
            
            # Get family names
            family_names = self.dataset.node_feature_names
            
            # Get the ACTUAL KNN graph structure from the dataset (not correlation-based!)
            sample_data = self.dataset.data_list[0]  # All samples should have same graph structure
            actual_edge_index = sample_data.edge_index.cpu().numpy()
            
            print(f"        Using ACTUAL KNN graph with {actual_edge_index.shape[1]} edges from dataset")
            
            # Create NetworkX graph from ACTUAL edge structure
            G = nx.Graph()
            for i, family in enumerate(family_names):
                G.add_node(i, name=family)
            
            # Add edges from the ACTUAL KNN graph
            edge_weights = sample_data.edge_weight.cpu().numpy() if hasattr(sample_data, 'edge_weight') and sample_data.edge_weight is not None else None
            edge_types = sample_data.edge_type.cpu().numpy() if hasattr(sample_data, 'edge_type') and sample_data.edge_type is not None else None
            
            positive_edges = []
            negative_edges = []
            
            for i in range(actual_edge_index.shape[1]):
                src, dst = actual_edge_index[0, i], actual_edge_index[1, i]
                if src != dst:  # Skip self-loops for visualization
                    weight = edge_weights[i] if edge_weights is not None else 1.0
                    
                    # Use edge_type if available, otherwise determine from edge attributes
                    if edge_types is not None:
                        is_positive = edge_types[i] == 1
                    else:
                        # Fallback: assume weight represents correlation strength, positive if > 0
                        is_positive = weight > 0
                    
                    G.add_edge(src, dst, weight=abs(weight))
                    
                    if is_positive:
                        positive_edges.append((src, dst))
                    else:
                        negative_edges.append((src, dst))
            
            # Dynamic color assignment (using the fixed version)
            def get_vibrant_node_colors(family_names):
                colors = []
                
                # Create a dynamic color palette with enough distinct colors
                vibrant_colors = [
                    '#FF4444', '#00CC88', '#4488FF', '#FF8800', '#AA4488', 
                    '#44AA88', '#FF6666', '#00AA66', '#6699FF', '#FFAA22',
                    '#CC6699', '#22DD77', '#FF2222', '#00EE99', '#2266DD',
                    '#88AAFF', '#FFAA00', '#888844', '#FF0000', '#00DD00',
                    '#0088FF', '#FF00AA', '#88FF00', '#AA00FF', '#FFDD00'
                ]
                
                # Keywords for functional classification (more flexible)
                functional_groups = {
                    'methanogen': ['methano', 'methan'],
                    'syntrophic': ['syntrop', 'syner', 'pelotomac'],
                    'bacteroidetes': ['bacteroid', 'rikenel', 'prevot', 'parabacter', 'alist'],
                    'firmicutes': ['clostrid', 'christens', 'ruminoc', 'lactobac'],
                    'proteobacteria': ['geobacter', 'desulfov', 'syntrophob'],
                    'spirochaetes': ['spiroch', 'lentimicro'],
                    'thermotoga': ['thermotog'],
                    'archaeoglobus': ['archaeog']
                }
                
                # Assign base colors to functional groups - MUTED COLORS like reference image
                group_colors = {
                    'methanogen': '#D2691E',      # Muted orange-brown (was bright red)
                    'syntrophic': '#4682B4',      # Steel blue (was bright teal)  
                    'bacteroidetes': '#708090',   # Slate gray (was bright blue)
                    'firmicutes': '#8FBC8F',      # Dark sea green (was purple)
                    'proteobacteria': '#CD853F',  # Peru brown (was bright orange)
                    'spirochaetes': '#696969',    # Dim gray (was olive)
                    'thermotoga': '#5F9EA0',      # Cadet blue (was teal-green)
                    'archaeoglobus': '#DAA520'    # Goldenrod (was yellow-orange)
                }
                
                # Create a muted color palette with softer colors
                vibrant_colors = [
                    '#8B4513', '#2F4F4F', '#556B2F', '#800080', '#B8860B', 
                    '#4682B4', '#D2691E', '#8FBC8F', '#708090', '#CD853F',
                    '#5F9EA0', '#BC8F8F', '#9ACD32', '#6495ED', '#DEB887',
                    '#F4A460', '#696969', '#778899', '#BDB76B', '#87CEEB',
                    '#D3D3D3', '#FFB6C1', '#98FB98', '#F0E68C', '#DDA0DD'
                ]
                
                # First pass: assign functional group colors
                assigned_families = set()
                
                for i, family in enumerate(family_names):
                    family_lower = family.lower()
                    assigned = False
                    
                    for group, keywords in functional_groups.items():
                        if any(keyword in family_lower for keyword in keywords):
                            base_color = group_colors[group]
                            # Add slight variation for families in same group
                            color_idx = sum(1 for f in family_names[:i] if any(kw in f.lower() for kw in keywords))
                            if color_idx == 0:
                                colors.append(base_color)
                            else:
                                # Create variations by adjusting brightness
                                colors.append(_adjust_color_brightness(base_color, 0.8 + (color_idx * 0.1)))
                            assigned_families.add(i)
                            assigned = True
                            break
                    
                    if not assigned:
                        # Assign from remaining vibrant colors for unclassified families
                        unassigned_idx = len(assigned_families) - i
                        color_idx = unassigned_idx % len(vibrant_colors)
                        colors.append(vibrant_colors[color_idx])
                
                return colors
            
            colors = get_vibrant_node_colors(family_names)
            
            # Debug: Print actual families and their colors for verification
            print(f"        DEBUG: Found {len(family_names)} families in REAL KNN graph:")
            for i, (family, color) in enumerate(zip(family_names[:8], colors[:8])):  # Show first 8
                print(f"          {i}: {family} -> {color}")
            if len(family_names) > 8:
                print(f"          ... and {len(family_names) - 8} more families")
            
            # Create the visualization
            fig, ax = plt.subplots(1, 1, figsize=(20, 16))
            
            # Use spring layout for better node positioning
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
            
            # Draw nodes with much larger size and better styling
            nx.draw_networkx_nodes(G, pos, 
                                  node_color=colors,
                                  node_size=2000,  # Much larger nodes
                                  alpha=0.9,
                                  ax=ax)
            
            # Draw positive correlation edges (lighter green)
            if positive_edges:
                nx.draw_networkx_edges(G, pos, 
                                     edgelist=positive_edges,
                                     edge_color='#90EE90',  # Light green (much lighter)
                                     alpha=0.5,  # More transparent
                                     width=2)
            
            # Draw negative correlation edges (lighter red)
            if negative_edges:
                nx.draw_networkx_edges(G, pos, 
                                     edgelist=negative_edges,
                                     edge_color='#FFB6C1',  # Light pink (much lighter red)
                                     alpha=0.5,  # More transparent
                                     width=2)
            
            # Draw labels with better formatting
            labels = {i: name.split('_')[-1][:8] for i, name in enumerate(family_names)}  # Shortened labels
            nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', ax=ax)
            
            # Add title and formatting
            ax.set_title(f'REAL k-NN Graph (k={k_neighbors})\\n'
                        f'{len(G.nodes())} nodes, {len(G.edges())} edges '
                        f'({len(positive_edges)} pos, {len(negative_edges)} neg)\\n'
                        f'Fold {fold_num} - Hidden Dim: {hidden_dim}', 
                        fontsize=18, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Add legend with muted colors
            edge_legend = [
                Patch(color='#90EE90', alpha=0.5, label='Positive Correlation'),
                Patch(color='#FFB6C1', alpha=0.5, label='Negative Correlation')
            ]
            ax.legend(handles=edge_legend, loc='upper right', fontsize=14, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save with hyperparameter info in filename
            plt.savefig(f"{fold_dir}/knn_graph_k{k_neighbors}_h{hidden_dim}_fold{fold_num}.png", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(f"{fold_dir}/knn_graph_k{k_neighbors}_h{hidden_dim}_fold{fold_num}.pdf", 
                       bbox_inches='tight')
            plt.close()
            
            print(f"        REAL k-NN graph saved: {len(G.nodes())} nodes, {len(G.edges())} edges ({len(positive_edges)} pos, {len(negative_edges)} neg), k={k_neighbors}, h={hidden_dim}")
                  
        except Exception as e:
            print(f"        Error generating REAL k-NN visualization for fold {fold_num}: {str(e)}")
            import traceback
            traceback.print_exc()

    def _generate_fold_explainer_graph_visualization(self, fold_num, k_neighbors, hidden_dim, fold_dir, target_name, fold_result):
        """Generate and save explainer (sparsified) graph visualization for a specific fold with side-by-side comparison"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
            
            print(f"      Generating REAL explainer graph visualization (k={k_neighbors}, h={hidden_dim}) for fold {fold_num}...")
            
            # Get family data for correlation analysis
            family_names = self.dataset.node_feature_names
            family_data = []
            for data in self.dataset.data_list:
                family_data.append(data.x.squeeze().numpy())
            family_data = np.array(family_data).T  # Shape: (families, samples)
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
            
            # Helper function for vibrant node colors - DYNAMIC based on actual data
            def get_vibrant_node_colors(family_names):
                colors = []
                
                # Create a dynamic color palette with enough distinct colors
                vibrant_colors = [
                    '#FF4444', '#00CC88', '#4488FF', '#FF8800', '#AA4488', 
                    '#44AA88', '#FF6666', '#00AA66', '#6699FF', '#FFAA22',
                    '#CC6699', '#22DD77', '#FF2222', '#00EE99', '#2266DD',
                    '#88AAFF', '#FFAA00', '#888844', '#FF0000', '#00DD00',
                    '#0088FF', '#FF00AA', '#88FF00', '#AA00FF', '#FFDD00'
                ]
                
                # Keywords for functional classification (more flexible)
                functional_groups = {
                    'methanogen': ['methano', 'methan'],
                    'syntrophic': ['syntrop', 'syner', 'pelotomac'],
                    'bacteroidetes': ['bacteroid', 'rikenel', 'prevot', 'parabacter', 'alist'],
                    'firmicutes': ['clostrid', 'christens', 'ruminoc', 'lactobac'],
                    'proteobacteria': ['geobacter', 'desulfov', 'syntrophob'],
                    'spirochaetes': ['spiroch', 'lentimicro'],
                    'thermotoga': ['thermotog'],
                    'archaeoglobus': ['archaeog']
                }
                
                # Assign base colors to functional groups - MUTED COLORS like reference image
                group_colors = {
                    'methanogen': '#D2691E',      # Muted orange-brown (was bright red)
                    'syntrophic': '#4682B4',      # Steel blue (was bright teal)  
                    'bacteroidetes': '#708090',   # Slate gray (was bright blue)
                    'firmicutes': '#8FBC8F',      # Dark sea green (was purple)
                    'proteobacteria': '#CD853F',  # Peru brown (was bright orange)
                    'spirochaetes': '#696969',    # Dim gray (was olive)
                    'thermotoga': '#5F9EA0',      # Cadet blue (was teal-green)
                    'archaeoglobus': '#DAA520'    # Goldenrod (was yellow-orange)
                }
                
                # Create a muted color palette with softer colors
                vibrant_colors = [
                    '#8B4513', '#2F4F4F', '#556B2F', '#800080', '#B8860B', 
                    '#4682B4', '#D2691E', '#8FBC8F', '#708090', '#CD853F',
                    '#5F9EA0', '#BC8F8F', '#9ACD32', '#6495ED', '#DEB887',
                    '#F4A460', '#696969', '#778899', '#BDB76B', '#87CEEB',
                    '#D3D3D3', '#FFB6C1', '#98FB98', '#F0E68C', '#DDA0DD'
                ]
                
                # First pass: assign functional group colors
                assigned_families = set()
                
                for i, family in enumerate(family_names):
                    family_lower = family.lower()
                    assigned = False
                    
                    for group, keywords in functional_groups.items():
                        if any(keyword in family_lower for keyword in keywords):
                            base_color = group_colors[group]
                            # Add slight variation for families in same group
                            color_idx = sum(1 for f in family_names[:i] if any(kw in f.lower() for kw in keywords))
                            if color_idx == 0:
                                colors.append(base_color)
                            else:
                                # Create variations by adjusting brightness
                                colors.append(_adjust_color_brightness(base_color, 0.8 + (color_idx * 0.1)))
                            assigned_families.add(i)
                            assigned = True
                            break
                    
                    if not assigned:
                        # Assign from remaining vibrant colors for unclassified families
                        unassigned_idx = len(assigned_families) - i
                        color_idx = unassigned_idx % len(vibrant_colors)
                        colors.append(vibrant_colors[color_idx])
                
                return colors

            def _adjust_color_brightness(hex_color, factor):
                """Adjust the brightness of a hex color"""
                # Convert hex to RGB
                hex_color = hex_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                
                # Adjust brightness
                rgb = tuple(min(255, max(0, int(c * factor))) for c in rgb)
                
                # Convert back to hex
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            
            # Get node colors
            node_colors = get_vibrant_node_colors(family_names)
            
            # Debug: Print actual families and their colors for verification
            print(f"        DEBUG: Found {len(family_names)} families in dataset:")
            for i, (family, color) in enumerate(zip(family_names[:10], node_colors[:10])):  # Show first 10
                print(f"          {i}: {family} -> {color}")
            if len(family_names) > 10:
                print(f"          ... and {len(family_names) - 10} more families")
            
            # Create KNN Graph (left side) - using actual KNN structure from the dataset
            G_knn = nx.Graph()
            for i, family in enumerate(family_names):
                G_knn.add_node(i, name=family)
            
            # Add KNN edges based on actual graph structure
            knn_edge_index = self.dataset.data_list[0].edge_index.cpu().numpy()
            for i in range(knn_edge_index.shape[1]):
                src, dst = knn_edge_index[0, i], knn_edge_index[1, i]
                if src != dst:  # Avoid self-loops for visualization
                    # Calculate correlation for edge color
                    if src < len(family_data) and dst < len(family_data):
                        corr = np.corrcoef(family_data[src], family_data[dst])[0, 1]
                        G_knn.add_edge(src, dst, weight=abs(corr), corr=corr)
            
            # Create Explainer Graph (right side) - Get explainer data from fold_result
            G_explainer = nx.Graph()
            for i, family in enumerate(family_names):
                G_explainer.add_node(i, name=family)
            
            # Check if explainer data exists in fold_result
            explainer_data = fold_result.get('explainer_data', None)
            if explainer_data is not None and len(explainer_data) > 0:
                print(f"        Using REAL explainer sparsified graph data from fold_result")
                explainer_edge_index = explainer_data[0].edge_index.cpu().numpy()
                
                # Add explainer edges
                for i in range(explainer_edge_index.shape[1]):
                    src, dst = explainer_edge_index[0, i], explainer_edge_index[1, i]
                    if src != dst:  # Avoid self-loops for visualization
                        # Calculate correlation for edge color
                        if src < len(family_data) and dst < len(family_data):
                            corr = np.corrcoef(family_data[src], family_data[dst])[0, 1]
                            G_explainer.add_edge(src, dst, weight=1.0, corr=corr)
            else:
                print(f"        WARNING: No explainer sparsified graph data found in fold_result! Creating empty graph")
            
            # Use spring layout for consistent positioning
            pos = nx.spring_layout(G_knn, k=3, iterations=100, seed=42)
            
            # Left plot: KNN Graph
            ax1.set_title(f'KNN Graph (k={k_neighbors})\n{len(G_knn.edges())} edges', 
                         fontsize=20, fontweight='bold', pad=20)
            
            # Draw nodes for KNN
            nx.draw_networkx_nodes(G_knn, pos, 
                                 node_color=node_colors, 
                                 node_size=2000,  # Larger nodes
                                 alpha=0.9, ax=ax1)
            
            # Draw edges for KNN by correlation sign
            positive_edges_knn = [(u, v) for u, v, d in G_knn.edges(data=True) if d.get('corr', 0) > 0]
            negative_edges_knn = [(u, v) for u, v, d in G_knn.edges(data=True) if d.get('corr', 0) < 0]
            
            if positive_edges_knn:
                nx.draw_networkx_edges(G_knn, pos, 
                                     edgelist=positive_edges_knn,
                                     edge_color='#90EE90',  # Light green (much lighter)
                                     alpha=0.5,  # More transparent
                                     width=2,  # Thinner
                                     ax=ax1)
            
            if negative_edges_knn:
                nx.draw_networkx_edges(G_knn, pos, 
                                     edgelist=negative_edges_knn,
                                     edge_color='#FFB6C1',  # Light pink (much lighter red)
                                     alpha=0.5,  # More transparent
                                     width=2,  # Thinner
                                     ax=ax1)
            
            # Draw labels for KNN
            labels = {i: name.split('_')[-1][:8] for i, name in enumerate(family_names)}  # Shortened labels
            nx.draw_networkx_labels(G_knn, pos, labels, font_size=12, font_weight='bold', ax=ax1)
            
            # Right plot: Explainer Sparsified Graph
            explainer_title = f'GNNExplainer Sparsified Graph (Real)\\n{len(G_explainer.edges())} edges'
            sparsification_ratio = 1 - (len(G_explainer.edges()) / max(len(G_knn.edges()), 1))
            explainer_title += f' ({sparsification_ratio*100:.1f}% sparsified)'
            ax2.set_title(explainer_title, fontsize=20, fontweight='bold', pad=20)
            
            # Draw nodes for Explainer
            nx.draw_networkx_nodes(G_explainer, pos, 
                                 node_color=node_colors, 
                                 node_size=2000,  # Larger nodes
                                 alpha=0.9, ax=ax2)
            
            # Draw edges for Explainer by correlation sign with lighter colors
            positive_edges_explainer = [(u, v) for u, v, d in G_explainer.edges(data=True) if d.get('corr', 0) > 0]
            negative_edges_explainer = [(u, v) for u, v, d in G_explainer.edges(data=True) if d.get('corr', 0) < 0]
            
            if positive_edges_explainer:
                nx.draw_networkx_edges(G_explainer, pos, 
                                     edgelist=positive_edges_explainer,
                                     edge_color='#90EE90',  # Light green (much lighter)
                                     alpha=0.5,  # More transparent
                                     width=2,  # Thinner
                                     ax=ax2)
            
            if negative_edges_explainer:
                nx.draw_networkx_edges(G_explainer, pos, 
                                     edgelist=negative_edges_explainer,
                                     edge_color='#FFB6C1',  # Light pink (much lighter red)
                                     alpha=0.5,  # More transparent
                                     width=2,  # Thinner
                                     ax=ax2)
            
            # Draw labels for Explainer
            nx.draw_networkx_labels(G_explainer, pos, labels, font_size=12, font_weight='bold', ax=ax2)
            
            # Remove axes
            ax1.axis('off')
            ax2.axis('off')
            
            # Create legends with lighter colors
            edge_legend = [
                Patch(color='#90EE90', alpha=0.5, label='Positive Correlation'),
                Patch(color='#FFB6C1', alpha=0.5, label='Negative Correlation')
            ]
            
            # Add edge legend to first subplot
            ax1.legend(handles=edge_legend, loc='upper right', fontsize=14, framealpha=0.9)
            
            # Create title with proper R² formatting
            r2_value = fold_result.get('r2', 'N/A')
            r2_display = r2_value if isinstance(r2_value, str) else f"{r2_value:.4f}"
            
            plt.suptitle(f'Graph Comparison: KNN vs GNNExplainer (Fold {fold_num})\\n'
                        f'Target: {target_name} | k_neighbors={k_neighbors} | hidden_dim={hidden_dim}\\n'
                        f'R² Score: {r2_display}', 
                        fontsize=18, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the visualization with hyperparameter info in filename
            plt.savefig(f"{fold_dir}/explainer_comparison_k{k_neighbors}_h{hidden_dim}_fold{fold_num}.png", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(f"{fold_dir}/explainer_comparison_k{k_neighbors}_h{hidden_dim}_fold{fold_num}.pdf", 
                       bbox_inches='tight')
            plt.close()
            
            # Print summary with proper R² formatting
            r2_value = fold_result.get('r2', 'N/A')
            r2_display = r2_value if isinstance(r2_value, str) else f"{r2_value:.4f}"
            
            print(f"        REAL Explainer graph saved: {len(G_explainer.edges())}/{len(G_knn.edges())} edges "
                  f"({100*sparsification_ratio:.1f}% sparsified), "
                  f"k={k_neighbors}, h={hidden_dim}, R²={r2_display}")
                  
        except Exception as e:
            print(f"        Error generating explainer visualization for fold {fold_num}: {str(e)}")
            import traceback
            traceback.print_exc()

    def train_gnn_model_with_visualization(self, model_type, target_idx, data_list, phase="knn", target_name="target"):
        """Train GNN model with nested CV and generate graph visualizations for each fold"""
        # Call the parent's nested CV training method
        results = self.train_gnn_model(model_type=model_type, target_idx=target_idx, data_list=data_list)
        
        # Extract fold information from results to generate visualizations
        if 'fold_results' in results:
            print(f"      Generating {phase} graph visualizations for {model_type.upper()} model...")
            
            # For explainer phase, we need to use the best trained model to run GNNExplainer
            explainer_data = None
            if phase == "explainer":
                # Get the best overall model (from the results)
                best_model = results.get('best_model')
                if best_model is not None:
                    # Run real GNNExplainer with the best model
                    print(f"      Running REAL GNNExplainer for {target_name}...")
                    explainer_data = create_explainer_sparsified_graph(
                        pipeline=self, 
                        model=best_model, 
                        target_idx=target_idx, 
                        importance_threshold=0.3
                    )
            
            for fold_result in results['fold_results']:
                try:
                    fold_num = fold_result.get('fold', 0)
                    best_params = fold_result.get('best_params', {})
                    k_neighbors = best_params.get('k_neighbors', self.k_neighbors)
                    hidden_dim = best_params.get('hidden_dim', self.hidden_dim)
                    
                    # Store explainer data in fold result for visualization access
                    if phase == "explainer" and explainer_data is not None:
                        fold_result['explainer_data'] = explainer_data
                    
                    # Create fold-specific directory
                    fold_dir = f"{self.save_dir}/graphs/fold_{fold_num}_{model_type}_{phase}"
                    os.makedirs(fold_dir, exist_ok=True)
                    
                    if phase == "knn":
                        # Generate KNN graph visualization
                        self._generate_fold_knn_graph_visualization(
                            fold_num, k_neighbors, hidden_dim, fold_dir, target_name
                        )
                    elif phase == "explainer":
                        # Generate explainer comparison visualization using real data
                        self._generate_fold_explainer_graph_visualization(
                            fold_num, k_neighbors, hidden_dim, fold_dir, target_name, fold_result
                        )
                        
                except Exception as e:
                    print(f"      Error generating visualization for fold {fold_result.get('fold', 'unknown')}: {str(e)}")
        
        return results

    def create_explainer_sparsified_graph_with_visualization(self, model, target_idx, target_name):
        """Create explainer-sparsified graph using real GNNExplainer and generate visualization"""
        # Call the actual GNNExplainer method
        print(f"      Using REAL GNNExplainer for {target_name} (target_idx={target_idx})")
        explainer_data = create_explainer_sparsified_graph(
            pipeline=self, 
            model=model, 
            target_idx=target_idx, 
            importance_threshold=0.3  # Can be adjusted
        )
        
        # Generate explainer overview visualization
        explainer_dir = f"{self.save_dir}/graphs/explainer_overview"
        os.makedirs(explainer_dir, exist_ok=True)
        
        # Create a summary visualization of the explainer process
        self._create_explainer_summary_visualization(explainer_data, explainer_dir, target_name)
        
        return explainer_data

    def _create_explainer_summary_visualization(self, explainer_data, explainer_dir, target_name):
        """Create a summary visualization of the explainer sparsification process"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        print(f"      Generating explainer summary visualization for {target_name}...")
        
        # Compare original vs explainer graph statistics
        original_edges = len(self.dataset.data_list[0].edge_index[0])
        explainer_edges = len(explainer_data[0].edge_index[0])
        sparsification_ratio = 1 - (explainer_edges / original_edges) if original_edges > 0 else 0
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Edge count comparison
        categories = ['Original\n(KNN)', 'Explainer\n(Sparsified)']
        edge_counts = [original_edges, explainer_edges]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax1.bar(categories, edge_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Number of Edges', fontsize=12)
        ax1.set_title(f'Graph Sparsification Overview\n{target_name}', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, edge_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(edge_counts),
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Add sparsification ratio text
        ax1.text(0.5, 0.8, f'Sparsification: {sparsification_ratio:.1%}', 
                transform=ax1.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontsize=12, fontweight='bold')
        
        # Plot 2: Sample visualization of one graph
        if len(explainer_data) > 0:
            sample_data = explainer_data[0]
            edge_index = sample_data.edge_index.cpu().numpy()
            
            # Create networkx graph
            G = nx.Graph()
            G.add_edges_from(zip(edge_index[0], edge_index[1]))
            
            # Use spring layout for visualization
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            
            # Draw the graph
            nx.draw(G, pos, ax=ax2, 
                   node_color='lightblue', 
                   node_size=100,
                   edge_color='gray',
                   alpha=0.7,
                   with_labels=False)
            
            ax2.set_title(f'Sample Explainer Graph\n({len(G.nodes)} nodes, {len(G.edges)} edges)', 
                         fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No explainer data available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=12)
            ax2.set_title('Sample Explainer Graph', fontsize=12, fontweight='bold')
        
        # Add overall title
        fig.suptitle(f'GNNExplainer Sparsification Summary - {target_name}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{explainer_dir}/explainer_summary_{target_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{explainer_dir}/explainer_summary_{target_name}.pdf", 
                   bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: explainer_summary_{target_name}.png/pdf")

    def _adjust_color_brightness(self, hex_color, factor):
        """Adjust the brightness of a hex color"""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Adjust brightness
        rgb = tuple(min(255, max(0, int(c * factor))) for c in rgb)
        
        # Convert back to hex
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def run_all_cases(data_path="../Data/New_data.csv"):
    """Run all domain expert cases"""
    print("Running all domain expert cases...")
    
    # cases = ['case1', 'case2', 'case3', 'case4', 'case5']
    cases = ['case1', 'case2', 'case3']
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
                hidden_dim=64,
                num_epochs=200,    # PRODUCTION: Back to 200 epochs
                num_folds=5,       # PRODUCTION: Back to 5 folds
                save_dir="./domain_expert_results"
            )
            
            case_results = pipeline.run_case_specific_pipeline()
            all_results[case] = case_results
            print(f"\n{case.upper()} completed successfully!")
            
        except Exception as e:
            print(f"Error in {case}: {e}")
            all_results[case] = None
            continue
    
    return all_results


if __name__ == "__main__":
    print("Starting Domain Expert Cases Pipeline...")
    print("=" * 60)
    
    # Run all domain expert cases
    try:
        results = run_all_cases()
        print("\n" + "=" * 60)
        print("Domain Expert Cases Pipeline completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running domain expert cases: {e}")
        import traceback
        traceback.print_exc()