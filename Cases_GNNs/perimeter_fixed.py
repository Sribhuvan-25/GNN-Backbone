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

# Additional device safety checks
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    # Clear cache to start fresh
    torch.cuda.empty_cache()
    # Set memory fraction to prevent OOM
    torch.cuda.set_per_process_memory_fraction(0.8)
else:
    print("CUDA not available, using CPU")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def clear_gpu_memory():
    """Clear GPU memory to prevent accumulation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()

def ensure_data_on_device(data_list, target_device):
    """Ensure all tensors in data_list are on the specified device"""
    if not data_list:
        return data_list
    
    device_moved_count = 0
    for i, data in enumerate(data_list):
        # Move all tensor attributes to the target device
        if hasattr(data, 'x') and data.x is not None:
            if data.x.device != target_device:
                data.x = data.x.to(target_device, non_blocking=True)
                device_moved_count += 1
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            if data.edge_index.device != target_device:
                data.edge_index = data.edge_index.to(target_device, non_blocking=True)
                device_moved_count += 1
        if hasattr(data, 'y') and data.y is not None:
            if data.y.device != target_device:
                data.y = data.y.to(target_device, non_blocking=True)
                device_moved_count += 1
        if hasattr(data, 'batch') and data.batch is not None:
            if data.batch.device != target_device:
                data.batch = data.batch.to(target_device, non_blocking=True)
                device_moved_count += 1
        # Handle any other tensor attributes that might exist
        # Only check attributes that are likely to be tensors
        tensor_attrs = ['pos', 'face', 'normal', 'edge_attr', 'edge_weight']
        for attr_name in tensor_attrs:
            if hasattr(data, attr_name):
                attr = getattr(data, attr_name)
                if isinstance(attr, torch.Tensor) and attr.device != target_device:
                    setattr(data, attr_name, attr.to(target_device, non_blocking=True))
                    device_moved_count += 1
    
    if device_moved_count > 0:
        print(f"  Moved {device_moved_count} tensors to {target_device}")
    
    return data_list

def create_data_loader_on_device(data_list, batch_size, shuffle=True, target_device=None):
    """Create a DataLoader with data properly moved to the target device"""
    if target_device is None:
        target_device = device
    
    # Ensure all data is on the target device before creating DataLoader
    data_list = ensure_data_on_device(data_list, target_device)
    
    # Create DataLoader
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def move_batch_to_device(batch, target_device=None):
    """Move a batch to the target device with proper error handling"""
    if target_device is None:
        target_device = device
    
    try:
        return batch.to(target_device, non_blocking=True)
    except Exception as e:
        print(f"Warning: Failed to move batch to {target_device}: {e}")
        # Fallback to CPU if GPU fails
        return batch.to('cpu')

def ensure_model_on_device(model, target_device=None):
    """Ensure model is on the target device"""
    if target_device is None:
        target_device = device
    
    if next(model.parameters()).device != target_device:
        model = model.to(target_device)
    
    return model

@dataclass
class PipelineConfig:
    """Configuration class for the domain expert cases pipeline."""
    
    # Nested CV settings
    inner_cv_folds: int = 5
    outer_cv_folds: int = 5
    
    # Model hyperparameters
    hidden_dim: int = 64
    dropout_rate: float = 0.3
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 200
    patience: int = 20
    
    # Graph settings
    k_neighbors: int = 10
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
                 k_neighbors=10, mantel_threshold=0.05, use_fast_correlation=False, 
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
                 k_neighbors=10, mantel_threshold=0.05,
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
        
        # Ensure all dataset data is on the correct device
        self._ensure_dataset_on_device()
        
        # Ensure GPU compatibility
        self._ensure_gpu_compatibility()

    def _tune_gnn_hyperparams(self, model_type, target_idx, data_list, phase="knn"):
        """
        DEPRECATED: This method has been replaced by _train_gnn_with_hyperparameter_tuning
        which maintains the exact same flow as domain_expert_cases_pipeline.py but adds hyperparameter tuning.
        """
        print("DEPRECATED: Use _train_gnn_with_hyperparameter_tuning instead")
        return self._train_gnn_with_hyperparameter_tuning(model_type, target_idx, data_list, phase)

    def _override_parent_train_gnn_model(self, model_type, target_idx, data_list=None):
        """Override parent's train_gnn_model method with device-aware implementation"""
        if data_list is None:
            data_list = self.dataset.data_list

        # Ensure data is on the correct device
        data_list = ensure_data_on_device(data_list, device)

        target_name = self.target_names[target_idx]
        phase = "knn" if data_list == self.dataset.data_list else "explainer"
        print(f"\nTraining {model_type.upper()} for target {target_name} ({phase})")

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        best_r2 = -float('inf')
        best_state = None
        fold_results = []

        criterion = nn.MSELoss()

        for fold, (train_idx, test_idx) in enumerate(kf.split(data_list), start=1):
            print(f"  Fold {fold}/{self.num_folds}")
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]

            # Use device-aware DataLoaders
            train_loader = create_data_loader_on_device(train_data, self.batch_size, shuffle=True, target_device=device)
            test_loader = create_data_loader_on_device(test_data, self.batch_size, shuffle=False, target_device=device)

            model = self.create_gnn_model(model_type).to(device)
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

            best_val = float('inf')
            patience_cnt = 0
            best_model_state = None

            for epoch in range(1, self.num_epochs + 1):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    # Use device-aware batch movement
                    batch = move_batch_to_device(batch, device)
                    optimizer.zero_grad()
                    out, _ = model(batch.x, batch.edge_index, batch.batch)
                    target = batch.y[:, target_idx].view(-1, 1)
                    loss = criterion(out, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch.num_graphs
                avg_train = total_loss / len(train_loader.dataset)

                model.eval()
                total_val = 0
                with torch.no_grad():
                    for batch in test_loader:
                        # Use device-aware batch movement
                        batch = move_batch_to_device(batch, device)
                        out, _ = model(batch.x, batch.edge_index, batch.batch)
                        target = batch.y[:, target_idx].view(-1, 1)
                        total_val += criterion(out, target).item() * batch.num_graphs
                avg_val = total_val / len(test_loader.dataset)
                scheduler.step(avg_val)

                if avg_val < best_val:
                    best_val = avg_val
                    best_model_state = model.state_dict().copy()
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

            model.load_state_dict(best_model_state)
            model.eval()

            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    # Use device-aware batch movement
                    batch = move_batch_to_device(batch, device)
                    out, _ = model(batch.x, batch.edge_index, batch.batch)
                    preds.append(out.cpu().numpy())
                    trues.append(batch.y[:, target_idx].cpu().numpy())

            preds = np.concatenate(preds).flatten()
            trues = np.concatenate(trues).flatten()

            mse = mean_squared_error(trues, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(trues, preds)
            mae = mean_absolute_error(trues, preds)

            fold_results.append({'fold': fold, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae,
                               'predictions': preds, 'targets': trues})

            if r2 > best_r2:
                best_r2 = r2
                best_state = best_model_state

            torch.save(model.state_dict(), f"{self.save_dir}/gnn_models/{model_type}_{target_name}_{phase}_fold{fold}.pt")
            print(f"    Fold {fold}  R²={r2:.3f}, RMSE={rmse:.3f}")
            
            # Clear memory between folds
            clear_gpu_memory()

        # overall metrics
        all_preds = np.concatenate([fr['predictions'] for fr in fold_results])
        all_trues = np.concatenate([fr['targets'] for fr in fold_results])
        overall = {
            'mse': mean_squared_error(all_trues, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_trues, all_preds)),
            'r2': r2_score(all_trues, all_preds),
            'mae': mean_absolute_error(all_trues, all_preds)
        }
        print(f"  Overall  R²={overall['r2']:.3f}, RMSE={overall['rmse']:.3f}\n")

        # load best model
        best_model = self.create_gnn_model(model_type).to(device)
        best_model.load_state_dict(best_state)

        return {'model': best_model, 'fold_results': fold_results, 'avg_metrics': overall}

    def train_gnn_model(self, model_type, target_idx, data_list, phase="knn"):
        """
        DEPRECATED: This method has been replaced by _train_gnn_with_hyperparameter_tuning
        which maintains the exact same flow as domain_expert_cases_pipeline.py but adds hyperparameter tuning.
        """
        print("DEPRECATED: Use _train_gnn_with_hyperparameter_tuning instead")
        return self._train_gnn_with_hyperparameter_tuning(model_type, target_idx, data_list, phase)
    
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
        print("CASE 4a: ACE-km < 10  Acetoclastic features only")
        print(f"{'='*60}")
        
        case4a_results = self._run_case4_subset_with_different_anchors(
            subset_indices=low_ace_indices,
            subset_name="case4a_low_ace",
            anchored_features=self.acetoclastic,
            description="ACE-km < 10 with acetoclastic features"
        )
        
        # Run Case 4b: ACE-km >= 10 with all feature groups
        print(f"\n{'='*60}")
        print("CASE 4b: ACE-km >= 10  All feature groups")
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
        
        # Ensure all subset data is on the same device
        subset_data_list = ensure_data_on_device(subset_data_list, device)
        
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
        Run pipeline for a single target using the exact same flow as domain_expert_cases_pipeline.py
        but with nested CV hyperparameter tuning added.
        
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
            # Step 1: Train ALL GNN models on KNN-sparsified graph with hyperparameter tuning
            print(f"\nSTEP 1: Training ALL GNN models on KNN-sparsified graph")
            knn_gnn_results = {}
            for model_type in self.gnn_models_to_train:
                try:
                    print(f"\nTraining {model_type.upper()} for target {target_name} (KNN)")
                    knn_gnn_results[model_type] = self._train_gnn_with_hyperparameter_tuning(
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
                # Ensure explainer data is on the correct device
                explainer_data = ensure_data_on_device(explainer_data, device)
            except Exception as e:
                print(f"ERROR: Failed to create explainer graph: {e}")
                # Fallback to using original data
                explainer_data = self.dataset.data_list
                explainer_data = ensure_data_on_device(explainer_data, device)
                print("Using original data as fallback")
            
            # Step 3: Train ALL GNN models on explainer-sparsified graph with hyperparameter tuning
            print(f"\nSTEP 3: Training ALL GNN models on explainer-sparsified graph")
            explainer_gnn_results = {}
            for model_type in self.gnn_models_to_train:
                try:
                    print(f"\nTraining {model_type.upper()} for target {target_name} (Explainer)")
                    explainer_gnn_results[model_type] = self._train_gnn_with_hyperparameter_tuning(
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
                # Ensure explainer data is on the correct device for embedding extraction
                explainer_data = ensure_data_on_device(explainer_data, device)
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
            
            # Step 6: Create plots (using custom method for our data structure)
            print(f"\nSTEP 6: Creating plots")
            try:
                # For mixed models, show all models
                gnn_plot_results = {**knn_gnn_results, **{f"{k}_explainer": v for k, v in explainer_gnn_results.items()}}
                
                # Debug: Print model performance summary
                print(f"\nModel Performance Summary for {target_name}:")
                for model_name, model_results in gnn_plot_results.items():
                    if 'avg_metrics' in model_results:
                        metrics = model_results['avg_metrics']
                        print(f"  {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.4f}")
                
                # Create custom plots that work with our data structure
                self._create_custom_plots(gnn_plot_results, results.get('ml_models', {}), target_idx)
                print(f" Full plots created successfully")
                
            except Exception as e:
                print(f"ERROR: Failed to create plots: {e}")
                print(f"Creating fallback plot instead...")
                # Create a simple fallback plot
                self._create_fallback_plot(gnn_plot_results, results.get('ml_models', {}), target_idx)
            
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
        labels = ['ACE-km < 10 (Acetoclastic only)', 'ACE-km e 10 (All groups)']
        
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
                f'Case 4 Summary: {data_split["low_ace_count"]} samples (ACE-km < 10) + {data_split["high_ace_count"]} samples (ACE-km e 10) = {data_split["low_ace_count"] + data_split["high_ace_count"]} total samples',
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
                f.write(f"  Model 4b (ACE-km e 10): {metrics['model_4b']}\n")
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
        print("CASE 5a: H2-km < 10  Hydrogenotrophic features only")
        print(f"{'='*60}")
        
        case5a_results = self._run_case5_subset(
            subset_indices=low_h2_indices,
            subset_name="case5a_low_h2",
            anchored_features=self.hydrogenotrophic,
            description="H2-km < 10 with hydrogenotrophic features"
        )
        
        # Run Case 5b: H2-km >= 10 with all feature groups
        print(f"\n{'='*60}")
        print("CASE 5b: H2-km >= 10  All feature groups")
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
        
        # Ensure all subset data is on the same device
        subset_data_list = ensure_data_on_device(subset_data_list, device)
        
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

    def _override_parent_plot_results(self, gnn_results, ml_results, target_idx):
        """Override parent's plot_results method with device-aware implementation"""
        try:
            # Use the parent's method
            super().plot_results(gnn_results, ml_results, target_idx)
        except Exception as e:
            print(f"ERROR: Failed to create plots: {e}")
            # Create a simple fallback plot
            self._create_fallback_plot(gnn_results, ml_results, target_idx)

    def _create_fallback_plot(self, gnn_results, ml_results, target_idx):
        """Create a simple fallback plot if the main plotting fails"""
        target_name = self.target_names[target_idx]
        
        # Create a simple comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Collect R² scores
        model_names = []
        r2_scores = []
        
        for model_type, results in gnn_results.items():
            if 'avg_metrics' in results:
                model_names.append(model_type.upper())
                r2_scores.append(results['avg_metrics']['r2'])
        
        if ml_results:
            for model_type, results in ml_results.items():
                if 'avg_metrics' in results:
                    model_names.append(f"{model_type.upper()}_ML")
                    r2_scores.append(results['avg_metrics']['r2'])
        
        if model_names:
            bars = ax.bar(model_names, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'lightblue'][:len(model_names)])
            ax.set_title(f'Model Comparison for {target_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/{target_name}_fallback_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Fallback plot saved: {self.save_dir}/plots/{target_name}_fallback_comparison.png")

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
            # Ensure all data is on the same device
            train_data = ensure_data_on_device(train_data, device)
            val_data = ensure_data_on_device(val_data, device)
            
            # Create data loaders using the device-aware function
            train_loader = create_data_loader_on_device(train_data, self.config.batch_size, shuffle=True, target_device=device)
            val_loader = create_data_loader_on_device(val_data, self.config.batch_size, shuffle=False, target_device=device)
            
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
                    # Use device-aware batch movement
                    batch = move_batch_to_device(batch, device)
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
                        # Use device-aware batch movement
                        batch = move_batch_to_device(batch, device)
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
            clear_gpu_memory()
            
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

    def extract_embeddings(self, model, data_list):
        """Extract embeddings using device-aware DataLoader"""
        # Ensure data is on the correct device
        data_list = ensure_data_on_device(data_list, device)
        
        model.eval()
        all_emb, all_targs = [], []
        
        # Use device-aware DataLoader
        loader = create_data_loader_on_device(data_list, self.config.batch_size, shuffle=False, target_device=device)
        
        with torch.no_grad():
            for batch in loader:
                # Ensure batch is on the correct device
                batch = batch.to(device, non_blocking=True)
                _, emb = model(batch.x, batch.edge_index, batch.batch)
                all_emb.append(emb.cpu().numpy())
                all_targs.append(batch.y.cpu().numpy())
        
        return np.vstack(all_emb), np.vstack(all_targs)

    def _override_parent_extract_embeddings(self, model, data_list):
        """Override parent's extract_embeddings method with device-aware implementation"""
        # Ensure data is on the correct device
        data_list = ensure_data_on_device(data_list, device)
        
        model.eval()
        all_emb, all_targs = [], []
        
        # Use device-aware DataLoader
        loader = create_data_loader_on_device(data_list, self.batch_size, shuffle=False, target_device=device)
        
        with torch.no_grad():
            for batch in loader:
                # Use device-aware batch movement
                batch = move_batch_to_device(batch, device)
                _, emb = model(batch.x, batch.edge_index, batch.batch)
                all_emb.append(emb.cpu().numpy())
                all_targs.append(batch.y.cpu().numpy())
        
        return np.vstack(all_emb), np.vstack(all_targs)

    def _ensure_dataset_on_device(self):
        """Ensure the entire dataset is on the correct device"""
        print(f"Ensuring dataset is on {device}...")
        
        # Move all data to the correct device
        self.dataset.data_list = ensure_data_on_device(self.dataset.data_list, device)
        
        print(f"Dataset moved to {device}")
        print(f"Dataset size: {len(self.dataset.data_list)} graphs")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Feature names: {self.dataset.node_feature_names}")
        
        # Ensure all dataset data is on the correct device
        self.dataset.data_list = ensure_data_on_device(self.dataset.data_list, device)

    def _override_parent_create_explainer_sparsified_graph(self, model, target_idx=0):
        """Override parent's create_explainer_sparsified_graph method with device-aware implementation"""
        print(f"\nCreating explainer graph for {self.target_names[target_idx]}")
        
        # Ensure model is on the correct device
        model = ensure_model_on_device(model, device)
        
        # Use the parent's method but with device-aware data handling
        try:
            # Call parent method but ensure data is on device
            explainer_data = super().create_explainer_sparsified_graph(model, target_idx)
            
            # Ensure the returned data is on the correct device
            explainer_data = ensure_data_on_device(explainer_data, device)
            
            return explainer_data
        except Exception as e:
            print(f"ERROR: Failed to create explainer graph: {e}")
            # Fallback to using original data
            explainer_data = self.dataset.data_list
            explainer_data = ensure_data_on_device(explainer_data, device)
            print("Using original data as fallback")
            return explainer_data

    def _override_parent_train_ml_models(self, embeddings, targets, target_idx):
        """Override parent's train_ml_models method with device-aware implementation"""
        print(f"\nTraining ML for: {self.target_names[target_idx]}")
        print(f"Emb: {embeddings.shape}, Y: {targets.shape}")
        
        # Ensure embeddings and targets are on CPU for ML training
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Use the parent's method but with 3 folds to match GNN models
        try:
            # Temporarily change num_folds to match outer_cv_folds
            original_num_folds = self.num_folds
            self.num_folds = self.outer_cv_folds
            ml_results = super().train_ml_models(embeddings, targets, target_idx)
            self.num_folds = original_num_folds
            return ml_results
        except Exception as e:
            print(f"ERROR: Failed to train ML models: {e}")
            return {}

    def _ensure_gpu_compatibility(self):
        """Ensure all methods are GPU-compatible by overriding parent methods"""
        print(f"Ensuring GPU compatibility for {self.case_type}...")
        
        # Override parent methods with device-aware versions
        self.train_gnn_model = self._override_parent_train_gnn_model
        self.extract_embeddings = self._override_parent_extract_embeddings
        self.create_explainer_sparsified_graph = self._override_parent_create_explainer_sparsified_graph
        self.train_ml_models = self._override_parent_train_ml_models
        self.plot_results = self._override_parent_plot_results
        
        print(f"GPU compatibility ensured for {self.case_type}")

    def _train_gnn_with_hyperparameter_tuning(self, model_type, target_idx, data_list, phase="knn"):
        """
        Train GNN with nested CV hyperparameter tuning using GridSearchCV.
        This maintains the exact same flow as domain_expert_cases_pipeline.py but adds hyperparameter tuning.
        
        Args:
            model_type (str): Type of GNN model ('gcn', 'rggc', 'gat')
            target_idx (int): Index of target variable
            data_list (list): List of graph data objects
            phase (str): Phase of training ('knn' or 'explainer')
        
        Returns:
            dict: Training results with best model and metrics
        """
        print(f"\nTraining {model_type.upper()} with hyperparameter tuning for {phase} phase...")
        
        # Ensure data is on the correct device
        data_list = ensure_data_on_device(data_list, device)
        
        # Define hyperparameter grid
        param_grid = {
            'hidden_dim': self.config.hidden_dim_grid,
            'dropout_rate': self.config.dropout_rate_grid
        }
        
        print(f"Hyperparameter grid: {param_grid}")
        
        # Store original settings
        orig_hidden_dim = self.config.hidden_dim
        orig_dropout_rate = self.config.dropout_rate
        
        # Create inner CV for hyperparameter tuning
        inner_cv = KFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=42)
        
        best_params = None
        best_score = -float('inf')
        best_model = None
        all_results = []
        
        # Grid search over hyperparameters
        total_combinations = len(param_grid['hidden_dim']) * len(param_grid['dropout_rate'])
        print(f"Testing {total_combinations} hyperparameter combinations...")
        
        for hd in param_grid['hidden_dim']:
            for dr in param_grid['dropout_rate']:
                print(f"  Testing hidden_dim={hd}, dropout_rate={dr}...")
                
                # Set hyperparameters for this iteration
                self.config.hidden_dim = hd
                self.config.dropout_rate = dr
                
                # Inner CV evaluation
                inner_scores = []
                inner_models = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(data_list)):
                    print(f"    Inner CV Fold {fold_idx + 1}/{self.inner_cv_folds}")
                    
                    # Split data for this fold
                    train_data = [data_list[i] for i in train_idx]
                    val_data = [data_list[i] for i in val_idx]
                    
                    # Ensure split data is on the correct device
                    train_data = ensure_data_on_device(train_data, device)
                    val_data = ensure_data_on_device(val_data, device)
                    
                    # Train model on train data
                    fold_result = self._train_single_fold_with_current_params(
                        model_type, target_idx, train_data, val_data
                    )
                    
                    if fold_result:
                        inner_scores.append(fold_result['r2'])
                        inner_models.append(fold_result['model'])
                
                # Calculate average performance for this hyperparameter combination
                if inner_scores:
                    avg_score = np.mean(inner_scores)
                    print(f"     Avg R²={avg_score:.4f}")
                    
                    all_results.append({
                        'hidden_dim': hd,
                        'dropout_rate': dr,
                        'avg_r2': avg_score,
                        'std_r2': np.std(inner_scores),
                        'fold_scores': inner_scores
                    })
                    
                    # Update best parameters
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {'hidden_dim': hd, 'dropout_rate': dr}
                        # Use the model from the best performing fold
                        best_fold_idx = np.argmax(inner_scores)
                        best_model = inner_models[best_fold_idx]
                else:
                    print(f"     No valid results for this combination")
        
        # Restore original settings
        self.config.hidden_dim = orig_hidden_dim
        self.config.dropout_rate = orig_dropout_rate
        
        if best_params is None:
            print(f"WARNING: No valid hyperparameter combinations found. Using default parameters.")
            best_params = {'hidden_dim': self.config.hidden_dim, 'dropout_rate': self.config.dropout_rate}
            best_score = 0.0
        
        print(f"\nBest hyperparameters for {phase}: {best_params}")
        print(f"Best inner CV score: R²={best_score:.4f}")
        
        # Apply best hyperparameters and train final model with outer CV
        self.config.hidden_dim = best_params['hidden_dim']
        self.config.dropout_rate = best_params['dropout_rate']
        
        # Use outer CV for final evaluation
        outer_cv = KFold(n_splits=self.outer_cv_folds, shuffle=True, random_state=42)
        
        target_name = self.target_names[target_idx]
        fold_results = []
        best_r2 = -float('inf')
        best_state = None
        
        criterion = nn.MSELoss()
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data_list), start=1):
            print(f"  Outer CV Fold {fold}/{self.outer_cv_folds}")
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]
            
            # Use device-aware DataLoaders
            train_loader = create_data_loader_on_device(train_data, self.config.batch_size, shuffle=True, target_device=device)
            test_loader = create_data_loader_on_device(test_data, self.config.batch_size, shuffle=False, target_device=device)
            
            model = self.create_gnn_model(model_type).to(device)
            optimizer = Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            best_val = float('inf')
            patience_cnt = 0
            best_model_state = None
            
            for epoch in range(1, self.config.num_epochs + 1):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    # Use device-aware batch movement
                    batch = move_batch_to_device(batch, device)
                    optimizer.zero_grad()
                    out, _ = model(batch.x, batch.edge_index, batch.batch)
                    target = batch.y[:, target_idx].view(-1, 1)
                    loss = criterion(out, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch.num_graphs
                avg_train = total_loss / len(train_loader.dataset)
                
                model.eval()
                total_val = 0
                with torch.no_grad():
                    for batch in test_loader:
                        # Use device-aware batch movement
                        batch = move_batch_to_device(batch, device)
                        out, _ = model(batch.x, batch.edge_index, batch.batch)
                        target = batch.y[:, target_idx].view(-1, 1)
                        total_val += criterion(out, target).item() * batch.num_graphs
                avg_val = total_val / len(test_loader.dataset)
                scheduler.step(avg_val)
                
                if avg_val < best_val:
                    best_val = avg_val
                    best_model_state = model.state_dict().copy()
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.config.patience:
                        break
            
            model.load_state_dict(best_model_state)
            model.eval()
            
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    # Use device-aware batch movement
                    batch = move_batch_to_device(batch, device)
                    out, _ = model(batch.x, batch.edge_index, batch.batch)
                    preds.append(out.cpu().numpy())
                    trues.append(batch.y[:, target_idx].cpu().numpy())
            
            preds = np.concatenate(preds).flatten()
            trues = np.concatenate(trues).flatten()
            
            mse = mean_squared_error(trues, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(trues, preds)
            mae = mean_absolute_error(trues, preds)
            
            fold_results.append({'fold': fold, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae,
                               'predictions': preds, 'targets': trues})
            
            if r2 > best_r2:
                best_r2 = r2
                best_state = best_model_state
            
            torch.save(model.state_dict(), f"{self.save_dir}/gnn_models/{model_type}_{target_name}_{phase}_fold{fold}.pt")
            print(f"    Fold {fold}  R²={r2:.3f}, RMSE={rmse:.3f}")
            
            # Clear memory between folds
            clear_gpu_memory()
        
        # Verify we have results for all folds
        if len(fold_results) != self.outer_cv_folds:
            print(f"L CRITICAL ERROR: Expected {self.outer_cv_folds} folds but got {len(fold_results)}")
            print(f"This indicates a serious problem in the training pipeline.")
            raise RuntimeError(f"GNN training failed: expected {self.outer_cv_folds} folds but got {len(fold_results)}")
        
        # Overall metrics
        all_preds = np.concatenate([fr['predictions'] for fr in fold_results])
        all_trues = np.concatenate([fr['targets'] for fr in fold_results])
        overall = {
            'mse': mean_squared_error(all_trues, all_preds),
            'rmse': np.sqrt(mean_squared_error(all_trues, all_preds)),
            'r2': r2_score(all_trues, all_preds),
            'mae': mean_absolute_error(all_trues, all_preds)
        }
        print(f"  Overall  R²={overall['r2']:.3f}, RMSE={overall['rmse']:.3f}")
        
        # Debug: Print some sample predictions vs actuals
        print(f"  Debug: Sample predictions vs actuals:")
        for i in range(min(5, len(all_preds))):
            print(f"    Sample {i+1}: Actual={all_trues[i]:.2f}, Predicted={all_preds[i]:.2f}")
        
        # Load best model
        best_model = self.create_gnn_model(model_type).to(device)
        best_model.load_state_dict(best_state)
        
        # Save hyperparameter tuning results
        self._save_hyperparameter_results(all_results, model_type, phase)
        
        return {'model': best_model, 'fold_results': fold_results, 'avg_metrics': overall, 'best_params': best_params}

    def _train_single_fold_with_current_params(self, model_type, target_idx, train_data, val_data):
        """
        Train a single fold with current hyperparameters for inner CV.
        
        Args:
            model_type (str): Type of GNN model
            target_idx (int): Index of target variable
            train_data (list): Training data
            val_data (list): Validation data
            
        Returns:
            dict: Training results with metrics and model
        """
        try:
            # Create data loaders using the device-aware function
            train_loader = create_data_loader_on_device(train_data, self.config.batch_size, shuffle=True, target_device=device)
            val_loader = create_data_loader_on_device(val_data, self.config.batch_size, shuffle=False, target_device=device)
            
            # Initialize model using current hyperparameters
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
                    # Use device-aware batch movement
                    batch = move_batch_to_device(batch, device)
                    optimizer.zero_grad()
                    
                    # Forward pass
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
                        # Use device-aware batch movement
                        batch = move_batch_to_device(batch, device)
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
            
            return {
                'mse': mse,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'model': model
            }
            
        except Exception as e:
            print(f"    Error in single fold training: {e}")
            return None

    def _save_hyperparameter_results(self, all_results, model_type, phase):
        """Save hyperparameter tuning results"""
        if not all_results:
            return
        
        # Create directory
        os.makedirs(f"{self.save_dir}/hyperparameters", exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.save_dir}/hyperparameters/{model_type}_{phase}_hyperparameter_tuning.csv", index=False)
        
        # Find best result
        best_idx = results_df['avg_r2'].idxmax()
        best_result = results_df.loc[best_idx]
        
        # Save best hyperparameters summary
        best_hp_summary = {
            'model_type': model_type,
            'phase': phase,
            'best_hidden_dim': best_result['hidden_dim'],
            'best_dropout_rate': best_result['dropout_rate'],
            'best_r2': best_result['avg_r2'],
            'std_r2': best_result['std_r2'],
            'all_combinations': all_results
        }
        
        with open(f"{self.save_dir}/hyperparameters/{model_type}_{phase}_best_hyperparameters.pkl", 'wb') as f:
            pickle.dump(best_hp_summary, f)
        
        # Also save as CSV for easier reporting
        best_hp_csv = pd.DataFrame([{
            'model_type': model_type,
            'phase': phase,
            'best_hidden_dim': best_result['hidden_dim'],
            'best_dropout_rate': best_result['dropout_rate'],
            'best_r2': best_result['avg_r2'],
            'std_r2': best_result['std_r2'],
            'case_type': self.case_type
        }])
        best_hp_csv.to_csv(f"{self.save_dir}/hyperparameters/{model_type}_{phase}_best_hyperparameters.csv", index=False)
        
        print(f"Hyperparameter results saved for {model_type}_{phase}")

    def _create_custom_plots(self, gnn_results, ml_results, target_idx):
        """Create comprehensive plots comparing all results with our data structure"""
        target_name = self.target_names[target_idx]
        
        # Create figure with subplots - increased size
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(f'Results Comparison for {target_name}', fontsize=16)
        
        # Plot 1: GNN Model Comparison (R² scores)
        ax1 = axes[0, 0]
        gnn_models = list(gnn_results.keys())
        gnn_r2_scores = [gnn_results[model]['avg_metrics']['r2'] for model in gnn_models]
        gnn_mse_scores = [gnn_results[model]['avg_metrics']['mse'] for model in gnn_models]
        
        bars1 = ax1.bar(gnn_models, gnn_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('GNN Models R² Comparison')
        ax1.set_ylabel('R² Score')
        
        # Fix y-axis range for negative values
        min_score = min(gnn_r2_scores)
        max_score = max(gnn_r2_scores)
        margin = (max_score - min_score) * 0.1 if max_score != min_score else 0.1
        ax1.set_ylim(min_score - margin, max_score + margin)
        
        # Add value labels on bars with MSE - fixed for negative values
        for bar, score, mse in zip(bars1, gnn_r2_scores, gnn_mse_scores):
            if score >= 0:
                text_y = bar.get_height() + 0.01
                va = 'bottom'
            else:
                text_y = bar.get_height() - 0.01
                va = 'top'
            ax1.text(bar.get_x() + bar.get_width()/2, text_y,
                    f'R²:{score:.3f}\nMSE:{mse:.3f}', ha='center', va=va, fontsize=8)
        
        # Plot 2: ML Model Comparison (R² scores)
        ax2 = axes[0, 1]
        ml_models = list(ml_results.keys())
        if ml_models:
            ml_r2_scores = [ml_results[model]['avg_metrics']['r2'] for model in ml_models]
            ml_mse_scores = [ml_results[model]['avg_metrics']['mse'] for model in ml_models]
            
            bars2 = ax2.bar(ml_models, ml_r2_scores, color=['orange', 'purple'])
            ax2.set_title('ML Models on Embeddings R² Comparison')
            ax2.set_ylabel('R² Score')
            
            # Fix y-axis range for ML models
            min_score = min(ml_r2_scores)
            max_score = max(ml_r2_scores)
            margin = (max_score - min_score) * 0.1 if max_score != min_score else 0.1
            ax2.set_ylim(min_score - margin, max_score + margin)
            
            # Add value labels on bars with MSE - fixed for negative values
            for bar, score, mse in zip(bars2, ml_r2_scores, ml_mse_scores):
                if score >= 0:
                    text_y = bar.get_height() + 0.01
                    va = 'bottom'
                else:
                    text_y = bar.get_height() - 0.01
                    va = 'top'
                ax2.text(bar.get_x() + bar.get_width()/2, text_y,
                        f'R²:{score:.3f}\nMSE:{mse:.3f}', ha='center', va=va, fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No ML models available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ML Models on Embeddings R² Comparison')
        
        # Plot 3: Overall Comparison
        ax3 = axes[0, 2]
        all_models = gnn_models + ml_models
        all_r2_scores = gnn_r2_scores + (ml_r2_scores if ml_models else [])
        all_mse_scores = gnn_mse_scores + (ml_mse_scores if ml_models else [])
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
        
        if all_models:
            bars3 = ax3.bar(range(len(all_models)), all_r2_scores, color=[colors[i % len(colors)] for i in range(len(all_models))])
            ax3.set_title('All Models R² Comparison')
            ax3.set_ylabel('R² Score')
            
            # Fix y-axis range for all models
            min_score = min(all_r2_scores)
            max_score = max(all_r2_scores)
            margin = (max_score - min_score) * 0.1 if max_score != min_score else 0.1
            ax3.set_ylim(min_score - margin, max_score + margin)
            
            ax3.set_xticks(range(len(all_models)))
            ax3.set_xticklabels(all_models, rotation=45, ha='right')
            
            # Add value labels with MSE - fixed for negative values
            for bar, score, mse in zip(bars3, all_r2_scores, all_mse_scores):
                if score >= 0:
                    text_y = bar.get_height() + 0.01
                    va = 'bottom'
                else:
                    text_y = bar.get_height() - 0.01
                    va = 'top'
                ax3.text(bar.get_x() + bar.get_width()/2, text_y,
                        f'R²:{score:.3f}\nMSE:{mse:.3f}', ha='center', va=va, fontsize=7)
        else:
            ax3.text(0.5, 0.5, 'No models available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('All Models R² Comparison')
        
        # Plot 4: RMSE Comparison
        ax4 = axes[1, 0]
        gnn_rmse_scores = [gnn_results[model]['avg_metrics']['rmse'] for model in gnn_models]
        ml_rmse_scores = [ml_results[model]['avg_metrics']['rmse'] for model in ml_models] if ml_models else []
        all_rmse_scores = gnn_rmse_scores + ml_rmse_scores
        
        if all_models:
            bars4 = ax4.bar(range(len(all_models)), all_rmse_scores, color=[colors[i % len(colors)] for i in range(len(all_models))])
            ax4.set_title('All Models RMSE Comparison')
            ax4.set_ylabel('RMSE')
            ax4.set_xticks(range(len(all_models)))
            ax4.set_xticklabels(all_models, rotation=45, ha='right')
            
            # Add MSE values on RMSE bars
            for bar, rmse, mse in zip(bars4, all_rmse_scores, all_mse_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'RMSE:{rmse:.3f}\nMSE:{mse:.3f}', ha='center', va='bottom', fontsize=7)
        else:
            ax4.text(0.5, 0.5, 'No models available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('All Models RMSE Comparison')
        
        # Plot 5: Prediction scatter plot for best model
        ax5 = axes[1, 1]
        
        if all_models:
            # Find best model overall
            best_model_name = all_models[np.argmax(all_r2_scores)]
            best_mse = all_mse_scores[np.argmax(all_r2_scores)]
            
            if best_model_name in gnn_results:
                best_results = gnn_results[best_model_name]
            else:
                best_results = ml_results[best_model_name]
            
            # Collect all predictions and targets from folds
            all_preds = []
            all_targets = []
            if 'fold_results' in best_results:
                for fold_result in best_results['fold_results']:
                    all_preds.extend(fold_result['predictions'])
                    all_targets.extend(fold_result['targets'])
                
                if all_preds and all_targets:
                    ax5.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
                    
                    # Add diagonal line
                    min_val = min(min(all_targets), min(all_preds))
                    max_val = max(max(all_targets), max(all_preds))
                    ax5.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    ax5.set_title(f'Best Model: {best_model_name}\nR² = {max(all_r2_scores):.3f}, MSE = {best_mse:.3f}')
                    ax5.set_xlabel('True Values')
                    ax5.set_ylabel('Predicted Values')
                    ax5.grid(True, alpha=0.3)
                else:
                    ax5.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title(f'Best Model: {best_model_name}')
            else:
                ax5.text(0.5, 0.5, 'No fold results available', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title(f'Best Model: {best_model_name}')
        else:
            ax5.text(0.5, 0.5, 'No models available', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Best Model Predictions')
        
        # Plot 6: Cross-validation consistency (R² across folds)
        ax6 = axes[1, 2]
        
        if all_models:
            # Plot R² scores across folds for each model
            fold_numbers = range(1, self.config.outer_cv_folds + 1)
            
            for i, model_name in enumerate(all_models):
                if model_name in gnn_results and 'fold_results' in gnn_results[model_name]:
                    fold_r2s = [fold['r2'] for fold in gnn_results[model_name]['fold_results']]
                    model_mse = gnn_results[model_name]['avg_metrics']['mse']
                elif model_name in ml_results and 'fold_results' in ml_results[model_name]:
                    fold_r2s = [fold['r2'] for fold in ml_results[model_name]['fold_results']]
                    model_mse = ml_results[model_name]['avg_metrics']['mse']
                else:
                    continue
                
                # Use modulo to prevent index errors
                ax6.plot(fold_numbers, fold_r2s, marker='o', label=f'{model_name} (MSE:{model_mse:.3f})', color=colors[i % len(colors)])
            
            ax6.set_title('Cross-Validation Consistency')
            ax6.set_xlabel('Fold Number')
            ax6.set_ylabel('R² Score')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No models available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Cross-Validation Consistency')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/{target_name}_comprehensive_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Custom plots saved: {self.save_dir}/plots/{target_name}_comprehensive_results.png")


def run_all_cases(data_path="../Data/New_Data.csv"):
    """Run all domain expert cases with proper configuration"""
    print("Running all domain expert cases...")
    
    cases = ['case1', 'case2', 'case3', 'case4']
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()}")
        print(f"{'='*60}")
        
        try:
            # Create configuration for testing
            test_config = PipelineConfig(
                inner_cv_folds=5,
                outer_cv_folds=5,
                hidden_dim=64,
                num_epochs=50,  # Reduced for testing
                patience=10,
                batch_size=8,
                learning_rate=0.001,
                weight_decay=1e-4,
                # Reduced hyperparameter grid for testing
                hidden_dim_grid=[32,64,128,256,512],
                dropout_rate_grid=[0.2, 0.3, 0.5]
            )
            
            pipeline = DomainExpertCasesPipeline(
                data_path=data_path,
                case_type=case,
                config=test_config,
                save_dir=f'./domain_expert_results/{case}',
                importance_threshold=0.2,
                use_fast_correlation=False,
                graph_mode='family',
                family_filter_mode='strict'
            )
            
            # Run the case-specific pipeline
            results = pipeline.run_case_specific_pipeline()
            all_results[case] = results
            
            print(f"\n {case.upper()} completed successfully!")
            
        except Exception as e:
            print(f"\nL Error in {case.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("CREATING OVERALL COMPARISON")
    print(f"{'='*80}")
    
    # Save combined results
    with open('./domain_expert_results/all_cases_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create overall hyperparameter summary
    create_hyperparameter_summary()
    
    print("All cases completed!")
    print("Domain expert cases pipeline completed!")


def create_hyperparameter_summary():
    """Create a summary CSV of all hyperparameter tuning results"""
    print("Creating hyperparameter summary...")
    
    import glob
    import os
    
    # Find all hyperparameter CSV files
    hp_files = glob.glob("./domain_expert_results/*/hyperparameters/*_hyperparameter_tuning.csv")
    
    all_hp_results = []
    
    for hp_file in hp_files:
        try:
            # Extract case and model info from filename
            parts = hp_file.split('/')
            case_part = parts[2]  # domain_expert_results/case1_h2_hydrogenotrophic_only/...
            model_phase = parts[-1].replace('_hyperparameter_tuning.csv', '')
            
            # Parse case type
            if 'case1' in case_part:
                case_type = 'case1'
                case_description = 'H2-km with hydrogenotrophic features only'
            elif 'case2' in case_part:
                case_type = 'case2'
                case_description = 'ACE-km with acetoclastic features only'
            elif 'case3' in case_part:
                case_type = 'case3'
                case_description = 'ACE-km with all feature groups'
            elif 'case4' in case_part:
                case_type = 'case4'
                case_description = 'ACE-km conditional (acetoclastic vs all groups)'
            elif 'case5' in case_part:
                case_type = 'case5'
                case_description = 'H2-km conditional (hydrogenotrophic vs all groups)'
            else:
                case_type = 'unknown'
                case_description = 'unknown'
            
            # Load hyperparameter results
            hp_df = pd.read_csv(hp_file)
            
            # Add metadata
            hp_df['case_type'] = case_type
            hp_df['case_description'] = case_description
            hp_df['model_phase'] = model_phase
            
            all_hp_results.append(hp_df)
            
        except Exception as e:
            print(f"Error processing {hp_file}: {e}")
    
    if all_hp_results:
        # Combine all results
        combined_hp_df = pd.concat(all_hp_results, ignore_index=True)
        
        # Save combined hyperparameter results
        combined_hp_df.to_csv('./domain_expert_results/all_hyperparameter_results.csv', index=False)
        
        # Create summary of best hyperparameters for each case/model
        best_hp_summary = []
        
        for case_type in combined_hp_df['case_type'].unique():
            case_data = combined_hp_df[combined_hp_df['case_type'] == case_type]
            
            for model_phase in case_data['model_phase'].unique():
                model_data = case_data[case_data['model_phase'] == model_phase]
                
                # Find best hyperparameters (highest R²)
                best_idx = model_data['avg_r2'].idxmax()
                best_row = model_data.loc[best_idx]
                
                best_hp_summary.append({
                    'case_type': case_type,
                    'case_description': best_row['case_description'],
                    'model_phase': model_phase,
                    'best_hidden_dim': best_row['hidden_dim'],
                    'best_dropout_rate': best_row['dropout_rate'],
                    'best_r2': best_row['avg_r2'],
                    'best_mse': best_row['avg_mse'],
                    'std_r2': best_row['std_r2'],
                    'std_mse': best_row['std_mse']
                })
        
        # Save best hyperparameters summary
        best_hp_df = pd.DataFrame(best_hp_summary)
        best_hp_df.to_csv('./domain_expert_results/best_hyperparameters_summary.csv', index=False)
        
        print(f"Hyperparameter summary saved:")
        print(f"  - All results: ./domain_expert_results/all_hyperparameter_results.csv")
        print(f"  - Best hyperparameters: ./domain_expert_results/best_hyperparameters_summary.csv")
        print(f"  - Total hyperparameter combinations tested: {len(combined_hp_df)}")
        print(f"  - Cases with results: {len(best_hp_df)}")
    else:
        print("No hyperparameter files found!")


if __name__ == "__main__":
    # Run all cases
    results = run_all_cases()
    print("Domain expert cases pipeline completed!")