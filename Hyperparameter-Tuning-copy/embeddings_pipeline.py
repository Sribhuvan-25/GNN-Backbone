# import os
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch import nn
# from torch.optim import Adam, lr_scheduler
# from torch_geometric.loader import DataLoader
# from sklearn.model_selection import KFold, ParameterGrid
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.svm import LinearSVR
# from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from tqdm import tqdm
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# # Import XGBoost and LightGBM
# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False
#     print("Warning: XGBoost not available. Install with: pip install xgboost")

# try:
#     import lightgbm as lgb
#     LIGHTGBM_AVAILABLE = True
# except ImportError:
#     LIGHTGBM_AVAILABLE = False
#     print("Warning: LightGBM not available. Install with: pip install lightgbm")

# # Import dataset and explainer modules (now from same directory)
# from dataset_regression import MicrobialGNNDataset
# from explainer_regression import GNNExplainerRegression
# from pipeline_explainer import create_explainer_sparsified_graph

# # Import the plus models that return embeddings (now from same directory)
# from GNNmodelsRegression import (
#     simple_GCN_res_plus_regression,
#     simple_RGGC_plus_regression,
#     simple_GAT_regression,
#     GaussianNLLLoss
# )

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# class MixedEmbeddingPipeline:
#     """
#     Complete pipeline for graph-based regression using GNN embeddings with ML models.
    
#     Pipeline Flow:
#     1. Generate graph from data
#     2. Create KNN graph sparsification
#     3. Train ALL GNN models (plus versions for embeddings) with nested CV hyperparameter tuning
#     4. Use best GNN model for GNNExplainer to get sparsified graph
#     5. Train ALL GNN models on sparsified graph with nested CV hyperparameter tuning
#     6. Extract embeddings from best overall GNN model
#     7. Train ML models (LinearSVR, ExtraTrees, RandomForest, XGBoost, LightGBM) on embeddings with 5-fold CV
#     """
    
#     def __init__(self, 
#                  data_path,
#                  k_neighbors=5,
#                  mantel_threshold=0.05,
#                  hidden_dim=64,
#                  dropout_rate=0.3,
#                  batch_size=8,
#                  learning_rate=0.001,
#                  weight_decay=1e-4,
#                  num_epochs=200,
#                  patience=20,
#                  num_folds=5,
#                  save_dir='./mixed_embedding_results',
#                  importance_threshold=0.2,  # Use default threshold - pipeline_explainer has adaptive thresholding
#                  use_fast_correlation=False,
#                  graph_mode='family',  # Changed default to family
#                  family_filter_mode='strict',
#                  use_enhanced_training=True,  # New parameter for enhanced training
#                  adaptive_hyperparameters=True,  # New parameter for adaptive hyperparameters
#                  use_nested_cv=True):  # New parameter for nested CV hyperparameter tuning
#         """
#         Initialize the mixed embedding pipeline
        
#         Args:
#             data_path: Path to the CSV file with data
#             k_neighbors: Number of neighbors for KNN graph sparsification
#             mantel_threshold: p-value threshold for Mantel test
#             hidden_dim: Hidden dimension size for GNN
#             dropout_rate: Dropout rate
#             batch_size: Batch size for training
#             learning_rate: Learning rate
#             weight_decay: Weight decay for optimizer
#             num_epochs: Maximum number of epochs
#             patience: Patience for early stopping
#             num_folds: Number of folds for cross-validation
#             save_dir: Directory to save results
#             importance_threshold: Threshold for edge importance in GNNExplainer sparsification
#             use_fast_correlation: If True, use fast correlation-based graph construction
#             graph_mode: Mode for graph construction ('otu' or 'family') - now defaults to 'family'
#             family_filter_mode: Mode for family filtering ('relaxed' or 'strict')
#             use_enhanced_training: If True, use enhanced training
#             adaptive_hyperparameters: If True, use adaptive hyperparameters
#             use_nested_cv: If True, use nested cross-validation for hyperparameter tuning
#         """
#         self.data_path = data_path
#         self.k_neighbors = k_neighbors
#         self.mantel_threshold = mantel_threshold
#         self.hidden_dim = hidden_dim
#         self.dropout_rate = dropout_rate
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.num_epochs = num_epochs
#         self.patience = patience
#         self.num_folds = num_folds
#         self.save_dir = save_dir
#         self.importance_threshold = importance_threshold
#         self.use_fast_correlation = use_fast_correlation
#         self.graph_mode = graph_mode
#         self.family_filter_mode = family_filter_mode
#         self.use_enhanced_training = use_enhanced_training
#         self.adaptive_hyperparameters = adaptive_hyperparameters
#         self.use_nested_cv = use_nested_cv
        
#         # Define hyperparameter search space for nested CV
#         self.gnn_hyperparams = {
#             'hidden_dim': [512, 256, 64],
#             'k_neighbors': [8, 10, 12]
#         }
#         self.param_grid = list(ParameterGrid(self.gnn_hyperparams))
        
#         # For explainer phase, we only tune hidden_dim since graph is already sparsified
#         self.explainer_hyperparams = {
#             'hidden_dim': [512, 256, 64]
#         }
#         self.explainer_param_grid = list(ParameterGrid(self.explainer_hyperparams))
        
#         # Always train all three models in mixed approach
#         self.gnn_models_to_train = ['gcn', 'rggc', 'gat']
#         print("Pipeline configured for MIXED model comparison")
        
#         if self.use_nested_cv:
#             print("Using NESTED CROSS-VALIDATION for hyperparameter tuning")
#             print(f"Hyperparameter search space: {len(self.param_grid)} combinations")
#         else:
#             print("Using standard cross-validation (no hyperparameter tuning)")
        
#         print(f"Using graph mode: {graph_mode} (family-level nodes)")
        
#         # Create comprehensive save directories (matching regression pipeline structure)
#         os.makedirs(save_dir, exist_ok=True)
#         os.makedirs(f"{save_dir}/gnn_models", exist_ok=True)
#         os.makedirs(f"{save_dir}/ml_models", exist_ok=True)
#         os.makedirs(f"{save_dir}/plots", exist_ok=True)
#         os.makedirs(f"{save_dir}/graphs", exist_ok=True)
#         os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
#         os.makedirs(f"{save_dir}/explanations", exist_ok=True)
#         os.makedirs(f"{save_dir}/detailed_results", exist_ok=True)
        
#         # Load and process data
#         print("\nLoading and processing data...")
#         self.dataset = MicrobialGNNDataset(
#             data_path=data_path,
#             k_neighbors=k_neighbors,
#             mantel_threshold=mantel_threshold,
#             use_fast_correlation=use_fast_correlation,
#             graph_mode=graph_mode,
#             family_filter_mode=family_filter_mode
#         )
        
#         # Get target names for reference
#         self.target_names = self.dataset.target_cols
#         print(f"Target variables: {self.target_names}")

#     def create_gnn_model(self, model_type, num_targets=1):
#         """Create a GNN plus model that returns embeddings"""
#         if model_type == 'gcn':
#             model = simple_GCN_res_plus_regression(
#                 hidden_channels=self.hidden_dim,
#                 output_dim=num_targets,
#                 dropout_prob=self.dropout_rate,
#                 input_channel=1,
#                 estimate_uncertainty=False
#             ).to(device)
#         elif model_type == 'rggc':
#             model = simple_RGGC_plus_regression(
#                 hidden_channels=self.hidden_dim,
#                 output_dim=num_targets,
#                 dropout_prob=self.dropout_rate,
#                 input_channel=1,
#                 estimate_uncertainty=False
#             ).to(device)
#         elif model_type == 'gat':
#             # Enhanced GAT with more heads and better architecture
#             model = simple_GAT_regression(
#                 hidden_channels=self.hidden_dim,
#                 output_dim=num_targets,
#                 dropout_prob=self.dropout_rate,
#                 input_channel=1,
#                 num_heads=8,  # Increased from 4 to 8 attention heads
#                 estimate_uncertainty=False
#             ).to(device)
#         # Add DGCNN option for dynamic graph construction
#         elif model_type == 'dgcnn':
#             try:
#                 from GNNmodelsRegression import Enhanced_DGCNN_regression
#                 model = Enhanced_DGCNN_regression(
#                     hidden_channels=self.hidden_dim,
#                     output_dim=num_targets,
#                     dropout_prob=self.dropout_rate,
#                     input_channel=1,
#                     k=min(self.k_neighbors, 10),  # Adaptive k-neighbors
#                     num_layers=4,
#                     estimate_uncertainty=False
#                 ).to(device)
#             except ImportError:
#                 print("Enhanced DGCNN not available, falling back to GAT")
#                 model = simple_GAT_regression(
#                     hidden_channels=self.hidden_dim,
#                     output_dim=num_targets,
#                     dropout_prob=self.dropout_rate,
#                     input_channel=1,
#                     num_heads=8,
#                     estimate_uncertainty=False
#                 ).to(device)
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")
        
#         return model

#     def _move_data_to_device(self, data_list):
#         """Move all data objects in the list to the correct device"""
#         device_data_list = []
#         for data in data_list:
#             device_data = data.to(device)
#             device_data_list.append(device_data)
#         return device_data_list

#     def _train_and_evaluate_once(self, model, data_list, train_idx, val_idx, target_idx, max_epochs=None):
#         """Train model on train_idx and evaluate on val_idx, returning R² score"""
#         if max_epochs is None:
#             max_epochs = min(self.num_epochs, 50)  # Shorter training for inner CV
        
#         # Ensure model is on the correct device
#         model = model.to(device)
        
#         # Split data and move to device
#         train_data = [data_list[i] for i in train_idx]
#         val_data = [data_list[i] for i in val_idx]
        
#         # Move all data to the correct device
#         train_data = self._move_data_to_device(train_data)
#         val_data = self._move_data_to_device(val_data)
        
#         # Create data loaders
#         batch_size = min(self.batch_size, len(train_data) // 4)
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
#         # Setup optimizer and scheduler
#         optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
        
#         criterion = nn.MSELoss()
        
#         # Training loop
#         best_val_loss = float('inf')
#         patience_counter = 0
        
#         for epoch in range(max_epochs):
#             # Training
#             model.train()
#             total_train_loss = 0
            
#             for batch_data in train_loader:
#                 # Data is already on device from _move_data_to_device
#                 optimizer.zero_grad()
                
#                 out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                 target = batch_data.y[:, target_idx].view(-1, 1)
                
#                 loss = criterion(out, target)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 total_train_loss += loss.item() * batch_data.num_graphs
            
#             # Validation
#             model.eval()
#             total_val_loss = 0
            
#             with torch.no_grad():
#                 for batch_data in val_loader:
#                     # Data is already on device from _move_data_to_device
#                     out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                     target = batch_data.y[:, target_idx].view(-1, 1)
#                     loss = criterion(out, target)
#                     total_val_loss += loss.item() * batch_data.num_graphs
            
#             avg_val_loss = total_val_loss / len(val_loader.dataset)
#             scheduler.step(avg_val_loss)
            
#             # Early stopping
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= min(self.patience, 10):  # Shorter patience for inner CV
#                     break
        
#         # Final evaluation
#         model.eval()
#         all_preds = []
#         all_targets = []
        
#         with torch.no_grad():
#             for batch_data in val_loader:
#                 # Data is already on device from _move_data_to_device
#                 out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                 target = batch_data.y[:, target_idx].view(-1, 1)
                
#                 all_preds.append(out.cpu().numpy())
#                 all_targets.append(target.cpu().numpy())
        
#         all_preds = np.vstack(all_preds).flatten()
#         all_targets = np.vstack(all_targets).flatten()
        
#         r2 = r2_score(all_targets, all_preds)
#         return r2

#     def _inner_loop_select(self, model_type, train_data, target_idx):
#         """Inner loop for hyperparameter selection using K-fold CV"""
#         inner_kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
#         best_score = -float('inf')
#         best_params = None
#         all_combinations_results = []
        
#         # FIXED: Proper explainer phase detection using data source comparison
#         # Check if this is explainer phase by comparing data source
#         is_explainer_phase = (train_data is not self.dataset.data_list)
        
#         # Use appropriate parameter grid based on phase
#         if is_explainer_phase:
#             param_grid = self.explainer_param_grid
#             print(f"    Inner CV (Explainer Phase): Testing {len(param_grid)} hidden_dim combinations...")
#         else:
#             param_grid = self.param_grid
#             print(f"    Inner CV (KNN Phase): Testing {len(param_grid)} hyperparameter combinations...")
        
#         print(f"    {'Combination':<15} {'hidden_dim':<12} {'k_neighbors':<12} {'Mean R²':<10} {'Std R²':<10} {'Best':<6}")
#         print(f"    {'-'*75}")
        
#         for i, params in enumerate(param_grid):
#             # FIXED: Create local copies of hyperparameters instead of modifying global state
#             local_hidden_dim = params['hidden_dim']
#             local_k_neighbors = params.get('k_neighbors', self.k_neighbors)
            
#             # FIXED: For explainer phase, use existing sparsified graph data
#             if is_explainer_phase:
#                 temp_data_list = train_data
#             else:
#                 # FIXED: For k_neighbors tuning, recreate dataset with local parameters
#                 if 'k_neighbors' in params:
#                     # Create a temporary dataset with the local k_neighbors value
#                     temp_dataset = MicrobialGNNDataset(
#                         data_path=self.data_path,
#                         k_neighbors=local_k_neighbors,  # Use local parameter
#                         mantel_threshold=self.mantel_threshold,
#                         use_fast_correlation=self.use_fast_correlation,
#                         graph_mode=self.graph_mode,
#                         family_filter_mode=self.family_filter_mode
#                     )
#                     # Move all data to the correct device
#                     temp_data_list = self._move_data_to_device(temp_dataset.data_list)
#                 else:
#                     temp_data_list = train_data
            
#             val_scores = []
            
#             for tr_idx, val_idx in inner_kf.split(temp_data_list):
#                 # FIXED: Create model with local hyperparameters
#                 model = self._create_gnn_model_with_params(model_type, local_hidden_dim, num_targets=1)
#                 r2_score = self._train_and_evaluate_once_with_params(
#                     model, temp_data_list, tr_idx, val_idx, target_idx, 
#                     hidden_dim=local_hidden_dim
#                 )
#                 val_scores.append(r2_score)
            
#             mean_val = np.mean(val_scores)
#             std_val = np.std(val_scores)
            
#             # Store results for this combination
#             combination_result = {
#                 'combination': i + 1,
#                 'params': params.copy(),
#                 'mean_r2': mean_val,
#                 'std_r2': std_val,
#                 'val_scores': val_scores,
#                 'is_best': False
#             }
#             all_combinations_results.append(combination_result)
            
#             if mean_val > best_score:
#                 best_score = mean_val
#                 best_params = params.copy()
#                 # Mark this as best
#                 combination_result['is_best'] = True
            
#             # Print progress for this combination
#             best_marker = "" if combination_result['is_best'] else " "
#             k_neighbors_val = params.get('k_neighbors', 'N/A') if not is_explainer_phase else 'Fixed'
#             print(f"    {i+1:<15} {params['hidden_dim']:<12} {k_neighbors_val:<12} {mean_val:<10.4f} {std_val:<10.4f} {best_marker:<6}")
        
#         print(f"    {'-'*75}")
#         print(f"    Best hyperparameters: {best_params} (R² = {best_score:.4f})")
        
#         # Print detailed results for top 3 combinations
#         print(f"\n    Top 3 Hyperparameter Combinations:")
#         sorted_results = sorted(all_combinations_results, key=lambda x: x['mean_r2'], reverse=True)
#         for i, result in enumerate(sorted_results[:3]):
#             params = result['params']
#             mean_r2 = result['mean_r2']
#             std_r2 = result['std_r2']
#             val_scores = result['val_scores']
#             rank = i + 1
            
#             print(f"    {rank}. hidden_dim={params['hidden_dim']}, k_neighbors={params.get('k_neighbors', 'Fixed')}")
#             print(f"       R² = {mean_r2:.4f} ± {std_r2:.4f}")
#             print(f"       Individual fold R² scores: {[f'{score:.4f}' for score in val_scores]}")
        
#         return best_params

#     def _train_model_full_with_params(self, model, train_data, target_idx, hidden_dim=None):
#         """Train model on full training data with specific hyperparameters"""
#         # Ensure model is on the correct device
#         model = model.to(device)
        
#         # Move all data to the correct device
#         train_data = self._move_data_to_device(train_data)
        
#         train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
#         optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
#         criterion = nn.MSELoss()
        
#         best_val_loss = float('inf')
#         best_model_state = None
#         patience_counter = 0
        
#         for epoch in range(self.num_epochs):
#             # Training
#             model.train()
#             total_train_loss = 0
            
#             for batch_data in train_loader:
#                 # Data is already on device from _move_data_to_device
#                 optimizer.zero_grad()
                
#                 out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                 target = batch_data.y[:, target_idx].view(-1, 1)
                
#                 loss = criterion(out, target)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 total_train_loss += loss.item() * batch_data.num_graphs
            
#             avg_train_loss = total_train_loss / len(train_loader.dataset)
#             scheduler.step(avg_train_loss)
            
#             # Early stopping based on training loss (no validation set in full training)
#             if avg_train_loss < best_val_loss:
#                 best_val_loss = avg_train_loss
#                 best_model_state = model.state_dict().copy()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= self.patience:
#                     break
        
#         # Load best model state
#         model.load_state_dict(best_model_state)
#         return model

#     def _evaluate_model(self, model, test_data, target_idx):
#         """Evaluate model on test data and return MSE and R²"""
#         # Ensure model is on the correct device
#         model = model.to(device)
        
#         # Move all data to the correct device
#         test_data = self._move_data_to_device(test_data)
        
#         test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
#         model.eval()
#         all_preds = []
#         all_targets = []
        
#         with torch.no_grad():
#             for batch_data in test_loader:
#                 # Data is already on device from _move_data_to_device
#                 out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                 target = batch_data.y[:, target_idx].view(-1, 1)
                
#                 all_preds.append(out.cpu().numpy())
#                 all_targets.append(target.cpu().numpy())
        
#         all_preds = np.vstack(all_preds).flatten()
#         all_targets = np.vstack(all_targets).flatten()
        
#         mse = mean_squared_error(all_targets, all_preds)
#         r2 = r2_score(all_targets, all_preds)
        
#         return mse, r2

#     def train_gnn_model_nested(self, model_type, target_idx, data_list=None):
#         """Train GNN model with nested cross-validation for hyperparameter tuning"""
#         if data_list is None:
#             data_list = self.dataset.data_list
        
#         target_name = self.target_names[target_idx]
#         phase = "explainer" if data_list != self.dataset.data_list else "knn"
#         print(f"\nTraining {model_type.upper()} model with NESTED CV for target: {target_name} ({phase} graph)")
        
#         outer_kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
#         outer_results = []
#         best_model_state = None
#         best_outer_r2 = -float('inf')
#         best_hyperparams = None  # Store the best hyperparameters
        
#         print(f"\n{'='*60}")
#         print(f"NESTED CV RESULTS FOR {model_type.upper()} - {target_name}")
#         print(f"{'='*60}")
        
#         for fold, (train_idx, test_idx) in enumerate(outer_kf.split(data_list)):
#             fold_num = fold + 1
#             print(f"\n  {'-'*50}")
#             print(f"  OUTER FOLD {fold_num}/{self.num_folds}")
#             print(f"  {'-'*50}")
            
#             train_data = [data_list[i] for i in train_idx]
#             test_data = [data_list[i] for i in test_idx]
            
#             # 1. Inner loop: pick hyperparameters
#             print(f"  Inner CV Hyperparameter Selection:")
#             best_params = self._inner_loop_select(model_type, train_data, target_idx)
            
#             # 2. FIXED: Use best hyperparameters without modifying global state
#             best_hidden_dim = best_params['hidden_dim']
#             best_k_neighbors = best_params.get('k_neighbors', self.k_neighbors)
            
#             # FIXED: Ensure data consistency between inner and outer loops
#             if 'k_neighbors' in best_params and not (data_list != self.dataset.data_list):
#                 # Only recreate dataset if we're in KNN phase (not explainer phase)
#                 print(f"    Recreating dataset with best k_neighbors={best_k_neighbors}")
#                 best_dataset = MicrobialGNNDataset(
#                     data_path=self.data_path,
#                     k_neighbors=best_k_neighbors,
#                     mantel_threshold=self.mantel_threshold,
#                     use_fast_correlation=self.use_fast_correlation,
#                     graph_mode=self.graph_mode,
#                     family_filter_mode=self.family_filter_mode
#                 )
#                 # Move all data to the correct device
#                 best_dataset_data_list = self._move_data_to_device(best_dataset.data_list)
#                 # Use the same indices as the original split
#                 best_train_data = [best_dataset_data_list[i] for i in train_idx]
#                 best_test_data = [best_dataset_data_list[i] for i in test_idx]
#             else:
#                 # For explainer phase or when k_neighbors wasn't tuned, use existing data
#                 best_train_data = train_data
#                 best_test_data = test_data
            
#             # FIXED: Create model with best hyperparameters without modifying global state
#             model = self._create_gnn_model_with_params(model_type, best_hidden_dim, num_targets=1)
#             model = self._train_model_full_with_params(model, best_train_data, target_idx, best_hidden_dim)
            
#             # 3. Evaluate on test_data
#             mse, r2 = self._evaluate_model(model, best_test_data, target_idx)
#             print(f"  Final Evaluation on Test Set:")
#             print(f"    R² = {r2:.4f}, MSE = {mse:.4f}")
            
#             # Get predictions and targets for this fold
#             model.eval()
#             all_preds = []
#             all_targets = []
            
#             test_loader = DataLoader(best_test_data, batch_size=self.batch_size, shuffle=False)
#             with torch.no_grad():
#                 for batch_data in test_loader:
#                     batch_data = batch_data.to(device)
#                     out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                     target = batch_data.y[:, target_idx].view(-1, 1)
                    
#                     all_preds.append(out.cpu().numpy())
#                     all_targets.append(target.cpu().numpy())
            
#             all_preds = np.vstack(all_preds).flatten()
#             all_targets = np.vstack(all_targets).flatten()
            
#             # Calculate additional metrics
#             rmse = np.sqrt(mse)
#             mae = mean_absolute_error(all_targets, all_preds)
            
#             outer_results.append({
#                 'fold': fold_num,
#                 'r2': r2,
#                 'mse': mse,
#                 'rmse': rmse,
#                 'mae': mae,
#                 'best_params': best_params,
#                 'train_size': len(train_data),
#                 'test_size': len(test_data),
#                 'predictions': all_preds,
#                 'targets': all_targets
#             })
            
#             if r2 > best_outer_r2:
#                 best_outer_r2 = r2
#                 best_model_state = model.state_dict().copy()
#                 best_hyperparams = best_params.copy()  # Store the best hyperparameters
        
#         # 4. Comprehensive Summary
#         print(f"\n{'='*60}")
#         print(f"COMPREHENSIVE NESTED CV SUMMARY")
#         print(f"{'='*60}")
        
#         r2_scores = [r['r2'] for r in outer_results]
#         mse_scores = [r['mse'] for r in outer_results]
        
#         avg_r2 = np.mean(r2_scores)
#         avg_mse = np.mean(mse_scores)
#         std_r2 = np.std(r2_scores)
#         std_mse = np.std(mse_scores)
        
#         print(f"Model: {model_type.upper()}")
#         print(f"Target: {target_name}")
#         print(f"Phase: {phase}")
#         print(f"Number of outer folds: {self.num_folds}")
        
#         print(f"\nFold-by-Fold Results:")
#         print(f"{'Fold':<6} {'R²':<10} {'MSE':<10} {'Best hidden_dim':<15} {'Best k_neighbors':<15}")
#         print(f"{'-'*60}")
        
#         for result in outer_results:
#             fold = result['fold']
#             r2 = result['r2']
#             mse = result['mse']
#             best_params = result['best_params']
#             hidden_dim = best_params.get('hidden_dim', 'N/A')
#             k_neighbors = best_params.get('k_neighbors', 'N/A')
#             print(f"{fold:<6} {r2:<10.4f} {mse:<10.4f} {hidden_dim:<15} {k_neighbors:<15}")
        
#         print(f"\nOverall Performance:")
#         print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
#         print(f"  MSE = {avg_mse:.4f} ± {std_mse:.4f}")
#         print(f"  RMSE = {np.sqrt(avg_mse):.4f}")
        
#         # Show hyperparameter selection frequency
#         print(f"\nHyperparameter Selection Frequency:")
#         hidden_dims = [r['best_params'].get('hidden_dim', 'N/A') for r in outer_results]
#         k_neighbors_list = [r['best_params'].get('k_neighbors', 'N/A') for r in outer_results]
        
#         from collections import Counter
#         hidden_dim_counts = Counter(hidden_dims)
#         k_neighbors_counts = Counter(k_neighbors_list)
        
#         print(f"  hidden_dim selections:")
#         for dim, count in hidden_dim_counts.items():
#             print(f"    {dim}: {count}/{self.num_folds} folds")
        
#         print(f"  k_neighbors selections:")
#         for k, count in k_neighbors_counts.items():
#             print(f"    {k}: {count}/{self.num_folds} folds")
        
#         # FIXED: Create final model with best hyperparameters without modifying global state
#         if best_hyperparams is not None:
#             best_hidden_dim = best_hyperparams['hidden_dim']
#         else:
#             best_hidden_dim = self.hidden_dim
        
#         best_model = self._create_gnn_model_with_params(model_type, best_hidden_dim, num_targets=1)
#         best_model.load_state_dict(best_model_state)
        
#         # Save detailed metrics including hyperparameter selection
#         self.save_detailed_metrics(outer_results, model_type, target_name, phase)
        
#         return {
#             'model': best_model,
#             'fold_results': outer_results,
#             'avg_metrics': {
#                 'r2': avg_r2,
#                 'mse': avg_mse,
#                 'rmse': np.sqrt(avg_mse),
#                 'mae': np.mean([r.get('mae', 0) for r in outer_results]),
#                 'std_r2': std_r2,
#                 'std_mse': std_mse
#             },
#             'model_type': model_type,
#             'target_name': target_name,
#             'target_idx': target_idx,
#             'phase': phase
#         }

#     def train_gnn_model(self, model_type, target_idx, data_list=None):
#         """Original GNN training method (kept for backward compatibility)"""
#         if self.use_nested_cv:
#             return self.train_gnn_model_nested(model_type, target_idx, data_list)
        
#         # Original training method (no hyperparameter tuning)
#         if data_list is None:
#             data_list = self.dataset.data_list
        
#         target_name = self.target_names[target_idx]
#         phase = "explainer" if data_list != self.dataset.data_list else "knn"
#         print(f"\nTraining {model_type.upper()} model for target: {target_name} ({phase} graph)")
        
#         # Setup k-fold cross-validation
#         kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
#         fold_results = []
#         best_model = None
#         best_r2 = -float('inf')
        
#         criterion = nn.MSELoss()
        
#         # Iterate through folds
#         for fold, (train_index, test_index) in enumerate(kf.split(data_list)):
#             fold_num = fold + 1
#             print(f"  Fold {fold_num}/{self.num_folds}")
            
#             # Split into train and test sets
#             train_dataset = [data_list[i] for i in train_index]
#             test_dataset = [data_list[i] for i in test_index]
            
#             # Create data loaders
#             train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#             test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
#             # Initialize model
#             model = self.create_gnn_model(model_type, num_targets=1)
            
#             # Setup optimizer and scheduler
#             optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#             scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
            
#             # Training loop
#             best_val_loss = float('inf')
#             best_model_state = None
#             patience_counter = 0
#             train_losses = []
#             val_losses = []
            
#             for epoch in range(self.num_epochs):
#                 # Training
#                 model.train()
#                 total_train_loss = 0
                
#                 for batch_data in train_loader:
#                     batch_data = batch_data.to(device)
#                     optimizer.zero_grad()
                    
#                     # Forward pass - get predictions and embeddings
#                     out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    
#                     # Extract target for this specific target_idx
#                     target = batch_data.y[:, target_idx].view(-1, 1)
                    
#                     loss = criterion(out, target)
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
#                     optimizer.step()
                    
#                     total_train_loss += loss.item() * batch_data.num_graphs
                
#                 avg_train_loss = total_train_loss / len(train_loader.dataset)
#                 train_losses.append(avg_train_loss)
                
#                 # Validation
#                 model.eval()
#                 total_val_loss = 0
                
#                 with torch.no_grad():
#                     for batch_data in test_loader:
#                         batch_data = batch_data.to(device)
#                         out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                         target = batch_data.y[:, target_idx].view(-1, 1)
#                         loss = criterion(out, target)
#                         total_val_loss += loss.item() * batch_data.num_graphs
                
#                 avg_val_loss = total_val_loss / len(test_loader.dataset)
#                 val_losses.append(avg_val_loss)
#                 scheduler.step(avg_val_loss)
                
#                 # Print progress
#                 if epoch % 20 == 0 or epoch == 1 or epoch == self.num_epochs - 1:
#                     print(f"    Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                
#                 # Early stopping
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     best_model_state = model.state_dict().copy()
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= self.patience:
#                         print(f"    Early stopping at epoch {epoch+1}")
#                         break
            
#             # Load best model for evaluation
#             model.load_state_dict(best_model_state)
            
#             # Final evaluation
#             model.eval()
#             all_preds = []
#             all_targets = []
            
#             with torch.no_grad():
#                 for batch_data in test_loader:
#                     batch_data = batch_data.to(device)
#                     out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                     target = batch_data.y[:, target_idx].view(-1, 1)
                    
#                     all_preds.append(out.cpu().numpy())
#                     all_targets.append(target.cpu().numpy())
            
#             # Calculate metrics
#             all_preds = np.vstack(all_preds).flatten()
#             all_targets = np.vstack(all_targets).flatten()
            
#             mse = mean_squared_error(all_targets, all_preds)
#             rmse = np.sqrt(mse)
#             r2 = r2_score(all_targets, all_preds)
#             mae = mean_absolute_error(all_targets, all_preds)
            
#             # Save model
#             model_path = f"{self.save_dir}/gnn_models/{model_type}_{target_name}_fold{fold_num}_{phase}.pt"
#             torch.save(model.state_dict(), model_path)
            
#             fold_results.append({
#                 'fold': fold_num,
#                 'mse': mse,
#                 'rmse': rmse,
#                 'r2': r2,
#                 'mae': mae,
#                 'predictions': all_preds,
#                 'targets': all_targets,
#                 'train_losses': train_losses,
#                 'val_losses': val_losses,
#                 'model_path': model_path
#             })
            
#             # Keep track of best model across folds
#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model = model.state_dict().copy()
            
#             print(f"    MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
#         # Calculate overall metrics from all validation samples combined
#         all_fold_preds = []
#         all_fold_targets = []
#         for fold_result in fold_results:
#             all_fold_preds.extend(fold_result['predictions'])
#             all_fold_targets.extend(fold_result['targets'])
        
#         all_fold_preds = np.array(all_fold_preds)
#         all_fold_targets = np.array(all_fold_targets)
        
#         # Calculate overall metrics (replaces avg_metrics)
#         overall_metrics = {
#             'mse': mean_squared_error(all_fold_targets, all_fold_preds),
#             'rmse': np.sqrt(mean_squared_error(all_fold_targets, all_fold_preds)),
#             'r2': r2_score(all_fold_targets, all_fold_preds),
#             'mae': mean_absolute_error(all_fold_targets, all_fold_preds)
#         }
        
#         print(f"  Overall - MSE: {overall_metrics['mse']:.4f}, RMSE: {overall_metrics['rmse']:.4f}, R²: {overall_metrics['r2']:.4f}, MAE: {overall_metrics['mae']:.4f}")
        
#         # Create overall plots (only overall, no individual fold plots)
#         self.plot_overall_gnn_results(fold_results, model_type, target_name, phase)
        
#         # Save detailed metrics
#         self.save_detailed_metrics(fold_results, model_type, target_name, phase)
        
#         # Create final model with best weights
#         final_model = self.create_gnn_model(model_type, num_targets=1)
#         final_model.load_state_dict(best_model)
        
#         return {
#             'model': final_model,
#             'fold_results': fold_results,
#             'avg_metrics': overall_metrics,  # Now contains overall metrics instead of averages
#             'model_type': model_type,
#             'target_name': target_name,
#             'target_idx': target_idx,
#             'phase': phase
#         }

#     def extract_embeddings(self, model, data_list):
#         """Extract embeddings from trained GNN model"""
#         model.eval()
#         all_embeddings = []
#         all_targets = []
        
#         # Create data loader for all data
#         data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        
#         with torch.no_grad():
#             for batch_data in data_loader:
#                 batch_data = batch_data.to(device)
                
#                 # Forward pass to get embeddings
#                 out, embeddings = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                
#                 all_embeddings.append(embeddings.cpu().numpy())
#                 all_targets.append(batch_data.y.cpu().numpy())
        
#         # Concatenate all embeddings and targets
#         embeddings = np.vstack(all_embeddings)
#         targets = np.vstack(all_targets)
        
#         return embeddings, targets

#     def train_ml_models(self, embeddings, targets, target_idx):
#         """Train ML models (LinearSVR, ExtraTrees, XGBoost, RandomForest, LightGBM) on embeddings with 5-fold CV"""
#         target_name = self.target_names[target_idx]
#         target_values = targets[:, target_idx]
        
#         print(f"\nTraining ML models on embeddings for target: {target_name}")
#         print(f"Embedding shape: {embeddings.shape}, Target shape: {target_values.shape}")
        
#         # Define ML models with preprocessing pipelines
#         ml_models = {
#             'LinearSVR': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', LinearSVR(epsilon=0.1, tol=1e-4, C=1.0, max_iter=10000))
#             ]),
#             'ExtraTrees': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
#             ]),
#             'RandomForest': Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10))
#             ])
#         }
        
#         # Add XGBoost if available
#         if XGBOOST_AVAILABLE:
#             ml_models['XGBoost'] = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', xgb.XGBRegressor(
#                     n_estimators=100,
#                     max_depth=6,
#                     learning_rate=0.1,
#                     random_state=42,
#                     n_jobs=-1,
#                     verbosity=0
#                 ))
#             ])
        
#         # Add LightGBM if available
#         if LIGHTGBM_AVAILABLE:
#             ml_models['LightGBM'] = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', lgb.LGBMRegressor(
#                     n_estimators=100,
#                     max_depth=6,
#                     learning_rate=0.1,
#                     random_state=42,
#                     n_jobs=-1,
#                     verbosity=-1
#                 ))
#             ])
        
#         print(f"Training {len(ml_models)} ML models: {list(ml_models.keys())}")
        
#         # Setup k-fold cross-validation
#         kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
#         ml_results = {}
        
#         for model_name, model_pipeline in ml_models.items():
#             print(f"\n  Training {model_name}...")
#             fold_results = []
            
#             for fold, (train_index, test_index) in enumerate(kf.split(embeddings)):
#                 fold_num = fold + 1
                
#                 # Split data
#                 X_train, X_test = embeddings[train_index], embeddings[test_index]
#                 y_train, y_test = target_values[train_index], target_values[test_index]
                
#                 # Train model
#                 model_pipeline.fit(X_train, y_train)
                
#                 # Predict
#                 y_pred = model_pipeline.predict(X_test)
                
#                 # Calculate metrics
#                 mse = mean_squared_error(y_test, y_pred)
#                 rmse = np.sqrt(mse)
#                 r2 = r2_score(y_test, y_pred)
#                 mae = mean_absolute_error(y_test, y_pred)
                
#                 fold_results.append({
#                     'fold': fold_num,
#                     'mse': mse,
#                     'rmse': rmse,
#                     'r2': r2,
#                     'mae': mae,
#                     'predictions': y_pred,
#                     'targets': y_test
#                 })
                
#                 print(f"    Fold {fold_num}: MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            
#             # Calculate overall metrics from all validation samples combined
#             all_fold_preds = []
#             all_fold_targets = []
#             for fold_result in fold_results:
#                 all_fold_preds.extend(fold_result['predictions'])
#                 all_fold_targets.extend(fold_result['targets'])
            
#             all_fold_preds = np.array(all_fold_preds)
#             all_fold_targets = np.array(all_fold_targets)
            
#             # Calculate fold-wise metrics for mean ± std
#             fold_mse_scores = [fold_result['mse'] for fold_result in fold_results]
#             fold_rmse_scores = [fold_result['rmse'] for fold_result in fold_results]
#             fold_r2_scores = [fold_result['r2'] for fold_result in fold_results]
#             fold_mae_scores = [fold_result['mae'] for fold_result in fold_results]
            
#             # Calculate mean ± std
#             avg_metrics = {
#                 'mse': np.mean(fold_mse_scores),
#                 'rmse': np.mean(fold_rmse_scores),
#                 'r2': np.mean(fold_r2_scores),
#                 'mae': np.mean(fold_mae_scores),
#                 'std_mse': np.std(fold_mse_scores),
#                 'std_rmse': np.std(fold_rmse_scores),
#                 'std_r2': np.std(fold_r2_scores),
#                 'std_mae': np.std(fold_mae_scores)
#             }
            
#             print(f"    Overall - MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['std_mse']:.4f}, RMSE: {avg_metrics['rmse']:.4f} ± {avg_metrics['std_rmse']:.4f}, R²: {avg_metrics['r2']:.4f} ± {avg_metrics['std_r2']:.4f}, MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['std_mae']:.4f}")
            
#             # Train final model on all data
#             final_model = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('regressor', ml_models[model_name]['regressor'])
#             ])
#             final_model.fit(embeddings, target_values)
            
#             ml_results[model_name] = {
#                 'model': final_model,
#                 'fold_results': fold_results,
#                 'avg_metrics': avg_metrics,  # Now contains mean ± std metrics
#                 'target_name': target_name,
#                 'target_idx': target_idx
#             }
        
#         return ml_results

#     def create_explainer_sparsified_graph(self, model, target_idx=0):
#         """Create sparsified graph using GNNExplainer - uses existing pipeline_explainer function"""
#         print(f"\nCreating GNNExplainer sparsified graph for target: {self.target_names[target_idx]}")
#         print(f"Using importance threshold: {self.importance_threshold}")
        
#         # Use the existing function from pipeline_explainer.py
#         sparsified_data_list = create_explainer_sparsified_graph(
#             pipeline=self,  # Pass self as pipeline
#             model=model,
#             target_idx=target_idx,
#             importance_threshold=self.importance_threshold
#         )
        
#         print(f"GNNExplainer sparsification complete: {len(sparsified_data_list)} samples created")
        
#         return sparsified_data_list

#     def plot_results(self, gnn_results, ml_results, target_idx):
#         """Create simple prediction vs actual plot for best ML model"""
#         target_name = self.target_names[target_idx]
        
#         # Find best ML model
#         ml_models = list(ml_results.keys())
#         ml_r2_scores = [ml_results[model]['avg_metrics']['r2'] for model in ml_models]
#         best_ml_model = ml_models[np.argmax(ml_r2_scores)]
#         best_r2 = max(ml_r2_scores)
#         best_mse = ml_results[best_ml_model]['avg_metrics']['mse']
        
#         print(f"\nBest ML model: {best_ml_model} (R² = {best_r2:.4f}, MSE = {best_mse:.4f})")
        
#         # Create simple prediction vs actual plot
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
#         # Collect all predictions and targets from folds
#         all_preds = []
#         all_targets = []
#         for fold_result in ml_results[best_ml_model]['fold_results']:
#             all_preds.extend(fold_result['predictions'])
#             all_targets.extend(fold_result['targets'])
        
#         # Create scatter plot
#         ax.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none', s=50)
        
#         # Add diagonal line
#         min_val = min(min(all_targets), min(all_preds))
#         max_val = max(max(all_targets), max(all_preds))
#         ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
#         ax.set_title(f'Best ML Model: {best_ml_model}\nR² = {best_r2:.4f}, MSE = {best_mse:.4f}')
#         ax.set_xlabel('True Values')
#         ax.set_ylabel('Predicted Values')
#         ax.grid(True, alpha=0.3)
        
#         # Add R² text
#         ax.text(0.05, 0.95, f'R² = {best_r2:.4f}', transform=ax.transAxes, 
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
#                 verticalalignment='top')
        
#         plt.tight_layout()
#         plt.savefig(f"{self.save_dir}/plots/{target_name}_best_ml_pred_vs_actual.png", 
#                    dpi=150, bbox_inches='tight')
#         plt.close()
        
#         print(f"Best ML model plot saved: {self.save_dir}/plots/{target_name}_best_ml_pred_vs_actual.png")

#     def save_results(self, all_results):
#         """Save all results to files"""
#         print(f"\nSaving results to {self.save_dir}")
        
#         # Save results as pickle
#         with open(f"{self.save_dir}/all_results.pkl", 'wb') as f:
#             pickle.dump(all_results, f)
        
#         # Save summary as CSV
#         summary_data = []
        
#         for target_name, target_results in all_results.items():
#             if target_name == 'summary':
#                 continue
                
#             # GNN results
#             for phase in ['knn', 'explainer']:
#                 if phase in target_results:
#                     for model_type, results in target_results[phase].items():
#                         # Check if std metrics are available
#                         if 'std_r2' in results['avg_metrics']:
#                             summary_data.append({
#                                 'target': target_name,
#                                 'phase': phase,
#                                 'model_type': model_type,
#                                 'model_category': 'GNN',
#                                 'mse': results['avg_metrics']['mse'],
#                                 'rmse': results['avg_metrics']['rmse'],
#                                 'r2': results['avg_metrics']['r2'],
#                                 'mae': results['avg_metrics']['mae'],
#                                 'std_mse': results['avg_metrics']['std_mse'],
#                                 'std_rmse': results['avg_metrics']['std_rmse'],
#                                 'std_r2': results['avg_metrics']['std_r2'],
#                                 'std_mae': results['avg_metrics']['std_mae']
#                             })
#                         else:
#                             # Fallback for non-nested CV results
#                             summary_data.append({
#                                 'target': target_name,
#                                 'phase': phase,
#                                 'model_type': model_type,
#                                 'model_category': 'GNN',
#                                 'mse': results['avg_metrics']['mse'],
#                                 'rmse': results['avg_metrics']['rmse'],
#                                 'r2': results['avg_metrics']['r2'],
#                                 'mae': results['avg_metrics']['mae'],
#                                 'std_mse': 'N/A',
#                                 'std_rmse': 'N/A',
#                                 'std_r2': 'N/A',
#                                 'std_mae': 'N/A'
#                             })
            
#             # ML results
#             if 'ml_models' in target_results:
#                 for model_type, results in target_results['ml_models'].items():
#                     # Debug: Print what's available in ML results
#                     print(f"DEBUG: ML model {model_type} avg_metrics keys: {list(results['avg_metrics'].keys())}")
                    
#                     # Check if std metrics are available
#                     if 'std_r2' in results['avg_metrics']:
#                         summary_data.append({
#                             'target': target_name,
#                             'phase': 'embeddings',
#                             'model_type': model_type,
#                             'model_category': 'ML',
#                             'mse': results['avg_metrics']['mse'],
#                             'rmse': results['avg_metrics']['rmse'],
#                             'r2': results['avg_metrics']['r2'],
#                             'mae': results['avg_metrics']['mae'],
#                             'std_mse': results['avg_metrics'].get('std_mse', 'N/A'),
#                             'std_rmse': results['avg_metrics'].get('std_rmse', 'N/A'),
#                             'std_r2': results['avg_metrics'].get('std_r2', 'N/A'),
#                             'std_mae': results['avg_metrics'].get('std_mae', 'N/A')
#                         })
#                     else:
#                         # Fallback for non-nested CV results
#                         summary_data.append({
#                             'target': target_name,
#                             'phase': 'embeddings',
#                             'model_type': model_type,
#                             'model_category': 'ML',
#                             'mse': results['avg_metrics']['mse'],
#                             'rmse': results['avg_metrics']['rmse'],
#                             'r2': results['avg_metrics']['r2'],
#                             'mae': results['avg_metrics']['mae'],
#                             'std_mse': 'N/A',
#                             'std_rmse': 'N/A',
#                             'std_r2': 'N/A',
#                             'std_mae': 'N/A'
#                         })
        
#         summary_df = pd.DataFrame(summary_data)
#         summary_df.to_csv(f"{self.save_dir}/results_summary.csv", index=False)
        
#         print("Results saved successfully!")

#     def run_pipeline(self):
#         """
#         Run the complete mixed embedding pipeline:
#         1. Train ALL GNN models on KNN-sparsified graph
#         2. Create GNNExplainer-sparsified graph using best model
#         3. Train ALL GNN models on explainer-sparsified graph
#         4. Extract embeddings from best explainer-trained model
#         5. Train ML models on embeddings with 5-fold CV
#         6. Compare and analyze all results
#         """
#         print("\n" + "="*80)
#         print("MIXED EMBEDDING PIPELINE - COMPREHENSIVE GNN + ML ANALYSIS")
#         print("="*80)
        
#         all_results = {}
        
#         # Process each target variable
#         for target_idx, target_name in enumerate(self.target_names):
#             print(f"\n{'='*60}")
#             print(f"PROCESSING TARGET: {target_name} ({target_idx + 1}/{len(self.target_names)})")
#             print(f"{'='*60}")
            
#             target_results = {}
            
#             # Step 1: Train ALL GNN models on KNN-sparsified graph
#             print(f"\nSTEP 1: Training ALL GNN models on KNN-sparsified graph")
#             print("Training all GNN models (GCN, RGGC, GAT)")
#             print("-" * 50)
            
#             knn_results = {}
            
#             for model_type in self.gnn_models_to_train:
#                 knn_results[model_type] = self.train_gnn_model(
#                     model_type=model_type,
#                     target_idx=target_idx,
#                     data_list=self.dataset.data_list
#                 )
            
#             target_results['knn'] = knn_results
            
#             # Find best GNN model for this target
#             best_gnn_model = None
#             best_gnn_r2 = -float('inf')
#             best_gnn_type = None
            
#             for model_type, results in knn_results.items():
#                 if results['avg_metrics']['r2'] > best_gnn_r2:
#                     best_gnn_r2 = results['avg_metrics']['r2']
#                     best_gnn_model = results['model']
#                     best_gnn_type = model_type
            
#             print(f"\nBest KNN GNN model: {best_gnn_type.upper()} (R² = {best_gnn_r2:.4f})")
            
#             # Step 2: Create GNNExplainer-sparsified graph
#             print(f"\nSTEP 2: Creating GNNExplainer-sparsified graph")
#             print(f"Using {best_gnn_type.upper()} model for explanation")
#             print("-" * 50)
            
#             explainer_data = self.create_explainer_sparsified_graph(
#                 model=best_gnn_model,
#                 target_idx=target_idx
#             )
            
#             # Step 3: Train ALL GNN models on explainer-sparsified graph
#             print(f"\nSTEP 3: Training ALL GNN models on explainer-sparsified graph")
#             print("Training all GNN models (GCN, RGGC, GAT)")
#             print("-" * 50)
            
#             explainer_results = {}
            
#             for model_type in self.gnn_models_to_train:
#                 explainer_results[model_type] = self.train_gnn_model(
#                     model_type=model_type,
#                     target_idx=target_idx,
#                     data_list=explainer_data
#                 )
            
#             target_results['explainer'] = explainer_results
            
#             # Find best model from explainer-sparsified graph ONLY (not KNN models)
#             # Only consider the 3 models trained on explainer-sparsified graph
#             best_explainer_model = None
#             best_explainer_r2 = -float('inf')
#             best_explainer_type = None
            
#             for model_type, results in explainer_results.items():
#                 if results['avg_metrics']['r2'] > best_explainer_r2:
#                     best_explainer_r2 = results['avg_metrics']['r2']
#                     best_explainer_model = results['model']
#                     best_explainer_type = model_type
            
#             # Always use explainer data since we're selecting from explainer-trained models
#             embedding_data = explainer_data
            
#             print(f"\nBest explainer-trained GNN model: {best_explainer_type.upper()} (R² = {best_explainer_r2:.4f})")
#             print("Using explainer-sparsified graph for embedding extraction")
            
#             # Step 4: Extract embeddings from best model
#             print(f"\nSTEP 4: Extracting embeddings from best GNN model")
#             print("-" * 50)
            
#             embeddings, targets = self.extract_embeddings(best_explainer_model, embedding_data)
            
#             # Save embeddings
#             embedding_filename = f"{target_name}_embeddings.npy"
#             targets_filename = f"{target_name}_targets.npy"
            
#             np.save(f"{self.save_dir}/embeddings/{embedding_filename}", embeddings)
#             np.save(f"{self.save_dir}/embeddings/{targets_filename}", targets)
            
#             print(f"Extracted embeddings shape: {embeddings.shape}")
#             print(f"Saved as: {embedding_filename}")
            
#             # Step 5: Train ML models on embeddings
#             print(f"\nSTEP 5: Training ML models on embeddings")
#             print("-" * 50)
            
#             ml_results = self.train_ml_models(embeddings, targets, target_idx)
#             target_results['ml_models'] = ml_results
            
#             # Determine embeddings source for naming
#             embeddings_source = f"{best_explainer_type.upper()}"
            
#             # Plot ML model results
#             self.plot_ml_model_results(ml_results, target_name, embeddings_source)
            
#             # Save ML model results
#             self.save_ml_model_results(ml_results, target_name, embeddings_source)
            
#             # Step 6: Create comprehensive plots
#             print(f"\nSTEP 6: Creating comprehensive plots")
#             print("-" * 50)
            
#             # For mixed models, show all models
#             gnn_plot_results = {**knn_results, **{f"{k}_explainer": v for k, v in explainer_results.items()}}
            
#             self.plot_results(
#                 gnn_results=gnn_plot_results,
#                 ml_results=ml_results,
#                 target_idx=target_idx
#             )
            
#             all_results[target_name] = target_results
        
#         # Visualize graphs (KNN and explainer)
#         print(f"\n{'='*60}")
#         print("CREATING GRAPH VISUALIZATIONS")
#         print(f"{'='*60}")
#         self.visualize_graphs()
        
#         # Create comprehensive comparison plots
#         self.create_comprehensive_comparison_plots(all_results)
        
#         # Create overall summary
#         print(f"\n{'='*60}")
#         print("CREATING OVERALL SUMMARY")
#         print(f"{'='*60}")
        
#         # Save all results
#         self.save_results(all_results)
        
#         # Print final summary
#         print(f"\n{'='*80}")
#         print("PIPELINE COMPLETED SUCCESSFULLY!")
#         print(f"{'='*80}")
        
#         print("\nSUMMARY OF BEST MODELS PER TARGET:")
#         print("-" * 50)
        
#         for target_name, target_results in all_results.items():
#             if target_name == 'summary':
#                 continue
                
#             # Find best model for this target across all categories
#             best_r2 = -float('inf')
#             best_model_info = None
            
#             # Check GNN models
#             for phase in ['knn', 'explainer']:
#                 if phase in target_results:
#                     for model_type, results in target_results[phase].items():
#                         if results['avg_metrics']['r2'] > best_r2:
#                             best_r2 = results['avg_metrics']['r2']
#                             best_model_info = f"{model_type.upper()} ({phase})"
            
#             # Check ML models
#             if 'ml_models' in target_results:
#                 for model_type, results in target_results['ml_models'].items():
#                     if results['avg_metrics']['r2'] > best_r2:
#                         best_r2 = results['avg_metrics']['r2']
#                         best_model_info = f"{model_type} (embeddings)"
            
#             print(f"{target_name}: {best_model_info} - R² = {best_r2:.4f}")
        
#         print(f"\nAll results saved to: {self.save_dir}")
#         print("Check the following files:")
#         print(f"  - results_summary.csv: Tabular summary of all results")
#         print(f"  - all_results.pkl: Complete results object")
#         print(f"  - plots/: Comprehensive visualization plots")
#         print(f"  - embeddings/: Extracted embeddings and targets")
        
#         print("\nKey features of this pipeline:")
#         print("- Trains ALL 3 GNN models (GCN, RGGC, GAT) on KNN graph")
#         print("- Selects BEST model for GNNExplainer sparsification")
#         print("- Trains ALL 3 GNN models again on explainer-sparsified graph")
#         print("- Selects BEST model from explainer-trained models for embedding extraction")
#         print("- Extracts embeddings from best explainer-trained model")
#         print("- Trains ML models (LinearSVR, ExtraTrees, RandomForest, XGBoost, LightGBM) on embeddings")
#         print("- Provides comprehensive comparison across all models and phases")
        
#         return all_results 

#     def plot_overall_gnn_results(self, fold_results, model_type, target_name, phase):
#         """Plot overall results across all folds for a GNN model"""
#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Collect all predictions and targets
#         all_preds = []
#         all_targets = []
#         fold_r2s = []
        
#         for fold_result in fold_results:
#             all_preds.extend(fold_result['predictions'])
#             all_targets.extend(fold_result['targets'])
#             fold_r2s.append(fold_result['r2'])
        
#         all_preds = np.array(all_preds)
#         all_targets = np.array(all_targets)
        
#         # Overall metrics
#         overall_mse = mean_squared_error(all_targets, all_preds)
#         overall_rmse = np.sqrt(overall_mse)
#         overall_r2 = r2_score(all_targets, all_preds)
#         overall_mae = mean_absolute_error(all_targets, all_preds)
        
#         # Plot 1: Overall prediction scatter
#         axes[0].scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
#         min_val = min(min(all_targets), min(all_preds))
#         max_val = max(max(all_targets), max(all_preds))
#         axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
#         textstr = f"Overall R² = {overall_r2:.3f}\nRMSE = {overall_rmse:.3f}\nMAE = {overall_mae:.3f}\nMSE = {overall_mse:.3f}"
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#         axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=12,
#                     verticalalignment='top', bbox=props)
        
#         axes[0].set_xlabel('True Values')
#         axes[0].set_ylabel('Predicted Values')
#         axes[0].set_title(f'{model_type.upper()} Overall Predictions - {target_name} ({phase})')
#         axes[0].grid(True, alpha=0.3)
        
#         # Plot 2: R² across folds
#         fold_nums = range(1, len(fold_r2s) + 1)
#         axes[1].bar(fold_nums, fold_r2s, alpha=0.7, color='skyblue', edgecolor='navy')
#         axes[1].axhline(y=overall_r2, color='red', linestyle='--', linewidth=2, label=f'Overall R² = {overall_r2:.3f} (MSE: {overall_mse:.3f})')
#         axes[1].set_xlabel('Fold Number')
#         axes[1].set_ylabel('R² Score')
#         axes[1].set_title(f'{model_type.upper()} R² Across Folds - {target_name} ({phase})')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Save plot
#         plot_path = f"{self.save_dir}/plots/{model_type}_{target_name}_{phase}_overall.png"
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return plot_path

#     def save_detailed_metrics(self, fold_results, model_type, target_name, phase):
#         """Save detailed metrics to CSV including hyperparameter selection"""
#         metrics_data = []
        
#         # Debug: Print the first fold result to see what fields are available
#         if fold_results:
#             print(f"DEBUG: First fold result keys: {list(fold_results[0].keys())}")
#             print(f"DEBUG: First fold result: {fold_results[0]}")
        
#         for fold_result in fold_results:
#             # Basic metrics
#             row_data = {
#                 'fold': fold_result['fold'],
#                 'model_type': model_type,
#                 'target_name': target_name,
#                 'phase': phase,
#                 'mse': fold_result['mse'],
#                 'rmse': fold_result.get('rmse', 'N/A'),  # Use get() with default
#                 'r2': fold_result['r2'],
#                 'mae': fold_result.get('mae', 'N/A')  # Use get() with default
#             }
            
#             # Add hyperparameter information if available
#             if 'best_params' in fold_result:
#                 best_params = fold_result['best_params']
#                 for param_name, param_value in best_params.items():
#                     row_data[f'best_{param_name}'] = param_value
#             else:
#                 # For non-nested CV results, add placeholder
#                 row_data['best_hidden_dim'] = 'N/A'
#                 row_data['best_k_neighbors'] = 'N/A'
            
#             # Add train/test sizes
#             if 'train_size' in fold_result:
#                 row_data['train_size'] = fold_result['train_size']
#                 row_data['test_size'] = fold_result['test_size']
#             else:
#                 row_data['train_size'] = 'N/A'
#                 row_data['test_size'] = 'N/A'
            
#             metrics_data.append(row_data)
        
#         # Calculate overall metrics
#         all_preds = []
#         all_targets = []
#         for fold_result in fold_results:
#             all_preds.extend(fold_result['predictions'])
#             all_targets.extend(fold_result['targets'])
        
#         all_preds = np.array(all_preds)
#         all_targets = np.array(all_targets)
        
#         overall_mse = mean_squared_error(all_targets, all_preds)
#         overall_rmse = np.sqrt(overall_mse)
#         overall_r2 = r2_score(all_targets, all_preds)
#         overall_mae = mean_absolute_error(all_targets, all_preds)
        
#         # Add overall metrics row
#         overall_row = {
#             'fold': 'overall',
#             'model_type': model_type,
#             'target_name': target_name,
#             'phase': phase,
#             'mse': overall_mse,
#             'rmse': overall_rmse,
#             'r2': overall_r2,
#             'mae': overall_mae,
#             'best_hidden_dim': 'N/A',
#             'best_k_neighbors': 'N/A',
#             'train_size': 'N/A',
#             'test_size': 'N/A'
#         }
#         metrics_data.append(overall_row)
        
#         # Save to CSV
#         metrics_df = pd.DataFrame(metrics_data)
#         csv_path = f"{self.save_dir}/detailed_results/{model_type}_{target_name}_{phase}_metrics.csv"
#         metrics_df.to_csv(csv_path, index=False)
        
#         # Also save hyperparameter selection summary
#         if any('best_params' in fold_result for fold_result in fold_results):
#             self.save_hyperparameter_summary(fold_results, model_type, target_name, phase)
        
#         return csv_path

#     def save_hyperparameter_summary(self, fold_results, model_type, target_name, phase):
#         """Save hyperparameter selection summary"""
#         # Collect hyperparameter selections
#         param_selections = {}
#         for fold_result in fold_results:
#             if 'best_params' in fold_result:
#                 best_params = fold_result['best_params']
#                 for param_name, param_value in best_params.items():
#                     if param_name not in param_selections:
#                         param_selections[param_name] = []
#                     param_selections[param_name].append(param_value)
        
#         if param_selections:
#             # Create summary
#             summary_data = []
#             for param_name, values in param_selections.items():
#                 from collections import Counter
#                 value_counts = Counter(values)
#                 total_folds = len(values)
                
#                 for value, count in value_counts.items():
#                     summary_data.append({
#                         'parameter': param_name,
#                         'value': value,
#                         'frequency': count,
#                         'percentage': f"{(count/total_folds)*100:.1f}%",
#                         'total_folds': total_folds
#                     })
            
#             summary_df = pd.DataFrame(summary_data)
#             csv_path = f"{self.save_dir}/detailed_results/{model_type}_{target_name}_{phase}_hyperparameter_summary.csv"
#             summary_df.to_csv(csv_path, index=False)
            
#             print(f"Hyperparameter selection summary saved: {csv_path}")
        
#         return csv_path

#     def visualize_graphs(self):
#         """Visualize the KNN and explainer graphs"""
#         print("\nCreating graph visualizations...")
        
#         # Use the dataset's visualization method
#         self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
#         print(f"Graph visualizations saved to {self.save_dir}/graphs/")

#     def plot_ml_model_results(self, ml_results, target_name, embeddings_source):
#         """Plot ML model results with detailed fold-by-fold analysis"""
#         # Create comprehensive comparison plot
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#         fig.suptitle(f'ML Models on {embeddings_source} Embeddings - {target_name}', fontsize=16)
        
#         model_names = list(ml_results.keys())
#         # Extended color palette for more models
#         colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'gold', 'pink', 'cyan']
        
#         # Plot 1: R² comparison across folds
#         ax1 = axes[0, 0]
#         fold_nums = range(1, self.num_folds + 1)
        
#         for i, (model_name, results) in enumerate(ml_results.items()):
#             fold_r2s = [fold['r2'] for fold in results['fold_results']]
#             color = colors[i % len(colors)]
#             ax1.plot(fold_nums, fold_r2s, marker='o', label=model_name, color=color, linewidth=2)
            
#             # Add average line with MSE
#             avg_r2 = results['avg_metrics']['r2']
#             avg_mse = results['avg_metrics']['mse']
#             ax1.axhline(y=avg_r2, color=color, linestyle='--', alpha=0.7, 
#                        label=f'{model_name} Overall = {avg_r2:.3f} (MSE: {avg_mse:.3f})')
        
#         ax1.set_xlabel('Fold Number')
#         ax1.set_ylabel('R² Score')
#         ax1.set_title('R² Score Across Folds')
#         # Use smaller font size for legend when there are many models
#         legend_fontsize = 8 if len(model_names) > 3 else 10
#         ax1.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax1.grid(True, alpha=0.3)
        
#         # Plot 2: Overall performance comparison
#         ax2 = axes[0, 1]
#         metrics = ['r2', 'rmse', 'mae']
#         x = np.arange(len(metrics))
#         width = 0.8 / len(model_names)  # Adjust width based on number of models
        
#         for i, (model_name, results) in enumerate(ml_results.items()):
#             values = [results['avg_metrics'][metric] for metric in metrics]
#             mse_val = results['avg_metrics']['mse']
#             # Normalize RMSE and MAE for better visualization
#             if len(values) > 1:
#                 values[1] = values[1] / max(values[1], 1)  # Normalize RMSE
#                 values[2] = values[2] / max(values[2], 1)  # Normalize MAE
            
#             color = colors[i % len(colors)]
#             ax2.bar(x + i*width, values, width, label=f'{model_name} (MSE: {mse_val:.3f})', 
#                    color=color, alpha=0.7)
        
#         ax2.set_xlabel('Metrics')
#         ax2.set_ylabel('Normalized Values')
#         ax2.set_title('Performance Metrics Comparison')
#         ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
#         ax2.set_xticklabels(metrics)
#         ax2.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax2.grid(True, alpha=0.3)
        
#         # Plot 3: Prediction scatter for best model
#         ax3 = axes[1, 0]
#         best_model_name = max(ml_results.keys(), key=lambda k: ml_results[k]['avg_metrics']['r2'])
#         best_results = ml_results[best_model_name]
        
#         # Collect all predictions and targets
#         all_preds = []
#         all_targets = []
#         for fold_result in best_results['fold_results']:
#             all_preds.extend(fold_result['predictions'])
#             all_targets.extend(fold_result['targets'])
        
#         ax3.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
#         min_val = min(min(all_targets), min(all_preds))
#         max_val = max(max(all_targets), max(all_preds))
#         ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
#         best_r2 = best_results['avg_metrics']['r2']
#         best_mse = best_results['avg_metrics']['mse']
#         textstr = f"Best Model: {best_model_name}\nR² = {best_r2:.3f}\nMSE = {best_mse:.3f}"
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#         ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
#                 verticalalignment='top', bbox=props)
        
#         ax3.set_xlabel('True Values')
#         ax3.set_ylabel('Predicted Values')
#         ax3.set_title(f'Best ML Model: {best_model_name}')
#         ax3.grid(True, alpha=0.3)
        
#         # Plot 4: Error distribution for top 3 models (to avoid overcrowding)
#         ax4 = axes[1, 1]
#         # Sort models by R² and take top 3
#         sorted_models = sorted(ml_results.items(), key=lambda x: x[1]['avg_metrics']['r2'], reverse=True)
#         top_models = sorted_models[:3]  # Show only top 3 models to avoid overcrowding
        
#         for i, (model_name, results) in enumerate(top_models):
#             all_preds = []
#             all_targets = []
#             for fold_result in results['fold_results']:
#                 all_preds.extend(fold_result['predictions'])
#                 all_targets.extend(fold_result['targets'])
            
#             errors = np.array(all_targets) - np.array(all_preds)
#             mse_val = results['avg_metrics']['mse']
#             color = colors[i % len(colors)]
#             ax4.hist(errors, bins=20, alpha=0.6, label=f'{model_name} (MSE: {mse_val:.3f})', color=color)
        
#         ax4.set_xlabel('Prediction Error')
#         ax4.set_ylabel('Frequency')
#         ax4.set_title('Error Distribution (Top 3 Models)')
#         ax4.legend(fontsize=legend_fontsize)
#         ax4.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Save comprehensive plot
#         plot_path = f"{self.save_dir}/plots/ml_models_{target_name}_{embeddings_source}_comprehensive.png"
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         # Create individual plots for each ML model
#         for model_name, results in ml_results.items():
#             self.plot_individual_ml_model(model_name, results, target_name, embeddings_source)
        
#         return plot_path

#     def plot_individual_ml_model(self, model_name, results, target_name, embeddings_source):
#         """Create individual plot for a specific ML model"""
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#         fig.suptitle(f'{model_name} on {embeddings_source} Embeddings - {target_name}', fontsize=16)
        
#         # Collect all predictions and targets
#         all_preds = []
#         all_targets = []
#         fold_r2s = []
#         fold_rmses = []
#         fold_mses = []
        
#         for fold_result in results['fold_results']:
#             all_preds.extend(fold_result['predictions'])
#             all_targets.extend(fold_result['targets'])
#             fold_r2s.append(fold_result['r2'])
#             fold_rmses.append(fold_result['rmse'])
#             fold_mses.append(fold_result['mse'])
        
#         all_preds = np.array(all_preds)
#         all_targets = np.array(all_targets)
        
#         # Plot 1: Prediction scatter
#         ax1 = axes[0, 0]
#         ax1.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
#         min_val = min(min(all_targets), min(all_preds))
#         max_val = max(max(all_targets), max(all_preds))
#         ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
#         r2 = results['avg_metrics']['r2']
#         rmse = results['avg_metrics']['rmse']
#         mae = results['avg_metrics']['mae']
#         mse = results['avg_metrics']['mse']
        
#         textstr = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nMSE = {mse:.3f}"
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#         ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
#                 verticalalignment='top', bbox=props)
        
#         ax1.set_xlabel('True Values')
#         ax1.set_ylabel('Predicted Values')
#         ax1.set_title('Predictions vs True Values')
#         ax1.grid(True, alpha=0.3)
        
#         # Plot 2: R² across folds
#         ax2 = axes[0, 1]
#         fold_nums = range(1, len(fold_r2s) + 1)
#         ax2.bar(fold_nums, fold_r2s, alpha=0.7, color='skyblue', edgecolor='navy')
#         ax2.axhline(y=r2, color='red', linestyle='--', linewidth=2, label=f'Overall R² = {r2:.3f} (MSE: {mse:.3f})')
#         ax2.set_xlabel('Fold Number')
#         ax2.set_ylabel('R² Score')
#         ax2.set_title('R² Score Across Folds')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         # Plot 3: Error distribution
#         ax3 = axes[1, 0]
#         errors = all_targets - all_preds
#         ax3.hist(errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
#         ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8)
#         ax3.set_xlabel('Prediction Error')
#         ax3.set_ylabel('Frequency')
#         ax3.set_title(f'Error Distribution (MSE: {mse:.3f})')
#         ax3.grid(True, alpha=0.3)
        
#         # Plot 4: MSE across folds (changed from RMSE to MSE)
#         ax4 = axes[1, 1]
#         ax4.bar(fold_nums, fold_mses, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
#         ax4.axhline(y=mse, color='red', linestyle='--', linewidth=2, label=f'Overall MSE = {mse:.3f}')
#         ax4.set_xlabel('Fold Number')
#         ax4.set_ylabel('MSE')
#         ax4.set_title('MSE Across Folds')
#         ax4.legend()
#         ax4.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Save individual plot
#         plot_path = f"{self.save_dir}/plots/{model_name}_{target_name}_{embeddings_source}_individual.png"
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"Individual {model_name} plot saved: {plot_path}")
        
#         return plot_path

#     def save_ml_model_results(self, ml_results, target_name, embeddings_source):
#         """Save ML model results to files"""
#         for model_name, results in ml_results.items():
#             # Save detailed metrics
#             metrics_data = []
#             for fold_result in results['fold_results']:
#                 metrics_data.append({
#                     'fold': fold_result['fold'],
#                     'model_name': model_name,
#                     'target_name': target_name,
#                     'embeddings_source': embeddings_source,
#                     'mse': fold_result['mse'],
#                     'rmse': fold_result['rmse'],
#                     'r2': fold_result['r2'],
#                     'mae': fold_result['mae']
#                 })
            
#             # Add overall metrics
#             metrics_data.append({
#                 'fold': 'overall',
#                 'model_name': model_name,
#                 'target_name': target_name,
#                 'embeddings_source': embeddings_source,
#                 'mse': results['avg_metrics']['mse'],
#                 'rmse': results['avg_metrics']['rmse'],
#                 'r2': results['avg_metrics']['r2'],
#                 'mae': results['avg_metrics']['mae']
#             })
            
#             # Save to CSV
#             metrics_df = pd.DataFrame(metrics_data)
#             csv_path = f"{self.save_dir}/detailed_results/ml_{model_name}_{target_name}_{embeddings_source}_metrics.csv"
#             metrics_df.to_csv(csv_path, index=False)
            
#             # Save model
#             model_path = f"{self.save_dir}/ml_models/{model_name}_{target_name}_{embeddings_source}.pkl"
#             with open(model_path, 'wb') as f:
#                 pickle.dump(results['model'], f)

#     def create_comprehensive_comparison_plots(self, all_results):
#         """Create comprehensive comparison plots across all models and phases"""
#         print("\nCreating comprehensive comparison plots...")
        
#         for target_name, target_results in all_results.items():
#             if target_name == 'summary':
#                 continue
            
#             # Collect all results for this target
#             comparison_data = []
            
#             # GNN results
#             for phase in ['knn', 'explainer']:
#                 if phase in target_results:
#                     for model_type, results in target_results[phase].items():
#                         comparison_data.append({
#                             'model': f"{model_type.upper()} ({phase})",
#                             'type': 'GNN',
#                             'r2': results['avg_metrics']['r2'],
#                             'rmse': results['avg_metrics']['rmse'],
#                             'mae': results['avg_metrics']['mae'],
#                             'mse': results['avg_metrics']['mse']
#                         })
            
#             # ML results
#             if 'ml_models' in target_results:
#                 embeddings_source = "Best GNN"
#                 for model_type, results in target_results['ml_models'].items():
#                     comparison_data.append({
#                         'model': f"{model_type} (on {embeddings_source} embeddings)",
#                         'type': 'ML',
#                         'r2': results['avg_metrics']['r2'],
#                         'rmse': results['avg_metrics']['rmse'],
#                         'mae': results['avg_metrics']['mae'],
#                         'mse': results['avg_metrics']['mse']
#                     })
            
#             # Create comprehensive plot
#             fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # Made wider to accommodate more models
#             fig.suptitle(f'Comprehensive Model Comparison - {target_name}', fontsize=16)
            
#             models = [item['model'] for item in comparison_data]
#             r2_scores = [item['r2'] for item in comparison_data]
#             rmse_scores = [item['rmse'] for item in comparison_data]
#             mae_scores = [item['mae'] for item in comparison_data]
#             mse_scores = [item['mse'] for item in comparison_data]
#             colors = ['skyblue' if item['type'] == 'GNN' else 'orange' for item in comparison_data]
            
#             # R² comparison
#             bars1 = axes[0, 0].bar(range(len(models)), r2_scores, color=colors, alpha=0.7)
#             axes[0, 0].set_title('R² Score Comparison')
#             axes[0, 0].set_ylabel('R² Score')
#             axes[0, 0].set_xticks(range(len(models)))
#             axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
#             axes[0, 0].grid(True, alpha=0.3)
            
#             # Add value labels with MSE
#             for bar, r2_score, mse_score in zip(bars1, r2_scores, mse_scores):
#                 axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                                f'R²:{r2_score:.3f}\nMSE:{mse_score:.3f}', ha='center', va='bottom', fontsize=7)
            
#             # RMSE comparison
#             bars2 = axes[0, 1].bar(range(len(models)), rmse_scores, color=colors, alpha=0.7)
#             axes[0, 1].set_title('RMSE Comparison')
#             axes[0, 1].set_ylabel('RMSE')
#             axes[0, 1].set_xticks(range(len(models)))
#             axes[0, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
#             axes[0, 1].grid(True, alpha=0.3)
            
#             # Add MSE labels on RMSE bars
#             for bar, rmse_score, mse_score in zip(bars2, rmse_scores, mse_scores):
#                 axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                                f'RMSE:{rmse_score:.3f}\nMSE:{mse_score:.3f}', ha='center', va='bottom', fontsize=7)
            
#             # MAE comparison
#             bars3 = axes[1, 0].bar(range(len(models)), mae_scores, color=colors, alpha=0.7)
#             axes[1, 0].set_title('MAE Comparison')
#             axes[1, 0].set_ylabel('MAE')
#             axes[1, 0].set_xticks(range(len(models)))
#             axes[1, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
#             axes[1, 0].grid(True, alpha=0.3)
            
#             # Add MSE labels on MAE bars
#             for bar, mae_score, mse_score in zip(bars3, mae_scores, mse_scores):
#                 axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                                f'MAE:{mae_score:.3f}\nMSE:{mse_score:.3f}', ha='center', va='bottom', fontsize=7)
            
#             # MSE comparison (new plot)
#             bars4 = axes[1, 1].bar(range(len(models)), mse_scores, color=colors, alpha=0.7)
#             axes[1, 1].set_title('MSE Comparison')
#             axes[1, 1].set_ylabel('MSE')
#             axes[1, 1].set_xticks(range(len(models)))
#             axes[1, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
#             axes[1, 1].grid(True, alpha=0.3)
            
#             # Add MSE value labels
#             for bar, mse_score in zip(bars4, mse_scores):
#                 axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                                f'{mse_score:.3f}', ha='center', va='bottom', fontsize=8)
            
#             # Add legend
#             from matplotlib.patches import Patch
#             legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='GNN Models'),
#                              Patch(facecolor='orange', alpha=0.7, label='ML Models')]
#             axes[0, 0].legend(handles=legend_elements, loc='upper right')
            
#             plt.tight_layout()
            
#             # Save plot
#             plot_path = f"{self.save_dir}/plots/comprehensive_comparison_{target_name}.png"
#             plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             print(f"Comprehensive comparison plot saved: {plot_path}")

#     def _create_gnn_model_with_params(self, model_type, hidden_dim, num_targets=1):
#         """Create a GNN model with specific hidden_dim parameter"""
#         if model_type == 'gcn':
#             model = simple_GCN_res_plus_regression(
#                 hidden_channels=hidden_dim,
#                 output_dim=num_targets,
#                 dropout_prob=self.dropout_rate,
#                 input_channel=1,
#                 estimate_uncertainty=False
#             ).to(device)
#         elif model_type == 'rggc':
#             model = simple_RGGC_plus_regression(
#                 hidden_channels=hidden_dim,
#                 output_dim=num_targets,
#                 dropout_prob=self.dropout_rate,
#                 input_channel=1,
#                 estimate_uncertainty=False
#             ).to(device)
#         elif model_type == 'gat':
#             model = simple_GAT_regression(
#                 hidden_channels=hidden_dim,
#                 output_dim=num_targets,
#                 dropout_prob=self.dropout_rate,
#                 input_channel=1,
#                 num_heads=8,
#                 estimate_uncertainty=False
#             ).to(device)
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")
        
#         return model
    
#     def _train_and_evaluate_once_with_params(self, model, data_list, train_idx, val_idx, target_idx, 
#                                            hidden_dim=None, max_epochs=None):
#         """Train model with specific parameters and evaluate"""
#         if max_epochs is None:
#             max_epochs = min(self.num_epochs, 50)  # Shorter training for inner CV
        
#         # Ensure model is on the correct device
#         model = model.to(device)
        
#         # Split data and move to device
#         train_data = [data_list[i] for i in train_idx]
#         val_data = [data_list[i] for i in val_idx]
        
#         # Move all data to the correct device
#         train_data = self._move_data_to_device(train_data)
#         val_data = self._move_data_to_device(val_data)
        
#         # Create data loaders
#         batch_size = min(self.batch_size, len(train_data) // 4)
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
#         # Setup optimizer and scheduler
#         optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
        
#         criterion = nn.MSELoss()
        
#         # Training loop
#         best_val_loss = float('inf')
#         patience_counter = 0
        
#         for epoch in range(max_epochs):
#             # Training
#             model.train()
#             total_train_loss = 0
            
#             for batch_data in train_loader:
#                 # Data is already on device from _move_data_to_device
#                 optimizer.zero_grad()
                
#                 out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                 target = batch_data.y[:, target_idx].view(-1, 1)
                
#                 loss = criterion(out, target)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 total_train_loss += loss.item() * batch_data.num_graphs
            
#             # Validation
#             model.eval()
#             total_val_loss = 0
            
#             with torch.no_grad():
#                 for batch_data in val_loader:
#                     # Data is already on device from _move_data_to_device
#                     out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                     target = batch_data.y[:, target_idx].view(-1, 1)
#                     loss = criterion(out, target)
#                     total_val_loss += loss.item() * batch_data.num_graphs
            
#             avg_val_loss = total_val_loss / len(val_loader.dataset)
#             scheduler.step(avg_val_loss)
            
#             # Early stopping
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= min(self.patience, 10):  # Shorter patience for inner CV
#                     break
        
#         # Final evaluation
#         model.eval()
#         all_preds = []
#         all_targets = []
        
#         with torch.no_grad():
#             for batch_data in val_loader:
#                 # Data is already on device from _move_data_to_device
#                 out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
#                 target = batch_data.y[:, target_idx].view(-1, 1)
                
#                 all_preds.append(out.cpu().numpy())
#                 all_targets.append(target.cpu().numpy())
        
#         all_preds = np.vstack(all_preds).flatten()
#         all_targets = np.vstack(all_targets).flatten()
        
#         r2 = r2_score(all_targets, all_preds)
#         return r2


# # Example usage
# if __name__ == "__main__":
#     print("="*80)
#     print("MIXED EMBEDDING PIPELINE WITH NESTED CV HYPERPARAMETER TUNING")
#     print("="*80)
#     print("This pipeline trains all 3 GNN models with nested CV hyperparameter tuning,")
#     print("selects best for explainer, then trains all 3 models again on sparsified graph")
    
#     # Create mixed pipeline with nested CV enabled
#     mixed_pipeline = MixedEmbeddingPipeline(
#         data_path="Data/New_Data.csv",
#         k_neighbors=10,
#         hidden_dim=64,
#         num_epochs=100,  # Reduced for faster testing
#         num_folds=5,
#         save_dir="./mixed_embedding_results_nested_cv",
#         graph_mode='family',
#         importance_threshold=0.2,
#         use_enhanced_training=True,
#         adaptive_hyperparameters=True,
#         use_nested_cv=True,  # Enable nested CV hyperparameter tuning
#     )
    
#     # Run mixed pipeline
#     results = mixed_pipeline.run_pipeline()
    
#     print("\n" + "="*80)
#     print("MIXED PIPELINE WITH NESTED CV ANALYSIS COMPLETE!")
#     print("="*80)
#     print("Results saved to: ./mixed_embedding_results_nested_cv/")
#     print("\nKey features of this pipeline:")
#     print("- Uses NESTED CROSS-VALIDATION for hyperparameter tuning")
#     print("- Trains ALL 3 GNN models (GCN, RGGC, GAT) on KNN graph with tuned hyperparameters")
#     print("- Selects BEST model for GNNExplainer sparsification")
#     print("- Trains ALL 3 GNN models again on explainer-sparsified graph with tuned hyperparameters")
#     print("- Selects BEST model from explainer-trained models for embedding extraction")
#     print("- Extracts embeddings from best explainer-trained model")
#     print("- Trains ML models (LinearSVR, ExtraTrees, RandomForest, XGBoost, LightGBM) on embeddings")
#     print("- Provides comprehensive comparison across all models and phases")
#     print("\nHyperparameter search space:")
#     print(f"- hidden_dim: {mixed_pipeline.gnn_hyperparams['hidden_dim']}")
#     print(f"- k_neighbors: {mixed_pipeline.gnn_hyperparams['k_neighbors']}")
#     print(f"- Total combinations: {len(mixed_pipeline.param_grid)}")



import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Import dataset and explainer modules (now from same directory)
from dataset_regression import MicrobialGNNDataset
from explainer_regression import GNNExplainerRegression
from pipeline_explainer import create_explainer_sparsified_graph

# Import the plus models that return embeddings (now from same directory)
from GNNmodelsRegression import (
    simple_GCN_res_plus_regression,
    simple_RGGC_plus_regression,
    simple_GAT_regression,
    GaussianNLLLoss
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MixedEmbeddingPipeline:
    """
    Complete pipeline for graph-based regression using GNN embeddings with ML models.
    
    Pipeline Flow:
    1. Generate graph from data
    2. Create KNN graph sparsification
    3. Train ALL GNN models (plus versions for embeddings) with nested CV hyperparameter tuning
    4. Use best GNN model for GNNExplainer to get sparsified graph
    5. Train ALL GNN models on sparsified graph with nested CV hyperparameter tuning
    6. Extract embeddings from best overall GNN model
    7. Train ML models (LinearSVR, ExtraTrees, RandomForest, XGBoost, LightGBM) on embeddings with 5-fold CV
    """
    
    def __init__(self, 
                 data_path,
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 hidden_dim=64,
                 dropout_rate=0.3,
                 batch_size=8,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 num_epochs=200,
                 patience=20,
                 num_folds=5,
                 save_dir='./mixed_embedding_results',
                 importance_threshold=0.2,  # Use default threshold - pipeline_explainer has adaptive thresholding
                 use_fast_correlation=False,
                 graph_mode='family',  # Changed default to family
                 family_filter_mode='strict',
                 use_enhanced_training=True,  # New parameter for enhanced training
                 adaptive_hyperparameters=True,  # New parameter for adaptive hyperparameters
                 use_nested_cv=True):  # New parameter for nested CV hyperparameter tuning
        """
        Initialize the mixed embedding pipeline
        
        Args:
            data_path: Path to the CSV file with data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
            hidden_dim: Hidden dimension size for GNN
            dropout_rate: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
            num_folds: Number of folds for cross-validation
            save_dir: Directory to save results
            importance_threshold: Threshold for edge importance in GNNExplainer sparsification
            use_fast_correlation: If True, use fast correlation-based graph construction
            graph_mode: Mode for graph construction ('otu' or 'family') - now defaults to 'family'
            family_filter_mode: Mode for family filtering ('relaxed' or 'strict')
            use_enhanced_training: If True, use enhanced training
            adaptive_hyperparameters: If True, use adaptive hyperparameters
            use_nested_cv: If True, use nested cross-validation for hyperparameter tuning
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_folds = num_folds
        self.save_dir = save_dir
        self.importance_threshold = importance_threshold
        self.use_fast_correlation = use_fast_correlation
        self.graph_mode = graph_mode
        self.family_filter_mode = family_filter_mode
        self.use_enhanced_training = use_enhanced_training
        self.adaptive_hyperparameters = adaptive_hyperparameters
        self.use_nested_cv = use_nested_cv
        
        # Define hyperparameter search space for nested CV
        self.gnn_hyperparams = {
            # 'hidden_dim': [512, 256, 64],
            # 'k_neighbors': [8, 10, 12],
            'hidden_dim': [64],
            'k_neighbors': [10]

        }
        self.param_grid = list(ParameterGrid(self.gnn_hyperparams))
        
        # For explainer phase, we only tune hidden_dim since graph is already sparsified
        self.explainer_hyperparams = {
            'hidden_dim': [64] # 256, 64
        }
        self.explainer_param_grid = list(ParameterGrid(self.explainer_hyperparams))
        
        # Always train all three models in mixed approach
        self.gnn_models_to_train = ['gcn', 'rggc', 'gat']
        print("Pipeline configured for MIXED model comparison")
        
        if self.use_nested_cv:
            print("Using NESTED CROSS-VALIDATION for hyperparameter tuning")
            print(f"Hyperparameter search space: {len(self.param_grid)} combinations")
        else:
            print("Using standard cross-validation (no hyperparameter tuning)")
        
        print(f"Using graph mode: {graph_mode} (family-level nodes)")
        
        # Create comprehensive save directories (matching regression pipeline structure)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/gnn_models", exist_ok=True)
        os.makedirs(f"{save_dir}/ml_models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/graphs", exist_ok=True)
        os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations", exist_ok=True)
        os.makedirs(f"{save_dir}/detailed_results", exist_ok=True)
        
        # Load and process data
        print("\nLoading and processing data...")
        self.dataset = MicrobialGNNDataset(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode,
            family_filter_mode=family_filter_mode
        )
        
        # Get target names for reference
        self.target_names = self.dataset.target_cols
        print(f"Target variables: {self.target_names}")

    def create_gnn_model(self, model_type, num_targets=1):
        """Create a GNN plus model that returns embeddings"""
        if model_type == 'gcn':
            model = simple_GCN_res_plus_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=False
            ).to(device)
        elif model_type == 'rggc':
            model = simple_RGGC_plus_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=False
            ).to(device)
        elif model_type == 'gat':
            # Enhanced GAT with more heads and better architecture
            model = simple_GAT_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                num_heads=8,  # Increased from 4 to 8 attention heads
                estimate_uncertainty=False
            ).to(device)
        # Add DGCNN option for dynamic graph construction
        elif model_type == 'dgcnn':
            try:
                from GNNmodelsRegression import Enhanced_DGCNN_regression
                model = Enhanced_DGCNN_regression(
                    hidden_channels=self.hidden_dim,
                    output_dim=num_targets,
                    dropout_prob=self.dropout_rate,
                    input_channel=1,
                    k=min(self.k_neighbors, 10),  # Adaptive k-neighbors
                    num_layers=4,
                    estimate_uncertainty=False
                ).to(device)
            except ImportError:
                print("Enhanced DGCNN not available, falling back to GAT")
                model = simple_GAT_regression(
                    hidden_channels=self.hidden_dim,
                    output_dim=num_targets,
                    dropout_prob=self.dropout_rate,
                    input_channel=1,
                    num_heads=8,
                    estimate_uncertainty=False
                ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model

    def _move_data_to_device(self, data_list):
        """Move all data objects in the list to the correct device"""
        device_data_list = []
        for data in data_list:
            device_data = data.to(device)
            device_data_list.append(device_data)
        return device_data_list

    def _train_and_evaluate_once(self, model, data_list, train_idx, val_idx, target_idx, max_epochs=None):
        """Train model on train_idx and evaluate on val_idx, returning R² score"""
        if max_epochs is None:
            max_epochs = min(self.num_epochs, 50)  # Shorter training for inner CV
        
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Split data and move to device
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        # Move all data to the correct device
        train_data = self._move_data_to_device(train_data)
        val_data = self._move_data_to_device(val_data)
        
        # Create data loaders
        batch_size = min(self.batch_size, len(train_data) // 4)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            total_train_loss = 0
            
            for batch_data in train_loader:
                # Data is already on device from _move_data_to_device
                optimizer.zero_grad()
                
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                loss = criterion(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item() * batch_data.num_graphs
            
            # Validation
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    # Data is already on device from _move_data_to_device
                    out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    target = batch_data.y[:, target_idx].view(-1, 1)
                    loss = criterion(out, target)
                    total_val_loss += loss.item() * batch_data.num_graphs
            
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= min(self.patience, 10):  # Shorter patience for inner CV
                    break
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Data is already on device from _move_data_to_device
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        
        r2 = r2_score(all_targets, all_preds)
        return r2

    def _inner_loop_select(self, model_type, train_data, target_idx):
        """Inner loop for hyperparameter selection using K-fold CV"""
        inner_kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        best_score = -float('inf')
        best_params = None
        all_combinations_results = []
        
        # SIMPLE LOGIC: 
        # If train_data is the original dataset, we're doing KNN phase (tune both hidden_dim and k_neighbors)
        # If train_data is different from original dataset, we're doing explainer phase (only tune hidden_dim)
        is_explainer_phase = (train_data is not self.dataset.data_list)
        
        # Use appropriate parameter grid
        if is_explainer_phase:
            param_grid = self.explainer_param_grid
            print(f"    Inner CV (Explainer Phase): Testing {len(param_grid)} hidden_dim combinations...")
        else:
            param_grid = self.param_grid
            print(f"    Inner CV (KNN Phase): Testing {len(param_grid)} hyperparameter combinations...")
        
        print(f"    {'Combination':<15} {'hidden_dim':<12} {'k_neighbors':<12} {'Mean R²':<10} {'Std R²':<10} {'Best':<6}")
        print(f"    {'-'*75}")
        
        for i, params in enumerate(param_grid):
            local_hidden_dim = params['hidden_dim']
            local_k_neighbors = params.get('k_neighbors', self.k_neighbors)
            
            # For explainer phase, use existing sparsified graph data
            if is_explainer_phase:
                temp_data_list = train_data
            else:
                # For KNN phase, recreate dataset with local k_neighbors value
                temp_dataset = MicrobialGNNDataset(
                    data_path=self.data_path,
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
            
            mean_val = np.mean(val_scores)
            std_val = np.std(val_scores)
            
            combination_result = {
                'combination': i + 1,
                'params': params.copy(),
                'mean_r2': mean_val,
                'std_r2': std_val,
                'val_scores': val_scores,
                'is_best': False
            }
            all_combinations_results.append(combination_result)
            
            if mean_val > best_score:
                best_score = mean_val
                best_params = params.copy()
                combination_result['is_best'] = True
            
            # Print progress
            best_marker = "✓" if combination_result['is_best'] else " "
            # Show actual k_neighbors value - only show 'Fixed' in explainer phase
            if is_explainer_phase:
                k_neighbors_val = 'Fixed'
            else:
                k_neighbors_val = params.get('k_neighbors', 'N/A')
            print(f"    {i+1:<15} {params['hidden_dim']:<12} {k_neighbors_val:<12} {mean_val:<10.4f} {std_val:<10.4f} {best_marker:<6}")
        
        print(f"    {'-'*75}")
        print(f"    Best hyperparameters: {best_params} (R² = {best_score:.4f})")
        
        # Print detailed results for top 3 combinations
        print(f"\n    Top 3 Hyperparameter Combinations:")
        sorted_results = sorted(all_combinations_results, key=lambda x: x['mean_r2'], reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            params = result['params']
            mean_r2 = result['mean_r2']
            std_r2 = result['std_r2']
            val_scores = result['val_scores']
            rank = i + 1
            
            # Show actual k_neighbors value in detailed results
            if is_explainer_phase:
                k_neighbors_display = 'Fixed'
            else:
                k_neighbors_display = params.get('k_neighbors', 'N/A')
            print(f"    {rank}. hidden_dim={params['hidden_dim']}, k_neighbors={k_neighbors_display}")
            print(f"       R² = {mean_r2:.4f} ± {std_r2:.4f}")
            print(f"       Individual fold R² scores: {[f'{score:.4f}' for score in val_scores]}")
        
        return best_params

    def _train_model_full_with_params(self, model, train_data, target_idx, hidden_dim=None):
        """Train model on full training data with specific hyperparameters"""
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Move all data to the correct device
        train_data = self._move_data_to_device(train_data)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training
            model.train()
            total_train_loss = 0
            
            for batch_data in train_loader:
                # Data is already on device from _move_data_to_device
                optimizer.zero_grad()
                
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                loss = criterion(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item() * batch_data.num_graphs
            
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            scheduler.step(avg_train_loss)
            
            # Early stopping based on training loss (no validation set in full training)
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Load best model state
        model.load_state_dict(best_model_state)
        return model

    def _evaluate_model(self, model, test_data, target_idx):
        """Evaluate model on test data and return MSE and R²"""
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Move all data to the correct device
        test_data = self._move_data_to_device(test_data)
        
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Data is already on device from _move_data_to_device
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        
        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return mse, r2

    def train_gnn_model_nested(self, model_type, target_idx, data_list=None):
        """Train GNN model with nested cross-validation for hyperparameter tuning"""
        if data_list is None:
            data_list = self.dataset.data_list
        
        target_name = self.target_names[target_idx]
        phase = "explainer" if data_list != self.dataset.data_list else "knn"
        print(f"\nTraining {model_type.upper()} model with NESTED CV for target: {target_name} ({phase} graph)")
        
        outer_kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        outer_results = []
        best_model_state = None
        best_outer_r2 = -float('inf')
        best_hyperparams = None  # Store the best hyperparameters
        
        print(f"\n{'='*60}")
        print(f"NESTED CV RESULTS FOR {model_type.upper()} - {target_name}")
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
            
            # 2. FIXED: Use best hyperparameters without modifying global state
            best_hidden_dim = best_params['hidden_dim']
            best_k_neighbors = best_params.get('k_neighbors', self.k_neighbors)
            
            # FIXED: Ensure data consistency between inner and outer loops
            if 'k_neighbors' in best_params and not (data_list != self.dataset.data_list):
                # Only recreate dataset if we're in KNN phase (not explainer phase)
                print(f"    Recreating dataset with best k_neighbors={best_k_neighbors}")
                best_dataset = MicrobialGNNDataset(
                    data_path=self.data_path,
                    k_neighbors=best_k_neighbors,
                    mantel_threshold=self.mantel_threshold,
                    use_fast_correlation=self.use_fast_correlation,
                    graph_mode=self.graph_mode,
                    family_filter_mode=self.family_filter_mode
                )
                # Move all data to the correct device
                best_dataset_data_list = self._move_data_to_device(best_dataset.data_list)
                # Use the same indices as the original split
                best_train_data = [best_dataset_data_list[i] for i in train_idx]
                best_test_data = [best_dataset_data_list[i] for i in test_idx]
            else:
                # For explainer phase or when k_neighbors wasn't tuned, use existing data
                best_train_data = train_data
                best_test_data = test_data
            
            # FIXED: Create model with best hyperparameters without modifying global state
            model = self._create_gnn_model_with_params(model_type, best_hidden_dim, num_targets=1)
            model = self._train_model_full_with_params(model, best_train_data, target_idx, best_hidden_dim)
            
            # 3. Evaluate on test_data
            mse, r2 = self._evaluate_model(model, best_test_data, target_idx)
            print(f"  Final Evaluation on Test Set:")
            print(f"    R² = {r2:.4f}, MSE = {mse:.4f}")
            
            # Get predictions and targets for this fold
            model.eval()
            all_preds = []
            all_targets = []
            
            test_loader = DataLoader(best_test_data, batch_size=self.batch_size, shuffle=False)
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    target = batch_data.y[:, target_idx].view(-1, 1)
                    
                    all_preds.append(out.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
            
            all_preds = np.vstack(all_preds).flatten()
            all_targets = np.vstack(all_targets).flatten()
            
            # Calculate additional metrics
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(all_targets, all_preds)
            
            outer_results.append({
                'fold': fold_num,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'best_params': best_params,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'predictions': all_preds,
                'targets': all_targets
            })
            
            if r2 > best_outer_r2:
                best_outer_r2 = r2
                best_model_state = model.state_dict().copy()
                best_hyperparams = best_params.copy()  # Store the best hyperparameters
        
        # 4. Comprehensive Summary
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE NESTED CV SUMMARY")
        print(f"{'='*60}")
        
        r2_scores = [r['r2'] for r in outer_results]
        mse_scores = [r['mse'] for r in outer_results]
        rmse_scores = [r['rmse'] for r in outer_results]
        mae_scores = [r['mae'] for r in outer_results]
        
        avg_r2 = np.mean(r2_scores)
        avg_mse = np.mean(mse_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)
        std_r2 = np.std(r2_scores)
        std_mse = np.std(mse_scores)
        std_rmse = np.std(rmse_scores)
        std_mae = np.std(mae_scores)
        
        print(f"Model: {model_type.upper()}")
        print(f"Target: {target_name}")
        print(f"Phase: {phase}")
        print(f"Number of outer folds: {self.num_folds}")
        
        print(f"\nFold-by-Fold Results:")
        print(f"{'Fold':<6} {'R²':<10} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'Best hidden_dim':<15} {'Best k_neighbors':<15}")
        print(f"{'-'*80}")
        
        for result in outer_results:
            fold = result['fold']
            r2 = result['r2']
            mse = result['mse']
            rmse = result['rmse']
            mae = result['mae']
            best_params = result['best_params']
            hidden_dim = best_params.get('hidden_dim', 'N/A')
            k_neighbors = best_params.get('k_neighbors', 'N/A')
            print(f"{fold:<6} {r2:<10.4f} {mse:<10.4f} {rmse:<10.4f} {mae:<10.4f} {hidden_dim:<15} {k_neighbors:<15}")
        
        print(f"\nOverall Performance:")
        print(f"  R² = {avg_r2:.4f} ± {std_r2:.4f}")
        print(f"  MSE = {avg_mse:.4f} ± {std_mse:.4f}")
        print(f"  RMSE = {avg_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  MAE = {avg_mae:.4f} ± {std_mae:.4f}")
        
        # Show hyperparameter selection frequency
        print(f"\nHyperparameter Selection Frequency:")
        hidden_dims = [r['best_params'].get('hidden_dim', 'N/A') for r in outer_results]
        k_neighbors_list = [r['best_params'].get('k_neighbors', 'N/A') for r in outer_results]
        
        from collections import Counter
        hidden_dim_counts = Counter(hidden_dims)
        k_neighbors_counts = Counter(k_neighbors_list)
        
        print(f"  hidden_dim selections:")
        for dim, count in hidden_dim_counts.items():
            print(f"    {dim}: {count}/{self.num_folds} folds")
        
        print(f"  k_neighbors selections:")
        for k, count in k_neighbors_counts.items():
            print(f"    {k}: {count}/{self.num_folds} folds")
        
        # FIXED: Create final model with best hyperparameters without modifying global state
        if best_hyperparams is not None:
            best_hidden_dim = best_hyperparams['hidden_dim']
        else:
            best_hidden_dim = self.hidden_dim
        
        best_model = self._create_gnn_model_with_params(model_type, best_hidden_dim, num_targets=1)
        best_model.load_state_dict(best_model_state)
        
        # Save detailed metrics including hyperparameter selection
        self.save_detailed_metrics(outer_results, model_type, target_name, phase)
        
        return {
            'model': best_model,
            'fold_results': outer_results,
            'avg_metrics': {
                'r2': avg_r2,
                'mse': avg_mse,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'std_r2': std_r2,
                'std_mse': std_mse,
                'std_rmse': std_rmse,
                'std_mae': std_mae
            },
            'model_type': model_type,
            'target_name': target_name,
            'target_idx': target_idx,
            'phase': phase
        }

    def train_gnn_model(self, model_type, target_idx, data_list=None):
        """Original GNN training method (kept for backward compatibility)"""
        if self.use_nested_cv:
            return self.train_gnn_model_nested(model_type, target_idx, data_list)
        
        # Original training method (no hyperparameter tuning)
        if data_list is None:
            data_list = self.dataset.data_list
        
        target_name = self.target_names[target_idx]
        phase = "explainer" if data_list != self.dataset.data_list else "knn"
        print(f"\nTraining {model_type.upper()} model for target: {target_name} ({phase} graph)")
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        best_model = None
        best_r2 = -float('inf')
        
        criterion = nn.MSELoss()
        
        # Iterate through folds
        for fold, (train_index, test_index) in enumerate(kf.split(data_list)):
            fold_num = fold + 1
            print(f"  Fold {fold_num}/{self.num_folds}")
            
            # Split into train and test sets
            train_dataset = [data_list[i] for i in train_index]
            test_dataset = [data_list[i] for i in test_index]
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            model = self.create_gnn_model(model_type, num_targets=1)
            
            # Setup optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
            
            # Training loop
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.num_epochs):
                # Training
                model.train()
                total_train_loss = 0
                
                for batch_data in train_loader:
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass - get predictions and embeddings
                    out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    
                    # Extract target for this specific target_idx
                    target = batch_data.y[:, target_idx].view(-1, 1)
                    
                    loss = criterion(out, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()
                    
                    total_train_loss += loss.item() * batch_data.num_graphs
                
                avg_train_loss = total_train_loss / len(train_loader.dataset)
                train_losses.append(avg_train_loss)
                
                # Validation
                model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        target = batch_data.y[:, target_idx].view(-1, 1)
                        loss = criterion(out, target)
                        total_val_loss += loss.item() * batch_data.num_graphs
                
                avg_val_loss = total_val_loss / len(test_loader.dataset)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                
                # Print progress
                if epoch % 20 == 0 or epoch == 1 or epoch == self.num_epochs - 1:
                    print(f"    Epoch {epoch+1:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model for evaluation
            model.load_state_dict(best_model_state)
            
            # Final evaluation
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    target = batch_data.y[:, target_idx].view(-1, 1)
                    
                    all_preds.append(out.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
            
            # Calculate metrics
            all_preds = np.vstack(all_preds).flatten()
            all_targets = np.vstack(all_targets).flatten()
            
            mse = mean_squared_error(all_targets, all_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(all_targets, all_preds)
            mae = mean_absolute_error(all_targets, all_preds)
            
            # Save model
            model_path = f"{self.save_dir}/gnn_models/{model_type}_{target_name}_fold{fold_num}_{phase}.pt"
            torch.save(model.state_dict(), model_path)
            
            fold_results.append({
                'fold': fold_num,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'predictions': all_preds,
                'targets': all_targets,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'model_path': model_path
            })
            
            # Keep track of best model across folds
            if r2 > best_r2:
                best_r2 = r2
                best_model = model.state_dict().copy()
            
            print(f"    MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        # Calculate overall metrics from all validation samples combined
        all_fold_preds = []
        all_fold_targets = []
        for fold_result in fold_results:
            all_fold_preds.extend(fold_result['predictions'])
            all_fold_targets.extend(fold_result['targets'])
        
        all_fold_preds = np.array(all_fold_preds)
        all_fold_targets = np.array(all_fold_targets)
        
        # Calculate overall metrics (replaces avg_metrics)
        overall_metrics = {
            'mse': mean_squared_error(all_fold_targets, all_fold_preds),
            'rmse': np.sqrt(mean_squared_error(all_fold_targets, all_fold_preds)),
            'r2': r2_score(all_fold_targets, all_fold_preds),
            'mae': mean_absolute_error(all_fold_targets, all_fold_preds)
        }
        
        print(f"  Overall - MSE: {overall_metrics['mse']:.4f}, RMSE: {overall_metrics['rmse']:.4f}, R²: {overall_metrics['r2']:.4f}, MAE: {overall_metrics['mae']:.4f}")
        
        # Create overall plots (only overall, no individual fold plots)
        self.plot_overall_gnn_results(fold_results, model_type, target_name, phase)
        
        # Save detailed metrics
        self.save_detailed_metrics(fold_results, model_type, target_name, phase)
        
        # Create final model with best weights
        final_model = self.create_gnn_model(model_type, num_targets=1)
        final_model.load_state_dict(best_model)
        
        return {
            'model': final_model,
            'fold_results': fold_results,
            'avg_metrics': overall_metrics,  # Now contains overall metrics instead of averages
            'model_type': model_type,
            'target_name': target_name,
            'target_idx': target_idx,
            'phase': phase
        }

    def extract_embeddings(self, model, data_list):
        """Extract embeddings from trained GNN model"""
        model.eval()
        all_embeddings = []
        all_targets = []
        
        # Create data loader for all data
        data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                
                # Forward pass to get embeddings
                out, embeddings = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_targets.append(batch_data.y.cpu().numpy())
        
        # Concatenate all embeddings and targets
        embeddings = np.vstack(all_embeddings)
        targets = np.vstack(all_targets)
        
        return embeddings, targets

    def train_ml_models(self, embeddings, targets, target_idx):
        """Train ML models (LinearSVR, ExtraTrees, XGBoost, RandomForest, LightGBM) on embeddings with 5-fold CV"""
        target_name = self.target_names[target_idx]
        target_values = targets[:, target_idx]
        
        print(f"\nTraining ML models on embeddings for target: {target_name}")
        print(f"Embedding shape: {embeddings.shape}, Target shape: {target_values.shape}")
        
        # Define ML models with preprocessing pipelines
        ml_models = {
            'LinearSVR': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearSVR(epsilon=0.1, tol=1e-4, C=1.0, max_iter=10000))
            ]),
            'ExtraTrees': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            ]),
            'RandomForest': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10))
            ])
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            ml_models['XGBoost'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                ))
            ])
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            ml_models['LightGBM'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                ))
            ])
        
        print(f"Training {len(ml_models)} ML models: {list(ml_models.keys())}")
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        ml_results = {}
        
        for model_name, model_pipeline in ml_models.items():
            print(f"\n  Training {model_name}...")
            fold_results = []
            
            for fold, (train_index, test_index) in enumerate(kf.split(embeddings)):
                fold_num = fold + 1
                
                # Split data
                X_train, X_test = embeddings[train_index], embeddings[test_index]
                y_train, y_test = target_values[train_index], target_values[test_index]
                
                # Train model
                model_pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred = model_pipeline.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                fold_results.append({
                    'fold': fold_num,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'predictions': y_pred,
                    'targets': y_test
                })
                
                print(f"    Fold {fold_num}: MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            
            # Calculate overall metrics from all validation samples combined
            all_fold_preds = []
            all_fold_targets = []
            for fold_result in fold_results:
                all_fold_preds.extend(fold_result['predictions'])
                all_fold_targets.extend(fold_result['targets'])
            
            all_fold_preds = np.array(all_fold_preds)
            all_fold_targets = np.array(all_fold_targets)
            
            # Calculate fold-wise metrics for mean ± std
            fold_mse_scores = [fold_result['mse'] for fold_result in fold_results]
            fold_rmse_scores = [fold_result['rmse'] for fold_result in fold_results]
            fold_r2_scores = [fold_result['r2'] for fold_result in fold_results]
            fold_mae_scores = [fold_result['mae'] for fold_result in fold_results]
            
            # Calculate mean ± std
            avg_metrics = {
                'mse': np.mean(fold_mse_scores),
                'rmse': np.mean(fold_rmse_scores),
                'r2': np.mean(fold_r2_scores),
                'mae': np.mean(fold_mae_scores),
                'std_mse': np.std(fold_mse_scores),
                'std_rmse': np.std(fold_rmse_scores),
                'std_r2': np.std(fold_r2_scores),
                'std_mae': np.std(fold_mae_scores)
            }
            
            print(f"    Overall - MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['std_mse']:.4f}, RMSE: {avg_metrics['rmse']:.4f} ± {avg_metrics['std_rmse']:.4f}, R²: {avg_metrics['r2']:.4f} ± {avg_metrics['std_r2']:.4f}, MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['std_mae']:.4f}")
            
            # Train final model on all data
            final_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', ml_models[model_name]['regressor'])
            ])
            final_model.fit(embeddings, target_values)
            
            ml_results[model_name] = {
                'model': final_model,
                'fold_results': fold_results,
                'avg_metrics': avg_metrics,  # Now contains mean ± std metrics
                'target_name': target_name,
                'target_idx': target_idx
            }
        
        return ml_results

    def create_explainer_sparsified_graph(self, model, target_idx=0):
        """Create sparsified graph using GNNExplainer - uses existing pipeline_explainer function"""
        print(f"\nCreating GNNExplainer sparsified graph for target: {self.target_names[target_idx]}")
        print(f"Using importance threshold: {self.importance_threshold}")
        
        # Use the existing function from pipeline_explainer.py
        sparsified_data_list = create_explainer_sparsified_graph(
            pipeline=self,  # Pass self as pipeline
            model=model,
            target_idx=target_idx,
            importance_threshold=self.importance_threshold
        )
        
        print(f"GNNExplainer sparsification complete: {len(sparsified_data_list)} samples created")
        
        return sparsified_data_list

    def plot_results(self, gnn_results, ml_results, target_idx):
        """Create simple prediction vs actual plot for best ML model"""
        target_name = self.target_names[target_idx]
        
        # Find best ML model
        ml_models = list(ml_results.keys())
        ml_r2_scores = [ml_results[model]['avg_metrics']['r2'] for model in ml_models]
        best_ml_model = ml_models[np.argmax(ml_r2_scores)]
        best_r2 = max(ml_r2_scores)
        best_mse = ml_results[best_ml_model]['avg_metrics']['mse']
        
        print(f"\nBest ML model: {best_ml_model} (R² = {best_r2:.4f}, MSE = {best_mse:.4f})")
        
        # Create simple prediction vs actual plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Collect all predictions and targets from folds
        all_preds = []
        all_targets = []
        for fold_result in ml_results[best_ml_model]['fold_results']:
            all_preds.extend(fold_result['predictions'])
            all_targets.extend(fold_result['targets'])
        
        # Create scatter plot
        ax.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none', s=50)
        
        # Add diagonal line
        min_val = min(min(all_targets), min(all_preds))
        max_val = max(max(all_targets), max(all_preds))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_title(f'Best ML Model: {best_ml_model}\nR² = {best_r2:.4f}, MSE = {best_mse:.4f}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.grid(True, alpha=0.3)
        
        # Add R² text
        ax.text(0.05, 0.95, f'R² = {best_r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/{target_name}_best_ml_pred_vs_actual.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Best ML model plot saved: {self.save_dir}/plots/{target_name}_best_ml_pred_vs_actual.png")

    def save_results(self, all_results):
        """Save all results to files"""
        print(f"\nSaving results to {self.save_dir}")
        
        # Save results as pickle
        with open(f"{self.save_dir}/all_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save summary as CSV
        summary_data = []
        
        for target_name, target_results in all_results.items():
            if target_name == 'summary':
                continue
                
            # GNN results
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    for model_type, results in target_results[phase].items():
                        # Check if std metrics are available
                        if 'std_r2' in results['avg_metrics']:
                            summary_data.append({
                                'target': target_name,
                                'phase': phase,
                                'model_type': model_type,
                                'model_category': 'GNN',
                                'mse': results['avg_metrics']['mse'],
                                'rmse': results['avg_metrics']['rmse'],
                                'r2': results['avg_metrics']['r2'],
                                'mae': results['avg_metrics']['mae'],
                                'std_mse': results['avg_metrics']['std_mse'],
                                'std_rmse': results['avg_metrics']['std_rmse'],
                                'std_r2': results['avg_metrics']['std_r2'],
                                'std_mae': results['avg_metrics']['std_mae']
                            })
                        else:
                            # Fallback for non-nested CV results
                            summary_data.append({
                                'target': target_name,
                                'phase': phase,
                                'model_type': model_type,
                                'model_category': 'GNN',
                                'mse': results['avg_metrics']['mse'],
                                'rmse': results['avg_metrics']['rmse'],
                                'r2': results['avg_metrics']['r2'],
                                'mae': results['avg_metrics']['mae'],
                                'std_mse': 'N/A',
                                'std_rmse': 'N/A',
                                'std_r2': 'N/A',
                                'std_mae': 'N/A'
                            })
            
            # ML results
            if 'ml_models' in target_results:
                for model_type, results in target_results['ml_models'].items():
                    # Debug: Print what's available in ML results
                    print(f"DEBUG: ML model {model_type} avg_metrics keys: {list(results['avg_metrics'].keys())}")
                    
                    # Check if std metrics are available
                    if 'std_r2' in results['avg_metrics']:
                        summary_data.append({
                            'target': target_name,
                            'phase': 'embeddings',
                            'model_type': model_type,
                            'model_category': 'ML',
                            'mse': results['avg_metrics']['mse'],
                            'rmse': results['avg_metrics']['rmse'],
                            'r2': results['avg_metrics']['r2'],
                            'mae': results['avg_metrics']['mae'],
                            'std_mse': results['avg_metrics'].get('std_mse', 'N/A'),
                            'std_rmse': results['avg_metrics'].get('std_rmse', 'N/A'),
                            'std_r2': results['avg_metrics'].get('std_r2', 'N/A'),
                            'std_mae': results['avg_metrics'].get('std_mae', 'N/A')
                        })
                    else:
                        # Fallback for non-nested CV results
                        summary_data.append({
                            'target': target_name,
                            'phase': 'embeddings',
                            'model_type': model_type,
                            'model_category': 'ML',
                            'mse': results['avg_metrics']['mse'],
                            'rmse': results['avg_metrics']['rmse'],
                            'r2': results['avg_metrics']['r2'],
                            'mae': results['avg_metrics']['mae'],
                            'std_mse': 'N/A',
                            'std_rmse': 'N/A',
                            'std_r2': 'N/A',
                            'std_mae': 'N/A'
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/results_summary.csv", index=False)
        
        print("Results saved successfully!")

    def run_pipeline(self):
        """
        Run the complete mixed embedding pipeline:
        1. Train ALL GNN models on KNN-sparsified graph
        2. Create GNNExplainer-sparsified graph using best model
        3. Train ALL GNN models on explainer-sparsified graph
        4. Extract embeddings from best explainer-trained model
        5. Train ML models on embeddings with 5-fold CV
        6. Compare and analyze all results
        """
        print("\n" + "="*80)
        print("MIXED EMBEDDING PIPELINE - COMPREHENSIVE GNN + ML ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        # Process each target variable
        for target_idx, target_name in enumerate(self.target_names):
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET: {target_name} ({target_idx + 1}/{len(self.target_names)})")
            print(f"{'='*60}")
            
            target_results = {}
            
            # Step 1: Train ALL GNN models on KNN-sparsified graph
            print(f"\nSTEP 1: Training ALL GNN models on KNN-sparsified graph")
            print("Training all GNN models (GCN, RGGC, GAT)")
            print("-" * 50)
            
            knn_results = {}
            
            for model_type in self.gnn_models_to_train:
                knn_results[model_type] = self.train_gnn_model(
                    model_type=model_type,
                    target_idx=target_idx,
                    data_list=self.dataset.data_list
                )
            
            target_results['knn'] = knn_results
            
            # Find best GNN model for this target
            best_gnn_model = None
            best_gnn_r2 = -float('inf')
            best_gnn_type = None
            
            for model_type, results in knn_results.items():
                if results['avg_metrics']['r2'] > best_gnn_r2:
                    best_gnn_r2 = results['avg_metrics']['r2']
                    best_gnn_model = results['model']
                    best_gnn_type = model_type
            
            print(f"\nBest KNN GNN model: {best_gnn_type.upper()} (R² = {best_gnn_r2:.4f})")
            
            # Step 2: Create GNNExplainer-sparsified graph
            print(f"\nSTEP 2: Creating GNNExplainer-sparsified graph")
            print(f"Using {best_gnn_type.upper()} model for explanation")
            print("-" * 50)
            
            explainer_data = self.create_explainer_sparsified_graph(
                model=best_gnn_model,
                target_idx=target_idx
            )
            
            # Step 3: Train ALL GNN models on explainer-sparsified graph
            print(f"\nSTEP 3: Training ALL GNN models on explainer-sparsified graph")
            print("Training all GNN models (GCN, RGGC, GAT)")
            print("-" * 50)
            
            explainer_results = {}
            
            for model_type in self.gnn_models_to_train:
                explainer_results[model_type] = self.train_gnn_model(
                    model_type=model_type,
                    target_idx=target_idx,
                    data_list=explainer_data
                )
            
            target_results['explainer'] = explainer_results
            
            # Find best model from explainer-sparsified graph ONLY (not KNN models)
            # Only consider the 3 models trained on explainer-sparsified graph
            best_explainer_model = None
            best_explainer_r2 = -float('inf')
            best_explainer_type = None
            
            for model_type, results in explainer_results.items():
                if results['avg_metrics']['r2'] > best_explainer_r2:
                    best_explainer_r2 = results['avg_metrics']['r2']
                    best_explainer_model = results['model']
                    best_explainer_type = model_type
            
            # Always use explainer data since we're selecting from explainer-trained models
            embedding_data = explainer_data
            
            print(f"\nBest explainer-trained GNN model: {best_explainer_type.upper()} (R² = {best_explainer_r2:.4f})")
            print("Using explainer-sparsified graph for embedding extraction")
            
            # Step 4: Extract embeddings from best model
            print(f"\nSTEP 4: Extracting embeddings from best GNN model")
            print("-" * 50)
            
            embeddings, targets = self.extract_embeddings(best_explainer_model, embedding_data)
            
            # Save embeddings
            embedding_filename = f"{target_name}_embeddings.npy"
            targets_filename = f"{target_name}_targets.npy"
            
            np.save(f"{self.save_dir}/embeddings/{embedding_filename}", embeddings)
            np.save(f"{self.save_dir}/embeddings/{targets_filename}", targets)
            
            print(f"Extracted embeddings shape: {embeddings.shape}")
            print(f"Saved as: {embedding_filename}")
            
            # Step 5: Train ML models on embeddings
            print(f"\nSTEP 5: Training ML models on embeddings")
            print("-" * 50)
            
            ml_results = self.train_ml_models(embeddings, targets, target_idx)
            target_results['ml_models'] = ml_results
            
            # Determine embeddings source for naming
            embeddings_source = f"{best_explainer_type.upper()}"
            
            # Plot ML model results
            self.plot_ml_model_results(ml_results, target_name, embeddings_source)
            
            # Save ML model results
            self.save_ml_model_results(ml_results, target_name, embeddings_source)
            
            # Step 6: Create comprehensive plots
            print(f"\nSTEP 6: Creating comprehensive plots")
            print("-" * 50)
            
            # For mixed models, show all models
            gnn_plot_results = {**knn_results, **{f"{k}_explainer": v for k, v in explainer_results.items()}}
            
            self.plot_results(
                gnn_results=gnn_plot_results,
                ml_results=ml_results,
                target_idx=target_idx
            )
            
            all_results[target_name] = target_results
        
        # Visualize graphs (KNN and explainer)
        print(f"\n{'='*60}")
        print("CREATING GRAPH VISUALIZATIONS")
        print(f"{'='*60}")
        self.visualize_graphs()
        
        # Create comprehensive comparison plots
        self.create_comprehensive_comparison_plots(all_results)
        
        # Create overall summary
        print(f"\n{'='*60}")
        print("CREATING OVERALL SUMMARY")
        print(f"{'='*60}")
        
        # Save all results
        self.save_results(all_results)
        
        # Save comprehensive hyperparameter tracking
        self.save_comprehensive_hyperparameter_tracking(all_results)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        print("\nSUMMARY OF BEST MODELS PER TARGET:")
        print("-" * 50)
        
        for target_name, target_results in all_results.items():
            if target_name == 'summary':
                continue
                
            # Find best model for this target across all categories
            best_r2 = -float('inf')
            best_model_info = None
            
            # Check GNN models
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    for model_type, results in target_results[phase].items():
                        if results['avg_metrics']['r2'] > best_r2:
                            best_r2 = results['avg_metrics']['r2']
                            best_model_info = f"{model_type.upper()} ({phase})"
            
            # Check ML models
            if 'ml_models' in target_results:
                for model_type, results in target_results['ml_models'].items():
                    if results['avg_metrics']['r2'] > best_r2:
                        best_r2 = results['avg_metrics']['r2']
                        best_model_info = f"{model_type} (embeddings)"
            
            print(f"{target_name}: {best_model_info} - R² = {best_r2:.4f}")
        
        print(f"\nAll results saved to: {self.save_dir}")
        print("Check the following files:")
        print(f"  - results_summary.csv: Tabular summary of all results")
        print(f"  - all_results.pkl: Complete results object")
        print(f"  - plots/: Comprehensive visualization plots")
        print(f"  - embeddings/: Extracted embeddings and targets")
        
        print("\nKey features of this pipeline:")
        print("- Trains ALL 3 GNN models (GCN, RGGC, GAT) on KNN graph")
        print("- Selects BEST model for GNNExplainer sparsification")
        print("- Trains ALL 3 GNN models again on explainer-sparsified graph")
        print("- Selects BEST model from explainer-trained models for embedding extraction")
        print("- Extracts embeddings from best explainer-trained model")
        print("- Trains ML models (LinearSVR, ExtraTrees, RandomForest, XGBoost, LightGBM) on embeddings")
        print("- Provides comprehensive comparison across all models and phases")
        
        return all_results 

    def plot_overall_gnn_results(self, fold_results, model_type, target_name, phase):
        """Plot overall results across all folds for a GNN model"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect all predictions and targets
        all_preds = []
        all_targets = []
        fold_r2s = []
        
        for fold_result in fold_results:
            all_preds.extend(fold_result['predictions'])
            all_targets.extend(fold_result['targets'])
            fold_r2s.append(fold_result['r2'])
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Overall metrics
        overall_mse = mean_squared_error(all_targets, all_preds)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(all_targets, all_preds)
        overall_mae = mean_absolute_error(all_targets, all_preds)
        
        # Plot 1: Overall prediction scatter
        axes[0].scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
        min_val = min(min(all_targets), min(all_preds))
        max_val = max(max(all_targets), max(all_preds))
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        textstr = f"Overall R² = {overall_r2:.3f}\nRMSE = {overall_rmse:.3f}\nMAE = {overall_mae:.3f}\nMSE = {overall_mse:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
        
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{model_type.upper()} Overall Predictions - {target_name} ({phase})')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: R² across folds
        fold_nums = range(1, len(fold_r2s) + 1)
        axes[1].bar(fold_nums, fold_r2s, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1].axhline(y=overall_r2, color='red', linestyle='--', linewidth=2, label=f'Overall R² = {overall_r2:.3f} (MSE: {overall_mse:.3f})')
        axes[1].set_xlabel('Fold Number')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title(f'{model_type.upper()} R² Across Folds - {target_name} ({phase})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.save_dir}/plots/{model_type}_{target_name}_{phase}_overall.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def save_detailed_metrics(self, fold_results, model_type, target_name, phase):
        """Save detailed metrics to CSV including hyperparameter selection"""
        metrics_data = []
        
        # Debug: Print the first fold result to see what fields are available
        if fold_results:
            print(f"DEBUG: First fold result keys: {list(fold_results[0].keys())}")
            print(f"DEBUG: First fold result: {fold_results[0]}")
        
        for fold_result in fold_results:
            # Basic metrics
            row_data = {
                'fold': fold_result['fold'],
                'model_type': model_type,
                'target_name': target_name,
                'phase': phase,
                'mse': fold_result['mse'],
                'rmse': fold_result.get('rmse', 'N/A'),  # Use get() with default
                'r2': fold_result['r2'],
                'mae': fold_result.get('mae', 'N/A')  # Use get() with default
            }
            
            # Add hyperparameter information if available
            if 'best_params' in fold_result:
                best_params = fold_result['best_params']
                for param_name, param_value in best_params.items():
                    row_data[f'best_{param_name}'] = param_value
            else:
                # For non-nested CV results, add placeholder
                row_data['best_hidden_dim'] = 'N/A'
                row_data['best_k_neighbors'] = 'N/A'
            
            # Add train/test sizes
            if 'train_size' in fold_result:
                row_data['train_size'] = fold_result['train_size']
                row_data['test_size'] = fold_result['test_size']
            else:
                row_data['train_size'] = 'N/A'
                row_data['test_size'] = 'N/A'
            
            metrics_data.append(row_data)
        
        # Calculate overall metrics
        all_preds = []
        all_targets = []
        for fold_result in fold_results:
            all_preds.extend(fold_result['predictions'])
            all_targets.extend(fold_result['targets'])
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        overall_mse = mean_squared_error(all_targets, all_preds)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(all_targets, all_preds)
        overall_mae = mean_absolute_error(all_targets, all_preds)
        
        # Add overall metrics row
        overall_row = {
            'fold': 'overall',
            'model_type': model_type,
            'target_name': target_name,
            'phase': phase,
            'mse': overall_mse,
            'rmse': overall_rmse,
            'r2': overall_r2,
            'mae': overall_mae,
            'best_hidden_dim': 'N/A',
            'best_k_neighbors': 'N/A',
            'train_size': 'N/A',
            'test_size': 'N/A'
        }
        metrics_data.append(overall_row)
        
        # Save to CSV
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = f"{self.save_dir}/detailed_results/{model_type}_{target_name}_{phase}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        # Also save hyperparameter selection summary
        if any('best_params' in fold_result for fold_result in fold_results):
            self.save_hyperparameter_summary(fold_results, model_type, target_name, phase)
        
        return csv_path

    def save_hyperparameter_summary(self, fold_results, model_type, target_name, phase):
        """Save hyperparameter selection summary"""
        # Collect hyperparameter selections
        param_selections = {}
        for fold_result in fold_results:
            if 'best_params' in fold_result:
                best_params = fold_result['best_params']
                for param_name, param_value in best_params.items():
                    if param_name not in param_selections:
                        param_selections[param_name] = []
                    param_selections[param_name].append(param_value)
        
        if param_selections:
            # Create summary
            summary_data = []
            for param_name, values in param_selections.items():
                from collections import Counter
                value_counts = Counter(values)
                total_folds = len(values)
                
                for value, count in value_counts.items():
                    summary_data.append({
                        'parameter': param_name,
                        'value': value,
                        'frequency': count,
                        'percentage': f"{(count/total_folds)*100:.1f}%",
                        'total_folds': total_folds
                    })
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = f"{self.save_dir}/detailed_results/{model_type}_{target_name}_{phase}_hyperparameter_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            
            print(f"Hyperparameter selection summary saved: {csv_path}")
        
        return csv_path

    def visualize_graphs(self):
        """Visualize the KNN and explainer graphs"""
        print("\nCreating graph visualizations...")
        
        # Use the dataset's visualization method
        self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
        print(f"Graph visualizations saved to {self.save_dir}/graphs/")

    def plot_ml_model_results(self, ml_results, target_name, embeddings_source):
        """Plot ML model results with detailed fold-by-fold analysis"""
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ML Models on {embeddings_source} Embeddings - {target_name}', fontsize=16)
        
        model_names = list(ml_results.keys())
        # Extended color palette for more models
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'gold', 'pink', 'cyan']
        
        # Plot 1: R² comparison across folds
        ax1 = axes[0, 0]
        fold_nums = range(1, self.num_folds + 1)
        
        for i, (model_name, results) in enumerate(ml_results.items()):
            fold_r2s = [fold['r2'] for fold in results['fold_results']]
            color = colors[i % len(colors)]
            ax1.plot(fold_nums, fold_r2s, marker='o', label=model_name, color=color, linewidth=2)
            
            # Add average line with MSE
            avg_r2 = results['avg_metrics']['r2']
            avg_mse = results['avg_metrics']['mse']
            ax1.axhline(y=avg_r2, color=color, linestyle='--', alpha=0.7, 
                       label=f'{model_name} Overall = {avg_r2:.3f} (MSE: {avg_mse:.3f})')
        
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Across Folds')
        # Use smaller font size for legend when there are many models
        legend_fontsize = 8 if len(model_names) > 3 else 10
        ax1.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overall performance comparison
        ax2 = axes[0, 1]
        metrics = ['r2', 'rmse', 'mae']
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)  # Adjust width based on number of models
        
        for i, (model_name, results) in enumerate(ml_results.items()):
            values = [results['avg_metrics'][metric] for metric in metrics]
            mse_val = results['avg_metrics']['mse']
            # Normalize RMSE and MAE for better visualization
            if len(values) > 1:
                values[1] = values[1] / max(values[1], 1)  # Normalize RMSE
                values[2] = values[2] / max(values[2], 1)  # Normalize MAE
            
            color = colors[i % len(colors)]
            ax2.bar(x + i*width, values, width, label=f'{model_name} (MSE: {mse_val:.3f})', 
                   color=color, alpha=0.7)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Normalized Values')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax2.set_xticklabels(metrics)
        ax2.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prediction scatter for best model
        ax3 = axes[1, 0]
        best_model_name = max(ml_results.keys(), key=lambda k: ml_results[k]['avg_metrics']['r2'])
        best_results = ml_results[best_model_name]
        
        # Collect all predictions and targets
        all_preds = []
        all_targets = []
        for fold_result in best_results['fold_results']:
            all_preds.extend(fold_result['predictions'])
            all_targets.extend(fold_result['targets'])
        
        ax3.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
        min_val = min(min(all_targets), min(all_preds))
        max_val = max(max(all_targets), max(all_preds))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        best_r2 = best_results['avg_metrics']['r2']
        best_mse = best_results['avg_metrics']['mse']
        textstr = f"Best Model: {best_model_name}\nR² = {best_r2:.3f}\nMSE = {best_mse:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax3.set_xlabel('True Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title(f'Best ML Model: {best_model_name}')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution for top 3 models (to avoid overcrowding)
        ax4 = axes[1, 1]
        # Sort models by R² and take top 3
        sorted_models = sorted(ml_results.items(), key=lambda x: x[1]['avg_metrics']['r2'], reverse=True)
        top_models = sorted_models[:3]  # Show only top 3 models to avoid overcrowding
        
        for i, (model_name, results) in enumerate(top_models):
            all_preds = []
            all_targets = []
            for fold_result in results['fold_results']:
                all_preds.extend(fold_result['predictions'])
                all_targets.extend(fold_result['targets'])
            
            errors = np.array(all_targets) - np.array(all_preds)
            mse_val = results['avg_metrics']['mse']
            color = colors[i % len(colors)]
            ax4.hist(errors, bins=20, alpha=0.6, label=f'{model_name} (MSE: {mse_val:.3f})', color=color)
        
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution (Top 3 Models)')
        ax4.legend(fontsize=legend_fontsize)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comprehensive plot
        plot_path = f"{self.save_dir}/plots/ml_models_{target_name}_{embeddings_source}_comprehensive.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual plots for each ML model
        for model_name, results in ml_results.items():
            self.plot_individual_ml_model(model_name, results, target_name, embeddings_source)
        
        return plot_path

    def plot_individual_ml_model(self, model_name, results, target_name, embeddings_source):
        """Create individual plot for a specific ML model"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} on {embeddings_source} Embeddings - {target_name}', fontsize=16)
        
        # Collect all predictions and targets
        all_preds = []
        all_targets = []
        fold_r2s = []
        fold_rmses = []
        fold_mses = []
        
        for fold_result in results['fold_results']:
            all_preds.extend(fold_result['predictions'])
            all_targets.extend(fold_result['targets'])
            fold_r2s.append(fold_result['r2'])
            fold_rmses.append(fold_result['rmse'])
            fold_mses.append(fold_result['mse'])
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Plot 1: Prediction scatter
        ax1 = axes[0, 0]
        ax1.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
        min_val = min(min(all_targets), min(all_preds))
        max_val = max(max(all_targets), max(all_preds))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        r2 = results['avg_metrics']['r2']
        rmse = results['avg_metrics']['rmse']
        mae = results['avg_metrics']['mae']
        mse = results['avg_metrics']['mse']
        
        textstr = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nMSE = {mse:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predictions vs True Values')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: R² across folds
        ax2 = axes[0, 1]
        fold_nums = range(1, len(fold_r2s) + 1)
        ax2.bar(fold_nums, fold_r2s, alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.axhline(y=r2, color='red', linestyle='--', linewidth=2, label=f'Overall R² = {r2:.3f} (MSE: {mse:.3f})')
        ax2.set_xlabel('Fold Number')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Across Folds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[1, 0]
        errors = all_targets - all_preds
        ax3.hist(errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Error Distribution (MSE: {mse:.3f})')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: MSE across folds (changed from RMSE to MSE)
        ax4 = axes[1, 1]
        ax4.bar(fold_nums, fold_mses, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax4.axhline(y=mse, color='red', linestyle='--', linewidth=2, label=f'Overall MSE = {mse:.3f}')
        ax4.set_xlabel('Fold Number')
        ax4.set_ylabel('MSE')
        ax4.set_title('MSE Across Folds')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        plot_path = f"{self.save_dir}/plots/{model_name}_{target_name}_{embeddings_source}_individual.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual {model_name} plot saved: {plot_path}")
        
        return plot_path

    def save_ml_model_results(self, ml_results, target_name, embeddings_source):
        """Save ML model results to files"""
        for model_name, results in ml_results.items():
            # Save detailed metrics
            metrics_data = []
            for fold_result in results['fold_results']:
                metrics_data.append({
                    'fold': fold_result['fold'],
                    'model_name': model_name,
                    'target_name': target_name,
                    'embeddings_source': embeddings_source,
                    'mse': fold_result['mse'],
                    'rmse': fold_result['rmse'],
                    'r2': fold_result['r2'],
                    'mae': fold_result['mae']
                })
            
            # Add overall metrics
            metrics_data.append({
                'fold': 'overall',
                'model_name': model_name,
                'target_name': target_name,
                'embeddings_source': embeddings_source,
                'mse': results['avg_metrics']['mse'],
                'rmse': results['avg_metrics']['rmse'],
                'r2': results['avg_metrics']['r2'],
                'mae': results['avg_metrics']['mae']
            })
            
            # Save to CSV
            metrics_df = pd.DataFrame(metrics_data)
            csv_path = f"{self.save_dir}/detailed_results/ml_{model_name}_{target_name}_{embeddings_source}_metrics.csv"
            metrics_df.to_csv(csv_path, index=False)
            
            # Save model
            model_path = f"{self.save_dir}/ml_models/{model_name}_{target_name}_{embeddings_source}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(results['model'], f)

    def create_comprehensive_comparison_plots(self, all_results):
        """Create comprehensive comparison plots across all models and phases"""
        print("\nCreating comprehensive comparison plots...")
        
        for target_name, target_results in all_results.items():
            if target_name == 'summary':
                continue
            
            # Collect all results for this target
            comparison_data = []
            
            # GNN results
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    for model_type, results in target_results[phase].items():
                        comparison_data.append({
                            'model': f"{model_type.upper()} ({phase})",
                            'type': 'GNN',
                            'r2': results['avg_metrics']['r2'],
                            'rmse': results['avg_metrics']['rmse'],
                            'mae': results['avg_metrics']['mae'],
                            'mse': results['avg_metrics']['mse']
                        })
            
            # ML results
            if 'ml_models' in target_results:
                embeddings_source = "Best GNN"
                for model_type, results in target_results['ml_models'].items():
                    comparison_data.append({
                        'model': f"{model_type} (on {embeddings_source} embeddings)",
                        'type': 'ML',
                        'r2': results['avg_metrics']['r2'],
                        'rmse': results['avg_metrics']['rmse'],
                        'mae': results['avg_metrics']['mae'],
                        'mse': results['avg_metrics']['mse']
                    })
            
            # Create comprehensive plot
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # Made wider to accommodate more models
            fig.suptitle(f'Comprehensive Model Comparison - {target_name}', fontsize=16)
            
            models = [item['model'] for item in comparison_data]
            r2_scores = [item['r2'] for item in comparison_data]
            rmse_scores = [item['rmse'] for item in comparison_data]
            mae_scores = [item['mae'] for item in comparison_data]
            mse_scores = [item['mse'] for item in comparison_data]
            colors = ['skyblue' if item['type'] == 'GNN' else 'orange' for item in comparison_data]
            
            # R² comparison
            bars1 = axes[0, 0].bar(range(len(models)), r2_scores, color=colors, alpha=0.7)
            axes[0, 0].set_title('R² Score Comparison')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_xticks(range(len(models)))
            axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels with MSE
            for bar, r2_score, mse_score in zip(bars1, r2_scores, mse_scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'R²:{r2_score:.3f}\nMSE:{mse_score:.3f}', ha='center', va='bottom', fontsize=7)
            
            # RMSE comparison
            bars2 = axes[0, 1].bar(range(len(models)), rmse_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_xticks(range(len(models)))
            axes[0, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add MSE labels on RMSE bars
            for bar, rmse_score, mse_score in zip(bars2, rmse_scores, mse_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'RMSE:{rmse_score:.3f}\nMSE:{mse_score:.3f}', ha='center', va='bottom', fontsize=7)
            
            # MAE comparison
            bars3 = axes[1, 0].bar(range(len(models)), mae_scores, color=colors, alpha=0.7)
            axes[1, 0].set_title('MAE Comparison')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].set_xticks(range(len(models)))
            axes[1, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add MSE labels on MAE bars
            for bar, mae_score, mse_score in zip(bars3, mae_scores, mse_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'MAE:{mae_score:.3f}\nMSE:{mse_score:.3f}', ha='center', va='bottom', fontsize=7)
            
            # MSE comparison (new plot)
            bars4 = axes[1, 1].bar(range(len(models)), mse_scores, color=colors, alpha=0.7)
            axes[1, 1].set_title('MSE Comparison')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].set_xticks(range(len(models)))
            axes[1, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add MSE value labels
            for bar, mse_score in zip(bars4, mse_scores):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{mse_score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='GNN Models'),
                             Patch(facecolor='orange', alpha=0.7, label='ML Models')]
            axes[0, 0].legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.save_dir}/plots/comprehensive_comparison_{target_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Comprehensive comparison plot saved: {plot_path}")

    def _create_gnn_model_with_params(self, model_type, hidden_dim, num_targets=1):
        """Create a GNN model with specific hidden_dim parameter"""
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
                num_heads=8,
                estimate_uncertainty=False
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def _train_and_evaluate_once_with_params(self, model, data_list, train_idx, val_idx, target_idx, 
                                           hidden_dim=None, max_epochs=None):
        """Train model with specific parameters and evaluate"""
        if max_epochs is None:
            max_epochs = min(self.num_epochs, 50)  # Shorter training for inner CV
        
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Split data and move to device
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        # Move all data to the correct device
        train_data = self._move_data_to_device(train_data)
        val_data = self._move_data_to_device(val_data)
        
        # Create data loaders
        batch_size = min(self.batch_size, len(train_data) // 4)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            total_train_loss = 0
            
            for batch_data in train_loader:
                # Data is already on device from _move_data_to_device
                optimizer.zero_grad()
                
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                loss = criterion(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item() * batch_data.num_graphs
            
            # Validation
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    # Data is already on device from _move_data_to_device
                    out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    target = batch_data.y[:, target_idx].view(-1, 1)
                    loss = criterion(out, target)
                    total_val_loss += loss.item() * batch_data.num_graphs
            
            avg_val_loss = total_val_loss / len(val_loader.dataset)
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= min(self.patience, 10):  # Shorter patience for inner CV
                    break
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Data is already on device from _move_data_to_device
                out, feat = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                target = batch_data.y[:, target_idx].view(-1, 1)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        
        r2 = r2_score(all_targets, all_preds)
        return r2

    def save_comprehensive_hyperparameter_tracking(self, all_results):
        """Save comprehensive hyperparameter tracking for the entire pipeline"""
        print("\nSaving comprehensive hyperparameter tracking...")
        
        # Create comprehensive hyperparameter tracking
        tracking_data = []
        
        for target_name, target_results in all_results.items():
            if target_name == 'summary':
                continue
                
            # Track initial pipeline parameters
            initial_params = {
                'target_name': target_name,
                'phase': 'pipeline_initial',
                'model_type': 'pipeline',
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
                'use_enhanced_training': self.use_enhanced_training,
                'adaptive_hyperparameters': self.adaptive_hyperparameters,
                'use_nested_cv': self.use_nested_cv
            }
            tracking_data.append(initial_params)
            
            # Track GNN model hyperparameters
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    for model_type, results in target_results[phase].items():
                        if 'fold_results' in results:
                            for fold_result in results['fold_results']:
                                if 'best_params' in fold_result:
                                    best_params = fold_result['best_params']
                                    fold_tracking = {
                                        'target_name': target_name,
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
        
        # Save comprehensive tracking
        tracking_df = pd.DataFrame(tracking_data)
        csv_path = f"{self.save_dir}/comprehensive_hyperparameter_tracking.csv"
        tracking_df.to_csv(csv_path, index=False)
        
        # Create hyperparameter selection frequency analysis
        self._create_hyperparameter_frequency_analysis(all_results)
        
        print(f"Comprehensive hyperparameter tracking saved: {csv_path}")
        return csv_path
    
    def _create_hyperparameter_frequency_analysis(self, all_results):
        """Create detailed hyperparameter frequency analysis"""
        frequency_data = []
        
        for target_name, target_results in all_results.items():
            if target_name == 'summary':
                continue
                
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    for model_type, results in target_results[phase].items():
                        if 'fold_results' in results:
                            # Collect all hyperparameter selections for this model/phase
                            hidden_dims = []
                            k_neighbors_list = []
                            
                            for fold_result in results['fold_results']:
                                if 'best_params' in fold_result:
                                    best_params = fold_result['best_params']
                                    hidden_dims.append(best_params.get('hidden_dim', 'N/A'))
                                    k_neighbors_list.append(best_params.get('k_neighbors', 'N/A'))
                            
                            # Calculate frequencies
                            from collections import Counter
                            hidden_dim_counts = Counter(hidden_dims)
                            k_neighbors_counts = Counter(k_neighbors_list)
                            
                            # Add frequency data
                            for value, count in hidden_dim_counts.items():
                                frequency_data.append({
                                    'target_name': target_name,
                                    'phase': phase,
                                    'model_type': model_type,
                                    'parameter': 'hidden_dim',
                                    'value': value,
                                    'frequency': count,
                                    'total_folds': len(hidden_dims),
                                    'percentage': f"{(count/len(hidden_dims))*100:.1f}%"
                                })
                            
                            for value, count in k_neighbors_counts.items():
                                frequency_data.append({
                                    'target_name': target_name,
                                    'phase': phase,
                                    'model_type': model_type,
                                    'parameter': 'k_neighbors',
                                    'value': value,
                                    'frequency': count,
                                    'total_folds': len(k_neighbors_list),
                                    'percentage': f"{(count/len(k_neighbors_list))*100:.1f}%"
                                })
        
        # Save frequency analysis
        frequency_df = pd.DataFrame(frequency_data)
        csv_path = f"{self.save_dir}/hyperparameter_frequency_analysis.csv"
        frequency_df.to_csv(csv_path, index=False)
        
        print(f"Hyperparameter frequency analysis saved: {csv_path}")
        return csv_path


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("MIXED EMBEDDING PIPELINE WITH NESTED CV HYPERPARAMETER TUNING")
    print("="*80)
    print("This pipeline trains all 3 GNN models with nested CV hyperparameter tuning,")
    print("selects best for explainer, then trains all 3 models again on sparsified graph")
    
    # Create mixed pipeline with nested CV enabled
    mixed_pipeline = MixedEmbeddingPipeline(
        data_path="Data/New_Data.csv",
        k_neighbors=10,
        hidden_dim=64,
        num_epochs=100,  # Reduced for faster testing
        num_folds=5,
        save_dir="./mixed_embedding_results_nested_cv",
        graph_mode='family',
        importance_threshold=0.2,
        use_enhanced_training=True,
        adaptive_hyperparameters=True,
        use_nested_cv=True,  # Enable nested CV hyperparameter tuning
    )
    
    # Run mixed pipeline
    results = mixed_pipeline.run_pipeline()
    
    print("\n" + "="*80)
    print("MIXED PIPELINE WITH NESTED CV ANALYSIS COMPLETE!")
    print("="*80)
    print("Results saved to: ./mixed_embedding_results_nested_cv/")
    print("\nKey features of this pipeline:")
    print("- Uses NESTED CROSS-VALIDATION for hyperparameter tuning")
    print("- Trains ALL 3 GNN models (GCN, RGGC, GAT) on KNN graph with tuned hyperparameters")
    print("- Selects BEST model for GNNExplainer sparsification")
    print("- Trains ALL 3 GNN models again on explainer-sparsified graph with tuned hyperparameters")
    print("- Selects BEST model from explainer-trained models for embedding extraction")
    print("- Extracts embeddings from best explainer-trained model")
    print("- Trains ML models (LinearSVR, ExtraTrees, RandomForest, XGBoost, LightGBM) on embeddings")
    print("- Provides comprehensive comparison across all models and phases")
    print("\nHyperparameter search space:")
    print(f"- hidden_dim: {mixed_pipeline.gnn_hyperparams['hidden_dim']}")
    print(f"- k_neighbors: {mixed_pipeline.gnn_hyperparams['k_neighbors']}")
    print(f"- Total combinations: {len(mixed_pipeline.param_grid)}")