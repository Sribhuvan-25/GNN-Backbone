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
from domain_expert_cases_pipeline import DomainExpertCasesPipeline, AnchoredMicrobialGNNDataset

# Import GNN models
from GNNmodelsRegression import (
    simple_GCN_res_plus_regression,
    simple_RGGC_plus_regression,
    simple_GAT_regression,
    GaussianNLLLoss
)

# Set device with better GPU detection
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class NestedCVDomainExpertCasesPipeline(DomainExpertCasesPipeline):
    """
    Extended Domain Expert Cases Pipeline with Nested Cross-Validation for GNN hyperparameter tuning
    
    This pipeline implements nested 5-fold CV:
    - Outer loop: 5-fold CV for final performance evaluation
    - Inner loop: 5-fold CV for hyperparameter tuning on training data of each outer fold
    """
    
    def __init__(self, data_path, case_type='case1', 
                 k_neighbors=5, mantel_threshold=0.05,
                 hidden_dim=64, dropout_rate=0.3, batch_size=8,
                 learning_rate=0.001, weight_decay=1e-4,
                 num_epochs=200, patience=20, num_folds=5,
                 save_dir='./domain_expert_results_nested_cv',
                 importance_threshold=0.2,
                 use_fast_correlation=False,
                 graph_mode='family', family_filter_mode='strict',
                 # New parameters for nested CV
                 inner_cv_folds=5,
                 max_hyperparameter_combinations=12):
        
        # Initialize parent class
        super().__init__(
            data_path=data_path,
            case_type=case_type,
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
        
        # Nested CV parameters
        self.inner_cv_folds = inner_cv_folds
        self.max_hyperparameter_combinations = max_hyperparameter_combinations
        
        # Define hyperparameter grids for each GNN model
        self.hyperparameter_grids = self._define_hyperparameter_grids()
        
        print(f"Nested CV Pipeline initialized with {self.num_folds}-fold outer CV and {self.inner_cv_folds}-fold inner CV")
    
    def _define_hyperparameter_grids(self):
        """Define hyperparameter grids for each GNN model type"""
        grids = {
            'gcn': {
                'hidden_dim': [64, 128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            },
            'gat': {
                'hidden_dim': [64, 128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            },
            'rggc': {
                'hidden_dim': [64, 128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            }
        }
        
        # Limit the number of combinations to prevent excessive computation
        for model_type, grid in grids.items():
            full_grid = list(ParameterGrid(grid))
            if len(full_grid) > self.max_hyperparameter_combinations:
                # Randomly sample combinations to limit computational cost
                np.random.seed(42)
                selected_indices = np.random.choice(
                    len(full_grid), 
                    self.max_hyperparameter_combinations, 
                    replace=False
                )
                grids[model_type] = [full_grid[i] for i in selected_indices]
            else:
                grids[model_type] = full_grid
        
        return grids
    
    def train_gnn_model_nested_cv(self, model_type, target_idx, data_list=None):
        """
        Train a single GNN model with nested cross-validation for hyperparameter tuning
        
        Outer loop: 5-fold CV for final performance evaluation
        Inner loop: 5-fold CV for hyperparameter tuning on training data of each outer fold
        """
        if data_list is None:
            data_list = self.dataset.data_list

        target_name = self.target_names[target_idx]
        phase = "knn" if data_list == self.dataset.data_list else "explainer"
        print(f"\nTraining {model_type.upper()} with Nested CV for target {target_name} ({phase})")
        print(f"Hyperparameter combinations to test: {len(self.hyperparameter_grids[model_type])}")

        # Outer cross-validation
        outer_cv = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        outer_fold_results = []
        best_overall_r2 = -float('inf')
        best_overall_model = None
        
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(data_list), start=1):
            print(f"\n  Outer Fold {outer_fold}/{self.num_folds}")
            
            # Split data for outer fold
            outer_train_data = [data_list[i] for i in outer_train_idx]
            outer_test_data = [data_list[i] for i in outer_test_idx]
            
            # Inner cross-validation for hyperparameter tuning
            best_hyperparams = self._inner_cv_hyperparameter_tuning(
                outer_train_data, model_type, target_idx, outer_fold
            )
            
            print(f"    Best hyperparameters for outer fold {outer_fold}: {best_hyperparams}")
            
            # Train final model on full outer training data with best hyperparameters
            final_model = self._train_single_model_with_hyperparams(
                outer_train_data, best_hyperparams, model_type, target_idx
            )
            
            # Evaluate on outer test data
            test_metrics = self._evaluate_model_on_test_data(
                final_model, outer_test_data, target_idx
            )
            
            print(f"    Outer fold {outer_fold} test metrics: R² = {test_metrics['r2']:.4f}, MSE = {test_metrics['mse']:.4f}, RMSE = {test_metrics['rmse']:.4f}")
            
            # Store results (WITHOUT storing the model to save memory)
            outer_fold_results.append({
                'fold': outer_fold,
                'best_hyperparams': best_hyperparams,
                'test_metrics': test_metrics
                # 'model': final_model  # REMOVED - Don't store models to save memory
            })
            
            # Track best overall model (but don't store it)
            if test_metrics['r2'] > best_overall_r2:
                best_overall_r2 = test_metrics['r2']
                # Don't store the actual model to save memory
                # best_overall_model = final_model
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(outer_fold_results)
        
        print(f"\n  {model_type.upper()} Nested CV Results:")
        print(f"    Overall R² = {overall_metrics['r2']:.4f} ± {overall_metrics['r2_std']:.4f}")
        print(f"    Overall MSE = {overall_metrics['mse']:.4f}")
        print(f"    Overall RMSE = {overall_metrics['rmse']:.4f} ± {overall_metrics['rmse_std']:.4f}")
        
        return {
            'model': None,  # Don't return model to save memory
            'fold_results': outer_fold_results,
            'avg_metrics': overall_metrics,
            'hyperparameter_analysis': self._analyze_hyperparameter_selection(outer_fold_results)
        }
    
    def _inner_cv_hyperparameter_tuning(self, train_data, model_type, target_idx, outer_fold):
        """
        Perform inner cross-validation for hyperparameter tuning
        """
        print(f"    Inner CV hyperparameter tuning for outer fold {outer_fold}...")
        
        # Report GPU memory if available
        if torch.cuda.is_available():
            print(f"    GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
        
        inner_cv = KFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=42)
        hyperparameter_results = []
        
        total_combinations = len(self.hyperparameter_grids[model_type])
        
        for idx, hyperparam_combo in enumerate(self.hyperparameter_grids[model_type], 1):
            print(f"      Testing hyperparameter combination {idx}/{total_combinations}: {hyperparam_combo}")
            
            # Test this hyperparameter combination across inner CV folds
            inner_fold_scores = []
            
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_data), start=1):
                inner_train_data = [train_data[i] for i in inner_train_idx]
                inner_val_data = [train_data[i] for i in inner_val_idx]
                
                # Train model with current hyperparameters
                model = self._train_single_model_with_hyperparams(
                    inner_train_data, hyperparam_combo, model_type, target_idx
                )
                
                # Evaluate on inner validation data
                val_metrics = self._evaluate_model_on_test_data(
                    model, inner_val_data, target_idx
                )
                
                inner_fold_scores.append(val_metrics['r2'])
                
                # Clear model from memory immediately
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average score for this hyperparameter combination
            avg_score = np.mean(inner_fold_scores)
            hyperparameter_results.append({
                'hyperparams': hyperparam_combo,
                'avg_r2': avg_score,
                'std_r2': np.std(inner_fold_scores),
                'inner_fold_scores': inner_fold_scores
            })
            
            print(f"        Combination {idx} avg R²: {avg_score:.4f} ± {np.std(inner_fold_scores):.4f}")
        
        # Find best hyperparameters
        best_result = max(hyperparameter_results, key=lambda x: x['avg_r2'])
        
        print(f"      Best inner CV R² = {best_result['avg_r2']:.4f} ± {best_result['std_r2']:.4f}")
        
        return best_result['hyperparams']
    
    def _train_single_model_with_hyperparams(self, train_data, hyperparams, model_type, target_idx):
        """
        Train a single GNN model with specified hyperparameters
        """
        # Create model with specified hyperparameters
        temp_hidden_dim = self.hidden_dim
        temp_dropout_rate = self.dropout_rate
        temp_learning_rate = self.learning_rate
        temp_weight_decay = self.weight_decay
        
        # Update hyperparameters
        self.hidden_dim = hyperparams['hidden_dim']
        self.dropout_rate = hyperparams['dropout_rate']
        
        model = self.create_gnn_model(model_type).to(device)
        
        # Setup optimizer with hyperparameters
        optimizer = Adam(
            model.parameters(), 
            lr=0.01,  # Fixed learning rate
            weight_decay=1e-4  # Fixed weight decay
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Create data loader
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        # Training loop with GPU optimization
        model.train()
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                if model_type == 'gcn':
                    output = model(batch.x, batch.edge_index, batch.batch)
                elif model_type == 'gat':
                    output = model(batch.x, batch.edge_index, batch.batch)
                elif model_type == 'rggc':
                    output = model(batch.x, batch.edge_index, batch.batch)
                
                # Handle tuple output (prediction, embedding) from GNN models
                if isinstance(output, tuple):
                    prediction = output[0]  # Extract prediction from tuple
                else:
                    prediction = output
                
                # Ensure prediction has correct shape
                if len(prediction.shape) == 1:
                    prediction = prediction.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
                
                # Calculate loss
                target = batch.y[:, target_idx].unsqueeze(1)
                loss = nn.MSELoss()(prediction, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Clear intermediate tensors from GPU memory
                if torch.cuda.is_available():
                    del batch, output, prediction, target, loss
                    torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_epoch_loss)
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Restore original hyperparameters
        self.hidden_dim = temp_hidden_dim
        self.dropout_rate = temp_dropout_rate
        self.learning_rate = temp_learning_rate
        self.weight_decay = temp_weight_decay
        
        return model
    
    def _evaluate_model_on_test_data(self, model, test_data, target_idx):
        """
        Evaluate a trained model on test data with GPU optimization
        """
        model.eval()
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        all_targets = []
        
        # Clear GPU cache before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Forward pass
                if hasattr(model, 'forward'):
                    output = model(batch.x, batch.edge_index, batch.batch)
                else:
                    output = model(batch.x, batch.edge_index, batch.batch)
                
                # Handle tuple output (prediction, embedding) from GNN models
                if isinstance(output, tuple):
                    prediction = output[0]  # Extract prediction from tuple
                else:
                    prediction = output
                
                # Ensure prediction has correct shape
                if len(prediction.shape) == 1:
                    prediction = prediction.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
                
                # Store predictions and targets
                all_predictions.extend(prediction.cpu().numpy().flatten())
                all_targets.extend(batch.y[:, target_idx].cpu().numpy().flatten())
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    del batch, output, prediction
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def _calculate_overall_metrics(self, fold_results):
        """
        Calculate overall metrics from all outer fold results
        """
        all_predictions = []
        all_targets = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for fold_result in fold_results:
            test_metrics = fold_result['test_metrics']
            all_predictions.extend(test_metrics['predictions'])
            all_targets.extend(test_metrics['targets'])
            r2_scores.append(test_metrics['r2'])
            rmse_scores.append(test_metrics['rmse'])
            mae_scores.append(test_metrics['mae'])
        
        # Calculate overall metrics
        overall_r2 = r2_score(all_targets, all_predictions)
        overall_mse = mean_squared_error(all_targets, all_predictions)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(all_targets, all_predictions)
        
        return {
            'r2': overall_r2,
            'r2_std': np.std(r2_scores),
            'mse': overall_mse,
            'rmse': overall_rmse,
            'rmse_std': np.std(rmse_scores),
            'mae': overall_mae,
            'mae_std': np.std(mae_scores),
            'fold_predictions': [{
                'fold': fold['fold'],
                'predicted': fold['test_metrics']['predictions'].tolist(),
                'actual': fold['test_metrics']['targets'].tolist()
            } for fold in fold_results]
        }
    
    def _analyze_hyperparameter_selection(self, fold_results):
        """
        Analyze which hyperparameters were selected across folds
        """
        hyperparameter_analysis = {}
        
        for fold_result in fold_results:
            hyperparams = fold_result['best_hyperparams']
            for param, value in hyperparams.items():
                if param not in hyperparameter_analysis:
                    hyperparameter_analysis[param] = []
                hyperparameter_analysis[param].append(value)
        
        # Calculate statistics for each hyperparameter
        analysis_summary = {}
        for param, values in hyperparameter_analysis.items():
            if isinstance(values[0], (int, float)):
                analysis_summary[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            else:
                # For categorical parameters
                from collections import Counter
                counts = Counter(values)
                analysis_summary[param] = {
                    'most_common': counts.most_common(1)[0][0],
                    'distribution': dict(counts),
                    'values': values
                }
        
        return analysis_summary
    
    def create_explainer_sparsified_graph(self, knn_results, target_idx):
        """
        Create explainer-sparsified graph using the best KNN model
        """
        # Find best KNN model
        best_knn_model_type = max(knn_results, key=lambda m: knn_results[m]['avg_metrics']['r2'])
        best_knn_results = knn_results[best_knn_model_type]
        
        print(f"Using best KNN model ({best_knn_model_type.upper()}) for explainer graph creation")
        
        # Get best hyperparameters from the best model
        best_hyperparams = None
        best_fold_r2 = -float('inf')
        for fold_result in best_knn_results['fold_results']:
            if fold_result['test_metrics']['r2'] > best_fold_r2:
                best_fold_r2 = fold_result['test_metrics']['r2']
                best_hyperparams = fold_result['best_hyperparams']
        
        # Retrain the best model on full dataset for explainer
        best_model = self._train_single_model_with_hyperparams(
            self.dataset.data_list, best_hyperparams, best_knn_model_type, target_idx
        )
        
        # Use parent class method to create explainer graph
        explainer_data = self.create_explainer_sparsified_graph_from_model(best_model, target_idx)
        
        return explainer_data
    
    def create_explainer_sparsified_graph_from_model(self, model, target_idx):
        """
        Create explainer-sparsified graph using a trained model (parent class method)
        """
        # This calls the parent class method
        return super().create_explainer_sparsified_graph(model, target_idx)
    
    def extract_embeddings_from_fold_results(self, fold_results):
        """
        Extract embeddings from fold results by retraining the best model
        """
        # Find the best fold
        best_fold = max(fold_results, key=lambda f: f['test_metrics']['r2'])
        best_hyperparams = best_fold['best_hyperparams']
        
        # Retrain the best model on full dataset
        best_model = self._train_single_model_with_hyperparams(
            self.dataset.data_list, best_hyperparams, 'gcn', target_idx=0  # Use GCN as default
        )
        
        # Extract embeddings using parent class method
        embeddings, targets = self.extract_embeddings(best_model, self.dataset.data_list)
        
        return embeddings, targets
    
    def train_ml_models_on_embeddings(self, embeddings, targets, target_idx):
        """
        Train ML models on embeddings using nested CV
        """
        print(f"Training ML models on embeddings with nested CV...")
        
        # Use parent class method for ML training
        ml_results = self.train_ml_models(embeddings, targets, target_idx)
        
        return ml_results

    def _run_single_target_pipeline_nested_cv(self, target_idx, target_name):
        """
        Run nested CV pipeline for a single target - overrides parent method
        """
        print(f"DEBUG: _run_single_target_pipeline_nested_cv called for {target_name}")
        print(f"\nRunning NESTED CV pipeline for {target_name}")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        
        results = {}
        
        # Step 1: Train ALL GNN models with nested CV on KNN-sparsified graph
        print(f"\nSTEP 1: Training GNN models on KNN-sparsified graph with nested CV...")
        knn_gnn_results = {}
        for model_type in ['gcn', 'gat', 'rggc']:
            knn_gnn_results[model_type] = self.train_gnn_model_nested_cv(model_type, target_idx)
        results['knn_nested_cv'] = knn_gnn_results
        
        # Step 2: Create explainer-sparsified graph using best model
        print(f"\nSTEP 2: Creating explainer-sparsified graph using best KNN model...")
        explainer_graph = self.create_explainer_sparsified_graph(knn_gnn_results, target_idx)
        
        # Step 3: Train ALL GNN models with nested CV on explainer-sparsified graph
        print(f"\nSTEP 3: Training GNN models on explainer-sparsified graph with nested CV...")
        explainer_gnn_results = {}
        for model_type in ['gcn', 'gat', 'rggc']:
            explainer_gnn_results[model_type] = self.train_gnn_model_nested_cv(model_type, target_idx, data_list=explainer_graph)
        results['explainer_nested_cv'] = explainer_gnn_results
        
        # Step 4: Extract embeddings from best explainer-trained model (for ML)
        print(f"\nSTEP 4: Extracting embeddings from best explainer-trained model...")
        best_explainer_model_type = max(explainer_gnn_results, key=lambda m: explainer_gnn_results[m]['avg_metrics']['r2'])
        best_explainer_model_results = explainer_gnn_results[best_explainer_model_type]
        embeddings, targets = self.extract_embeddings_from_fold_results(best_explainer_model_results['fold_results'])
        
        # Step 5: Train ML models on embeddings
        print(f"\nSTEP 5: Training ML models on embeddings...")
        ml_results = self.train_ml_models_on_embeddings(embeddings, targets, target_idx)
        results['ml_models'] = ml_results
        
        # --- Convert nested CV results to parent class format for plotting ---
        # The parent expects: gnn_results (dict), ml_results (dict), target_idx
        # gnn_results should have both KNN and explainer results, with explainer models suffixed as in parent
        
        # Convert nested CV fold results to parent class expected format
        def convert_nested_cv_fold_results(fold_results):
            """Convert nested CV fold results to parent class expected format"""
            converted_fold_results = []
            for fold_result in fold_results:
                # Extract metrics from test_metrics and create parent-expected format
                converted_fold = {
                    'r2': fold_result['test_metrics']['r2'],
                    'mse': fold_result['test_metrics']['mse'],
                    'rmse': fold_result['test_metrics']['rmse'],
                    'mae': fold_result['test_metrics']['mae'],
                    'predictions': fold_result['test_metrics']['predictions'],
                    'targets': fold_result['test_metrics']['targets']
                }
                converted_fold_results.append(converted_fold)
            return converted_fold_results
        
        gnn_plot_results = {}
        for model_type, result in knn_gnn_results.items():
            gnn_plot_results[model_type] = {
                'avg_metrics': result['avg_metrics'],
                'fold_results': convert_nested_cv_fold_results(result['fold_results'])
            }
        for model_type, result in explainer_gnn_results.items():
            gnn_plot_results[f"{model_type}_explainer"] = {
                'avg_metrics': result['avg_metrics'],
                'fold_results': convert_nested_cv_fold_results(result['fold_results'])
            }
        
        # --- Call parent class plot_results (standardizes file name/content) ---
        self.plot_results(
            gnn_results=gnn_plot_results,
            ml_results=ml_results,
            target_idx=target_idx
        )
        # --- Save results (summary, hyperparameter analysis, etc.) ---
        self._save_nested_cv_results(results, target_name)
        return results

    def _save_nested_cv_results(self, results, target_name):
        """
        Save nested CV results with hyperparameter analysis
        """
        print(f"DEBUG: _save_nested_cv_results called for {target_name}")
        print(f"DEBUG: Save directory: {self.save_dir}")
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save complete results
        results_file = f"{self.save_dir}/{self.case_type}_{target_name}_nested_cv_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"DEBUG: Saved results to {results_file}")
        
        # Create detailed summary
        summary_data = []
        
        # Process nested CV results
        for phase in ['knn_nested_cv', 'explainer_nested_cv']:
            if phase in results:
                for model_type, result in results[phase].items():
                    summary_data.append({
                        'case': self.case_type,
                        'target': target_name,
                        'phase': phase,
                        'model_type': model_type,
                        'model_category': 'GNN_NestedCV',
                        'mse': result['avg_metrics']['mse'],
                        'rmse': result['avg_metrics']['rmse'],
                        'r2': result['avg_metrics']['r2'],
                        'r2_std': result['avg_metrics']['r2_std'],
                        'mae': result['avg_metrics']['mae'],
                        'num_features': len(self.dataset.node_feature_names),
                        'num_hyperparameter_combinations': len(self.hyperparameter_grids[model_type])
                    })
        
        # Process ML results
        if 'ml_models' in results:
            for model_type, result in results['ml_models'].items():
                summary_data.append({
                    'case': self.case_type,
                    'target': target_name,
                    'phase': 'embeddings',
                    'model_type': model_type,
                    'model_category': 'ML_NestedCV',
                    'mse': result['avg_metrics']['mse'],
                    'rmse': result['avg_metrics']['rmse'],
                    'r2': result['avg_metrics']['r2'],
                    'r2_std': result['avg_metrics'].get('r2_std', 0),
                    'mae': result['avg_metrics']['mae'],
                    'num_features': len(self.dataset.node_feature_names),
                    'num_hyperparameter_combinations': 'N/A'
                })
        
        # Save summary
        summary_file = f"{self.save_dir}/{self.case_type}_{target_name}_nested_cv_summary.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"DEBUG: Saved summary to {summary_file}")
        
        # Save hyperparameter analysis
        self._save_hyperparameter_analysis(results, target_name)
        
        print(f"\nNested CV results saved to {self.save_dir}")
        print(f"DEBUG: Summary contains {len(summary_data)} model results")
    
    def _save_hyperparameter_analysis(self, results, target_name):
        """
        Save detailed hyperparameter analysis
        """
        hyperparameter_analysis = {}
        
        for phase in ['knn_nested_cv', 'explainer_nested_cv']:
            if phase in results:
                hyperparameter_analysis[phase] = {}
                for model_type, result in results[phase].items():
                    if 'hyperparameter_analysis' in result:
                        hyperparameter_analysis[phase][model_type] = result['hyperparameter_analysis']
        
        # Save hyperparameter analysis
        with open(f"{self.save_dir}/{self.case_type}_{target_name}_hyperparameter_analysis.pkl", 'wb') as f:
            pickle.dump(hyperparameter_analysis, f)
        
        # Create readable summary
        with open(f"{self.save_dir}/{self.case_type}_{target_name}_hyperparameter_summary.txt", 'w') as f:
            f.write(f"Hyperparameter Analysis for {self.case_type} - {target_name}\n")
            f.write("="*80 + "\n\n")
            
            for phase, phase_results in hyperparameter_analysis.items():
                f.write(f"{phase.upper()}\n")
                f.write("-"*40 + "\n")
                
                for model_type, analysis in phase_results.items():
                    f.write(f"\n{model_type.upper()} Model:\n")
                    
                    for param, stats in analysis.items():
                        f.write(f"  {param}:\n")
                        if isinstance(stats, dict):
                            if 'mean' in stats:
                                f.write(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                                f.write(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                                f.write(f"    Values across folds: {stats['values']}\n")
                            elif 'most_common' in stats:
                                f.write(f"    Most common: {stats['most_common']}\n")
                                f.write(f"    Distribution: {stats['distribution']}\n")
                        f.write("\n")
                
                f.write("\n")
    
    # Override the main pipeline methods to use nested CV
    def _run_single_target_pipeline(self, target_idx, target_name):
        """Override to use nested CV pipeline"""
        print(f"DEBUG: Nested CV _run_single_target_pipeline called for {target_name}")
        return self._run_single_target_pipeline_nested_cv(target_idx, target_name)
    
    # Override case methods to ensure they use nested CV
    def _run_case1(self):
        """Case 1: Use only hydrogenotrophic features for the H2 dataset - Nested CV version"""
        print("Case 1: Using only hydrogenotrophic features for H2 dataset (Nested CV)")
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
        
        # Run nested CV pipeline for H2 target
        return self._run_single_target_pipeline(h2_target_idx, "H2-km")
    
    def _run_case2(self):
        """Case 2: Use only acetoclastic features for ACE dataset - Nested CV version"""
        print("Case 2: Using only acetoclastic features for ACE dataset (Nested CV)")
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
        
        # Run nested CV pipeline for ACE target
        return self._run_single_target_pipeline(ace_target_idx, "ACE-km")
    
    def _run_case3(self):
        """Case 3: Use all feature groups for ACE dataset - Nested CV version"""
        print("Case 3: Using acetoclastic + hydrogenotrophic + syntrophic features for ACE dataset (Nested CV)")
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
        
        # Run nested CV pipeline for ACE target
        return self._run_single_target_pipeline(ace_target_idx, "ACE-km")


def run_all_cases_nested_cv(data_path="../Data/New_data.csv"):
    """Run all domain expert cases with nested CV"""
    print("Running all domain expert cases with NESTED CROSS-VALIDATION...")
    
    # For testing, only run case1
    cases = ['case1']  # Comment out other cases for testing
    # cases = ['case1', 'case2', 'case3', 'case4', 'case5']  # Uncomment for full run
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"RUNNING {case.upper()} WITH NESTED CV")
        print(f"{'='*60}")
        
        try:
            pipeline = NestedCVDomainExpertCasesPipeline(
                data_path=data_path,
                case_type=case,
                k_neighbors=15,
                hidden_dim=512,  # This will be tuned by nested CV
                num_epochs=200,
                num_folds=5,
                inner_cv_folds=5,
                save_dir="./domain_expert_results_nested_cv",
                importance_threshold=0.2,
                use_fast_correlation=False,
                family_filter_mode='strict',
                max_hyperparameter_combinations=12
            )
            
            case_results = pipeline.run_case_specific_pipeline()
            all_results[case] = case_results
            
        except Exception as e:
            print(f"Error running {case} with nested CV: {e}")
            import traceback
            traceback.print_exc()
            all_results[case] = None
    
    # Save combined results
    with open("./domain_expert_results_nested_cv/all_cases_nested_cv_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Print quick summary
    if results and 'knn_nested_cv' in results:
        print("Quick Test Results Summary:")
        for model_type, model_results in results['knn_nested_cv'].items():
            if model_results and 'avg_metrics' in model_results:
                r2 = model_results['avg_metrics']['r2']
                r2_std = model_results['avg_metrics']['r2_std']
                mse = model_results['avg_metrics']['mse']
                rmse = model_results['avg_metrics']['rmse']
                print(f"  {model_type.upper()}: R² = {r2:.4f}±{r2_std:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}")
    
    print("All cases with nested CV completed!")
    return all_results


if __name__ == "__main__":
    # Run all cases with nested CV
    results = run_all_cases_nested_cv()
    print("Domain expert cases pipeline with nested CV completed!")