import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
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

class EmbeddingPipeline:
    """
    Complete pipeline for graph-based regression using GNN embeddings with ML models.
    
    Pipeline Flow:
    1. Generate graph from data
    2. Create KNN graph sparsification
    3. Train GNN models (plus versions for embeddings)
    4. Use GNNExplainer to get sparsified graph
    5. Train GNN models on sparsified graph
    6. Extract embeddings from best GNN model
    7. Train ML models (LinearSVR, ExtraTrees) on embeddings with 5-fold CV
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
                 save_dir='./embedding_results',
                 importance_threshold=0.3,  # Use default threshold - pipeline_explainer has adaptive thresholding
                 use_fast_correlation=True,
                 graph_mode='family',  # Changed default to family
                 family_filter_mode='relaxed',
                 single_model_type=None):
        """
        Initialize the embedding pipeline
        
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
            importance_threshold: Threshold for edge importance in GNNExplainer sparsification (lowered to 0.1)
            use_fast_correlation: If True, use fast correlation-based graph construction
            graph_mode: Mode for graph construction ('otu' or 'family') - now defaults to 'family'
            family_filter_mode: Mode for family filtering ('relaxed' or 'strict')
            single_model_type: If specified ('gcn', 'rggc', 'gat'), only use this model type throughout the entire pipeline
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
        self.single_model_type = single_model_type
        
        # Determine which models to train
        if single_model_type:
            if single_model_type not in ['gcn', 'rggc', 'gat']:
                raise ValueError(f"single_model_type must be one of ['gcn', 'rggc', 'gat'], got {single_model_type}")
            self.gnn_models_to_train = [single_model_type]
            print(f"Pipeline configured for {single_model_type.upper()}-ONLY flow")
        else:
            self.gnn_models_to_train = ['gcn', 'rggc', 'gat']
            print("Pipeline configured for MIXED model comparison")
        
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
            model = simple_GAT_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                num_heads=4,
                estimate_uncertainty=False
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model

    def train_gnn_model(self, model_type, target_idx, data_list=None):
        """Train a single GNN model and return the trained model with metrics"""
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
        
        # Calculate average metrics
        avg_metrics = {
            'mse': np.mean([r['mse'] for r in fold_results]),
            'rmse': np.mean([r['rmse'] for r in fold_results]),
            'r2': np.mean([r['r2'] for r in fold_results]),
            'mae': np.mean([r['mae'] for r in fold_results])
        }
        
        print(f"  Average - MSE: {avg_metrics['mse']:.4f}, RMSE: {avg_metrics['rmse']:.4f}, R²: {avg_metrics['r2']:.4f}, MAE: {avg_metrics['mae']:.4f}")
        
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
            'avg_metrics': avg_metrics,
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
        """Train ML models (LinearSVR and ExtraTrees) on embeddings with 5-fold CV"""
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
            ])
        }
        
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
            
            # Calculate average metrics
            avg_metrics = {
                'mse': np.mean([r['mse'] for r in fold_results]),
                'rmse': np.mean([r['rmse'] for r in fold_results]),
                'r2': np.mean([r['r2'] for r in fold_results]),
                'mae': np.mean([r['mae'] for r in fold_results])
            }
            
            print(f"    Average - MSE: {avg_metrics['mse']:.4f}, RMSE: {avg_metrics['rmse']:.4f}, R²: {avg_metrics['r2']:.4f}, MAE: {avg_metrics['mae']:.4f}")
            
            # Train final model on all data
            final_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', ml_models[model_name]['regressor'])
            ])
            final_model.fit(embeddings, target_values)
            
            ml_results[model_name] = {
                'model': final_model,
                'fold_results': fold_results,
                'avg_metrics': avg_metrics,
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
        """Create comprehensive plots comparing all results"""
        target_name = self.target_names[target_idx]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Results Comparison for {target_name}', fontsize=16)
        
        # Plot 1: GNN Model Comparison (R² scores)
        ax1 = axes[0, 0]
        gnn_models = list(gnn_results.keys())
        gnn_r2_scores = [gnn_results[model]['avg_metrics']['r2'] for model in gnn_models]
        gnn_mse_scores = [gnn_results[model]['avg_metrics']['mse'] for model in gnn_models]
        
        bars1 = ax1.bar(gnn_models, gnn_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('GNN Models R² Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, max(gnn_r2_scores) * 1.1)
        
        # Add value labels on bars with MSE
        for bar, score, mse in zip(bars1, gnn_r2_scores, gnn_mse_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'R²:{score:.3f}\nMSE:{mse:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: ML Model Comparison (R² scores)
        ax2 = axes[0, 1]
        ml_models = list(ml_results.keys())
        ml_r2_scores = [ml_results[model]['avg_metrics']['r2'] for model in ml_models]
        ml_mse_scores = [ml_results[model]['avg_metrics']['mse'] for model in ml_models]
        
        bars2 = ax2.bar(ml_models, ml_r2_scores, color=['orange', 'purple'])
        ax2.set_title('ML Models on Embeddings R² Comparison')
        ax2.set_ylabel('R² Score')
        ax2.set_ylim(0, max(ml_r2_scores) * 1.1)
        
        # Add value labels on bars with MSE
        for bar, score, mse in zip(bars2, ml_r2_scores, ml_mse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'R²:{score:.3f}\nMSE:{mse:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Overall Comparison
        ax3 = axes[0, 2]
        all_models = gnn_models + ml_models
        all_r2_scores = gnn_r2_scores + ml_r2_scores
        all_mse_scores = gnn_mse_scores + ml_mse_scores
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
        
        bars3 = ax3.bar(range(len(all_models)), all_r2_scores, color=colors[:len(all_models)])
        ax3.set_title('All Models R² Comparison')
        ax3.set_ylabel('R² Score')
        ax3.set_ylim(0, max(all_r2_scores) * 1.1)
        ax3.set_xticks(range(len(all_models)))
        ax3.set_xticklabels(all_models, rotation=45, ha='right')
        
        # Add value labels with MSE
        for bar, score, mse in zip(bars3, all_r2_scores, all_mse_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'R²:{score:.3f}\nMSE:{mse:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: RMSE Comparison
        ax4 = axes[1, 0]
        gnn_rmse_scores = [gnn_results[model]['avg_metrics']['rmse'] for model in gnn_models]
        ml_rmse_scores = [ml_results[model]['avg_metrics']['rmse'] for model in ml_models]
        all_rmse_scores = gnn_rmse_scores + ml_rmse_scores
        
        bars4 = ax4.bar(range(len(all_models)), all_rmse_scores, color=colors[:len(all_models)])
        ax4.set_title('All Models RMSE Comparison')
        ax4.set_ylabel('RMSE')
        ax4.set_xticks(range(len(all_models)))
        ax4.set_xticklabels(all_models, rotation=45, ha='right')
        
        # Add MSE values on RMSE bars
        for bar, rmse, mse in zip(bars4, all_rmse_scores, all_mse_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'RMSE:{rmse:.3f}\nMSE:{mse:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Prediction scatter plot for best model
        ax5 = axes[1, 1]
        
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
        for fold_result in best_results['fold_results']:
            all_preds.extend(fold_result['predictions'])
            all_targets.extend(fold_result['targets'])
        
        ax5.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
        
        # Add diagonal line
        min_val = min(min(all_targets), min(all_preds))
        max_val = max(max(all_targets), max(all_preds))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax5.set_title(f'Best Model: {best_model_name}\nR² = {max(all_r2_scores):.3f}, MSE = {best_mse:.3f}')
        ax5.set_xlabel('True Values')
        ax5.set_ylabel('Predicted Values')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Cross-validation consistency (R² across folds)
        ax6 = axes[1, 2]
        
        # Plot R² scores across folds for each model
        fold_numbers = range(1, self.num_folds + 1)
        
        for i, model_name in enumerate(all_models):
            if model_name in gnn_results:
                fold_r2s = [fold['r2'] for fold in gnn_results[model_name]['fold_results']]
                model_mse = gnn_results[model_name]['avg_metrics']['mse']
            else:
                fold_r2s = [fold['r2'] for fold in ml_results[model_name]['fold_results']]
                model_mse = ml_results[model_name]['avg_metrics']['mse']
            
            ax6.plot(fold_numbers, fold_r2s, marker='o', label=f'{model_name} (MSE:{model_mse:.3f})', color=colors[i])
        
        ax6.set_title('Cross-Validation Consistency')
        ax6.set_xlabel('Fold Number')
        ax6.set_ylabel('R² Score')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/{target_name}_comprehensive_results.png", dpi=300, bbox_inches='tight')
        plt.close()

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
                        summary_data.append({
                            'target': target_name,
                            'phase': phase,
                            'model_type': model_type,
                            'model_category': 'GNN',
                            'mse': results['avg_metrics']['mse'],
                            'rmse': results['avg_metrics']['rmse'],
                            'r2': results['avg_metrics']['r2'],
                            'mae': results['avg_metrics']['mae']
                        })
            
            # ML results
            if 'ml_models' in target_results:
                for model_type, results in target_results['ml_models'].items():
                    summary_data.append({
                        'target': target_name,
                        'phase': 'embeddings',
                        'model_type': model_type,
                        'model_category': 'ML',
                        'mse': results['avg_metrics']['mse'],
                        'rmse': results['avg_metrics']['rmse'],
                        'r2': results['avg_metrics']['r2'],
                        'mae': results['avg_metrics']['mae']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.save_dir}/results_summary.csv", index=False)
        
        print("Results saved successfully!")

    def run_pipeline(self):
        """
        Run the complete embedding pipeline:
        1. Train GNN models on KNN-sparsified graph
        2. Create GNNExplainer-sparsified graph using best model
        3. Train GNN models on explainer-sparsified graph
        4. Extract embeddings from best GNN model
        5. Train ML models on embeddings with 5-fold CV
        6. Compare and analyze all results
        
        If single_model_type is specified, the entire pipeline uses only that model type.
        """
        print("\n" + "="*80)
        if self.single_model_type:
            print(f"EMBEDDING PIPELINE - {self.single_model_type.upper()}-ONLY ANALYSIS")
        else:
            print("EMBEDDING PIPELINE - COMPREHENSIVE GNN + ML ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        # Process each target variable
        for target_idx, target_name in enumerate(self.target_names):
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET: {target_name} ({target_idx + 1}/{len(self.target_names)})")
            print(f"{'='*60}")
            
            target_results = {}
            
            # Step 1: Train GNN models on KNN-sparsified graph
            print(f"\nSTEP 1: Training GNN models on KNN-sparsified graph")
            if self.single_model_type:
                print(f"Training only {self.single_model_type.upper()} model")
            else:
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
            
            # Find best GNN model for this target (or use the single model)
            if self.single_model_type:
                best_gnn_model = knn_results[self.single_model_type]['model']
                best_gnn_r2 = knn_results[self.single_model_type]['avg_metrics']['r2']
                best_gnn_type = self.single_model_type
                print(f"\nUsing {best_gnn_type.upper()} model (R² = {best_gnn_r2:.4f})")
            else:
                best_gnn_model = None
                best_gnn_r2 = -float('inf')
                best_gnn_type = None
                
                for model_type, results in knn_results.items():
                    if results['avg_metrics']['r2'] > best_gnn_r2:
                        best_gnn_r2 = results['avg_metrics']['r2']
                        best_gnn_model = results['model']
                        best_gnn_type = model_type
                
                print(f"\nBest GNN model: {best_gnn_type.upper()} (R² = {best_gnn_r2:.4f})")
            
            # Step 2: Create GNNExplainer-sparsified graph
            print(f"\nSTEP 2: Creating GNNExplainer-sparsified graph")
            print(f"Using {best_gnn_type.upper()} model for explanation")
            print("-" * 50)
            
            explainer_data = self.create_explainer_sparsified_graph(
                model=best_gnn_model,
                target_idx=target_idx
            )
            
            # Step 3: Train GNN models on explainer-sparsified graph
            print(f"\nSTEP 3: Training GNN models on explainer-sparsified graph")
            if self.single_model_type:
                print(f"Training only {self.single_model_type.upper()} model")
            else:
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
            
            # Find best model overall (KNN vs Explainer)
            if self.single_model_type:
                # Compare only the single model type between KNN and explainer phases
                knn_r2 = knn_results[self.single_model_type]['avg_metrics']['r2']
                explainer_r2 = explainer_results[self.single_model_type]['avg_metrics']['r2']
                
                if explainer_r2 > knn_r2:
                    best_overall_model = explainer_results[self.single_model_type]['model']
                    best_overall_r2 = explainer_r2
                    best_overall_type = f"{self.single_model_type}_explainer"
                    embedding_data = explainer_data
                    print(f"\nBest {self.single_model_type.upper()} model: Explainer-sparsified (R² = {best_overall_r2:.4f})")
                else:
                    best_overall_model = knn_results[self.single_model_type]['model']
                    best_overall_r2 = knn_r2
                    best_overall_type = self.single_model_type
                    embedding_data = self.dataset.data_list
                    print(f"\nBest {self.single_model_type.upper()} model: KNN-sparsified (R² = {best_overall_r2:.4f})")
            else:
                # Compare all models across both phases
                all_gnn_results = {**knn_results, **{f"{k}_explainer": v for k, v in explainer_results.items()}}
                best_overall_model = None
                best_overall_r2 = -float('inf')
                best_overall_type = None
                
                for model_name, results in all_gnn_results.items():
                    if results['avg_metrics']['r2'] > best_overall_r2:
                        best_overall_r2 = results['avg_metrics']['r2']
                        best_overall_model = results['model']
                        best_overall_type = model_name
                
                # Use the appropriate data list based on best model
                if 'explainer' in best_overall_type:
                    embedding_data = explainer_data
                else:
                    embedding_data = self.dataset.data_list
                
                print(f"\nBest overall GNN model: {best_overall_type} (R² = {best_overall_r2:.4f})")
            
            # Step 4: Extract embeddings from best model
            print(f"\nSTEP 4: Extracting embeddings from best {best_gnn_type.upper()} model")
            print("-" * 50)
            
            embeddings, targets = self.extract_embeddings(best_overall_model, embedding_data)
            
            # Save embeddings with model type info
            if self.single_model_type:
                embedding_filename = f"{target_name}_{self.single_model_type}_embeddings.npy"
                targets_filename = f"{target_name}_{self.single_model_type}_targets.npy"
            else:
                embedding_filename = f"{target_name}_embeddings.npy"
                targets_filename = f"{target_name}_targets.npy"
            
            np.save(f"{self.save_dir}/embeddings/{embedding_filename}", embeddings)
            np.save(f"{self.save_dir}/embeddings/{targets_filename}", targets)
            
            print(f"Extracted embeddings shape: {embeddings.shape}")
            print(f"Saved as: {embedding_filename}")
            
            # Step 5: Train ML models on embeddings
            print(f"\nSTEP 5: Training ML models on {best_gnn_type.upper()} embeddings")
            print("-" * 50)
            
            ml_results = self.train_ml_models(embeddings, targets, target_idx)
            target_results['ml_models'] = ml_results
            
            # Determine embeddings source for naming
            embeddings_source = f"{best_gnn_type.upper()}_{best_overall_type.split('_')[-1] if '_' in best_overall_type else 'knn'}"
            
            # Plot ML model results
            self.plot_ml_model_results(ml_results, target_name, embeddings_source)
            
            # Save ML model results
            self.save_ml_model_results(ml_results, target_name, embeddings_source)
            
            # Step 6: Create comprehensive plots
            print(f"\nSTEP 6: Creating comprehensive plots")
            print("-" * 50)
            
            if self.single_model_type:
                # For single model, show KNN vs Explainer comparison
                gnn_plot_results = {
                    f"{self.single_model_type}_knn": knn_results[self.single_model_type],
                    f"{self.single_model_type}_explainer": explainer_results[self.single_model_type]
                }
            else:
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
                            if self.single_model_type:
                                best_model_info = f"{model_type.upper()} ({phase})"
                            else:
                                best_model_info = f"{model_type.upper()} ({phase})"
            
            # Check ML models
            if 'ml_models' in target_results:
                for model_type, results in target_results['ml_models'].items():
                    if results['avg_metrics']['r2'] > best_r2:
                        best_r2 = results['avg_metrics']['r2']
                        if self.single_model_type:
                            best_model_info = f"{model_type} (on {self.single_model_type.upper()} embeddings)"
                        else:
                            best_model_info = f"{model_type} (embeddings)"
            
            print(f"{target_name}: {best_model_info} - R² = {best_r2:.4f}")
        
        print(f"\nAll results saved to: {self.save_dir}")
        print("Check the following files:")
        print(f"  - results_summary.csv: Tabular summary of all results")
        print(f"  - all_results.pkl: Complete results object")
        print(f"  - plots/: Comprehensive visualization plots")
        print(f"  - embeddings/: Extracted embeddings and targets")
        
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
        """Save detailed metrics to CSV"""
        metrics_data = []
        
        for fold_result in fold_results:
            metrics_data.append({
                'fold': fold_result['fold'],
                'model_type': model_type,
                'target_name': target_name,
                'phase': phase,
                'mse': fold_result['mse'],
                'rmse': fold_result['rmse'],
                'r2': fold_result['r2'],
                'mae': fold_result['mae']
            })
        
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
        
        # Add overall metrics
        metrics_data.append({
            'fold': 'overall',
            'model_type': model_type,
            'target_name': target_name,
            'phase': phase,
            'mse': overall_mse,
            'rmse': overall_rmse,
            'r2': overall_r2,
            'mae': overall_mae
        })
        
        # Save to CSV
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = f"{self.save_dir}/detailed_results/{model_type}_{target_name}_{phase}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
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
        colors = ['orange', 'purple']
        
        # Plot 1: R² comparison across folds
        ax1 = axes[0, 0]
        fold_nums = range(1, self.num_folds + 1)
        
        for i, (model_name, results) in enumerate(ml_results.items()):
            fold_r2s = [fold['r2'] for fold in results['fold_results']]
            ax1.plot(fold_nums, fold_r2s, marker='o', label=model_name, color=colors[i], linewidth=2)
            
            # Add average line with MSE
            avg_r2 = results['avg_metrics']['r2']
            avg_mse = results['avg_metrics']['mse']
            ax1.axhline(y=avg_r2, color=colors[i], linestyle='--', alpha=0.7, 
                       label=f'{model_name} Avg = {avg_r2:.3f} (MSE: {avg_mse:.3f})')
        
        ax1.set_xlabel('Fold Number')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Across Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overall performance comparison
        ax2 = axes[0, 1]
        metrics = ['r2', 'rmse', 'mae']
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model_name, results) in enumerate(ml_results.items()):
            values = [results['avg_metrics'][metric] for metric in metrics]
            mse_val = results['avg_metrics']['mse']
            # Normalize RMSE and MAE for better visualization
            if len(values) > 1:
                values[1] = values[1] / max(values[1], 1)  # Normalize RMSE
                values[2] = values[2] / max(values[2], 1)  # Normalize MAE
            
            ax2.bar(x + i*width, values, width, label=f'{model_name} (MSE: {mse_val:.3f})', color=colors[i], alpha=0.7)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Normalized Values')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(metrics)
        ax2.legend()
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
        
        # Plot 4: Error distribution
        ax4 = axes[1, 1]
        for i, (model_name, results) in enumerate(ml_results.items()):
            all_preds = []
            all_targets = []
            for fold_result in results['fold_results']:
                all_preds.extend(fold_result['predictions'])
                all_targets.extend(fold_result['targets'])
            
            errors = np.array(all_targets) - np.array(all_preds)
            mse_val = results['avg_metrics']['mse']
            ax4.hist(errors, bins=20, alpha=0.6, label=f'{model_name} (MSE: {mse_val:.3f})', color=colors[i])
        
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.legend()
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
        ax2.axhline(y=r2, color='red', linestyle='--', linewidth=2, label=f'Average R² = {r2:.3f} (MSE: {mse:.3f})')
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
        ax4.axhline(y=mse, color='red', linestyle='--', linewidth=2, label=f'Average MSE = {mse:.3f}')
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
                            'mae': results['avg_metrics']['mae']
                        })
            
            # ML results
            if 'ml_models' in target_results:
                embeddings_source = f"{self.single_model_type.upper()}" if self.single_model_type else "Best GNN"
                for model_type, results in target_results['ml_models'].items():
                    comparison_data.append({
                        'model': f"{model_type} (on {embeddings_source} embeddings)",
                        'type': 'ML',
                        'r2': results['avg_metrics']['r2'],
                        'rmse': results['avg_metrics']['rmse'],
                        'mae': results['avg_metrics']['mae']
                    })
            
            # Create comprehensive plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Comprehensive Model Comparison - {target_name}', fontsize=16)
            
            models = [item['model'] for item in comparison_data]
            r2_scores = [item['r2'] for item in comparison_data]
            rmse_scores = [item['rmse'] for item in comparison_data]
            mae_scores = [item['mae'] for item in comparison_data]
            colors = ['skyblue' if item['type'] == 'GNN' else 'orange' for item in comparison_data]
            
            # R² comparison
            bars1 = axes[0].bar(range(len(models)), r2_scores, color=colors, alpha=0.7)
            axes[0].set_title('R² Score Comparison')
            axes[0].set_ylabel('R² Score')
            axes[0].set_xticks(range(len(models)))
            axes[0].set_xticklabels(models, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars1, r2_scores):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            # RMSE comparison
            bars2 = axes[1].bar(range(len(models)), rmse_scores, color=colors, alpha=0.7)
            axes[1].set_title('RMSE Comparison')
            axes[1].set_ylabel('RMSE')
            axes[1].set_xticks(range(len(models)))
            axes[1].set_xticklabels(models, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3)
            
            # MAE comparison
            bars3 = axes[2].bar(range(len(models)), mae_scores, color=colors, alpha=0.7)
            axes[2].set_title('MAE Comparison')
            axes[2].set_ylabel('MAE')
            axes[2].set_xticks(range(len(models)))
            axes[2].set_xticklabels(models, rotation=45, ha='right')
            axes[2].grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='GNN Models'),
                             Patch(facecolor='orange', alpha=0.7, label='ML Models')]
            axes[0].legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.save_dir}/plots/comprehensive_comparison_{target_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Comprehensive comparison plot saved: {plot_path}")


# Example usage
if __name__ == "__main__":
    # Option 1: GAT-ONLY Pipeline (entire flow uses only GAT) with family-level nodes
    print("="*60)
    print("OPTION 1: GAT-ONLY PIPELINE (FAMILY-LEVEL NODES)")
    print("="*60)
    
    gat_pipeline = EmbeddingPipeline(
        data_path="../Data/New_data.csv",
        k_neighbors=10,
        hidden_dim=64,
        num_epochs=50,  # Reduced for faster testing
        num_folds=5,
        save_dir="./gat_family_results",
        single_model_type='gat',  # This ensures GAT-only flow
        graph_mode='family',  # Use family-level nodes
        importance_threshold=0.3,  # Use default threshold - pipeline_explainer has adaptive thresholding
    )
    
    # Run GAT-only pipeline
    gat_results = gat_pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("OPTION 2: MIXED PIPELINE (FAMILY-LEVEL NODES)")
    print("="*60)
    
    # Option 2: Mixed Pipeline (compare all models) with family-level nodes
    mixed_pipeline = EmbeddingPipeline(
        data_path="../Data/New_data.csv",
        k_neighbors=10,
        hidden_dim=64,
        num_epochs=50,  # Reduced for faster testing
        num_folds=5,
        save_dir="./mixed_family_results",
        graph_mode='family',  # Use family-level nodes
        importance_threshold=0.3,  # Use default threshold - pipeline_explainer has adaptive thresholding
        # single_model_type=None (default) - trains all models
    )
    
    # Run mixed pipeline
    mixed_results = mixed_pipeline.run_pipeline()
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("GAT-only results saved to: ./gat_family_results/")
    print("Mixed results saved to: ./mixed_family_results/")
    print("\nKey improvements:")
    print("- Using family-level nodes instead of OTU-level")
    print("- Improved GNNExplainer sparsification (threshold=0.3)")
    print("- Individual plots for both ExtraTrees and LinearSVR")
    print("- No individual fold plots (only overall results)")
    print("- Better sparsification monitoring and debugging")
    print("\nGAT-only pipeline ensures:")
    print("- Only GAT models are trained")
    print("- GAT model is used for GNNExplainer")
    print("- GAT embeddings are used for ML models")
    print("- End-to-end GAT consistency") 