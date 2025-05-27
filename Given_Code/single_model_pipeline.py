import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Import dataset and models
from dataset_regression import MicrobialGNNDataset
from explainer_regression import GNNExplainerRegression
from GNNmodelsRegression import (
    simple_GCN_res_regression,
    simple_RGGC_regression,
    simple_GAT_regression,
    GaussianNLLLoss
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
DEFAULT_SAVE_DIR = './single_model_results'

class SingleModelRegressionPipeline:
    """Single-model regression pipeline - trains only one specified GNN model throughout"""
    
    def __init__(self, 
                 data_path,
                 model_type='gat',  # Single model to use throughout
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
                 save_dir=DEFAULT_SAVE_DIR,
                 importance_threshold=0.3,
                 estimate_uncertainty=False,
                 use_fast_correlation=True,
                 graph_mode='otu',
                 family_filter_mode='relaxed'):
        """
        Initialize the single-model regression pipeline
        
        Args:
            data_path: Path to the CSV file with data
            model_type: Type of GNN model to use throughout ('gcn', 'gat', 'rggc')
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
            estimate_uncertainty: Whether to estimate uncertainty in predictions
            use_fast_correlation: If True, use fast correlation-based graph construction
            graph_mode: Mode for graph construction ('otu' or 'family')
            family_filter_mode: Mode for family filtering ('relaxed' or 'strict')
        """
        self.data_path = data_path
        self.model_type = model_type.lower()
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
        self.importance_threshold = importance_threshold
        self.estimate_uncertainty = estimate_uncertainty
        self.use_fast_correlation = use_fast_correlation
        self.graph_mode = graph_mode
        self.family_filter_mode = family_filter_mode
        
        # Create save directory with k_neighbors suffix like the multi-model pipeline
        self.save_dir = f"{save_dir}-k={k_neighbors}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/graphs", exist_ok=True)
        os.makedirs(f"{self.save_dir}/explanations", exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset from {data_path}...")
        self.dataset = MicrobialGNNDataset(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold,
            use_fast_correlation=use_fast_correlation,
            graph_mode=graph_mode,
            family_filter_mode=family_filter_mode
        )
        
        self.target_names = self.dataset.target_names
        print(f"Target variables: {self.target_names}")
        print(f"Using {model_type.upper()} model throughout the pipeline")

    def create_model(self, num_targets=1):
        """Create the specified GNN model"""
        if self.model_type == 'gcn':
            model = simple_GCN_res_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        elif self.model_type == 'rggc':
            model = simple_RGGC_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        elif self.model_type == 'gat':
            model = simple_GAT_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                num_heads=4,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model

    def train_model(self, target_idx=None, data_list=None, stage="original"):
        """
        Train the GNN model for regression
        
        Args:
            target_idx: Index of the target variable to predict (if None, predict all targets)
            data_list: List of graph data objects to use (if None, use self.dataset.data_list)
            stage: Stage of training ("original" or "sparsified")
            
        Returns:
            Dictionary with training results
        """
        if data_list is None:
            data_list = self.dataset.data_list
        
        # Determine how many targets to predict
        if target_idx is not None:
            target_name = self.target_names[target_idx]
            num_targets = 1
            print(f"\n{'='*60}")
            print(f"Training {self.model_type.upper()} for target: {target_name} ({stage} graph)")
            print(f"{'='*60}")
        else:
            num_targets = len(self.target_names)
            target_idx = list(range(num_targets))
            print(f"\n{'='*60}")
            print(f"Training {self.model_type.upper()} for all {num_targets} targets ({stage} graph)")
            print(f"{'='*60}")
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        
        # Define loss function
        if self.estimate_uncertainty:
            criterion = GaussianNLLLoss()
        else:
            criterion = nn.MSELoss()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
            print(f"\nFold {fold + 1}/{self.num_folds}")
            
            # Create data loaders
            train_data = [data_list[i] for i in train_idx]
            val_data = [data_list[i] for i in val_idx]
            
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
            
            # Create model
            model = self.create_model(num_targets)
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.num_epochs):
                # Training
                model.train()
                epoch_train_loss = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    if isinstance(target_idx, list):
                        # Multi-target
                        targets = batch.y
                    else:
                        # Single target
                        targets = batch.y[:, target_idx].unsqueeze(1)
                    
                    # All models have the same signature: forward(X, edge_index, batch)
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                    
                    if self.estimate_uncertainty and isinstance(outputs, tuple):
                        pred, var = outputs
                        loss = criterion(pred, targets, var)
                    else:
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # Validation
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        
                        if isinstance(target_idx, list):
                            targets = batch.y
                        else:
                            targets = batch.y[:, target_idx].unsqueeze(1)
                        
                        # All models have the same signature: forward(X, edge_index, batch)
                        outputs = model(batch.x, batch.edge_index, batch.batch)
                        
                        if self.estimate_uncertainty and isinstance(outputs, tuple):
                            pred, var = outputs
                            loss = criterion(pred, targets, var)
                        else:
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            loss = criterion(outputs, targets)
                        
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate on validation set
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    
                    if isinstance(target_idx, list):
                        targets = batch.y
                    else:
                        targets = batch.y[:, target_idx].unsqueeze(1)
                    
                    # All models have the same signature: forward(X, edge_index, batch)
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                    
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    all_predictions.append(outputs.cpu())
                    all_targets.append(targets.cpu())
            
            all_predictions = torch.cat(all_predictions, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            
            # Calculate metrics
            fold_metrics = []
            if isinstance(target_idx, list):
                # Multi-target
                for i, t_name in enumerate(self.target_names):
                    pred = all_predictions[:, i]
                    true = all_targets[:, i]
                    
                    mse = mean_squared_error(true, pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(true, pred)
                    
                    fold_metrics.append({
                        'target_name': t_name,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2
                    })
            else:
                # Single target
                pred = all_predictions.flatten()
                true = all_targets.flatten()
                
                mse = mean_squared_error(true, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(true, pred)
                
                fold_metrics.append({
                    'target_name': self.target_names[target_idx],
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                })
            
            # Save model
            model_path = f"{self.save_dir}/models/{self.model_type}_{stage}_fold_{fold+1}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Plot training curves
            if isinstance(target_idx, list):
                # Multi-target case - save curves for each target
                for i, t_name in enumerate(self.target_names):
                    plt.figure(figsize=(10, 6))
                    plt.plot(train_losses, label='Train Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'{self.model_type.upper()} - {t_name} - Fold {fold+1} ({stage} graph)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{t_name}_fold{fold+1}_loss.png", dpi=300)
                    plt.close()
            else:
                # Single target case
                t_name = self.target_names[target_idx]
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{self.model_type.upper()} - {t_name} - Fold {fold+1} ({stage} graph)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{t_name}_fold{fold+1}_loss.png", dpi=300)
                plt.close()
            
            # Plot prediction vs actual for each target
            if isinstance(target_idx, list):
                # Multi-target case
                for i, t_name in enumerate(self.target_names):
                    pred = all_predictions[:, i]
                    true = all_targets[:, i]
                    
                    plt.figure(figsize=(8, 6))
                    plt.scatter(true, pred, alpha=0.6)
                    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', lw=2)
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title(f'{self.model_type.upper()} - {t_name} - Fold {fold+1} ({stage} graph)')
                    
                    # Add R² to the plot
                    r2 = r2_score(true, pred)
                    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{t_name}_fold{fold+1}_pred.png", dpi=300)
                    plt.close()
            else:
                # Single target case
                pred = all_predictions.flatten()
                true = all_targets.flatten()
                t_name = self.target_names[target_idx]
                
                plt.figure(figsize=(8, 6))
                plt.scatter(true, pred, alpha=0.6)
                plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title(f'{self.model_type.upper()} - {t_name} - Fold {fold+1} ({stage} graph)')
                
                # Add R² to the plot
                r2 = r2_score(true, pred)
                plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{t_name}_fold{fold+1}_pred.png", dpi=300)
                plt.close()
            
            fold_results.append({
                'fold': fold + 1,
                'model': model,
                'metrics': fold_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'predictions': all_predictions,
                'targets': all_targets
            })
            
            # Print fold results
            for metric in fold_metrics:
                print(f"Fold {fold+1} - {metric['target_name']}: MSE={metric['mse']:.4f}, RMSE={metric['rmse']:.4f}, R²={metric['r2']:.4f}")
        
        # Create overall summary plots and save metrics for each target
        if isinstance(target_idx, list):
            # Multi-target case
            for i, t_name in enumerate(self.target_names):
                self._create_overall_plots(fold_results, t_name, stage, target_idx=i)
        else:
            # Single target case
            t_name = self.target_names[target_idx]
            self._create_overall_plots(fold_results, t_name, stage, target_idx=target_idx)
        
        return fold_results
    
    def _create_overall_plots(self, fold_results, target_name, stage, target_idx=None):
        """Create overall summary plots and metrics CSV for a specific target"""
        
        # Collect all predictions and targets across folds
        all_fold_predictions = []
        all_fold_targets = []
        all_fold_metrics = []
        
        for fold_result in fold_results:
            if isinstance(target_idx, int):
                # Single target case
                if len(fold_result['predictions'].shape) == 1:
                    pred = fold_result['predictions']
                    true = fold_result['targets']
                else:
                    pred = fold_result['predictions'].flatten()
                    true = fold_result['targets'].flatten()
            else:
                # Multi-target case - extract specific target
                pred = fold_result['predictions'][:, target_idx]
                true = fold_result['targets'][:, target_idx]
            
            all_fold_predictions.extend(pred)
            all_fold_targets.extend(true)
            
            # Find metrics for this target
            for metric in fold_result['metrics']:
                if metric['target_name'] == target_name:
                    all_fold_metrics.append(metric)
                    break
        
        all_fold_predictions = np.array(all_fold_predictions)
        all_fold_targets = np.array(all_fold_targets)
        
        # Create overall prediction vs actual plot
        plt.figure(figsize=(10, 8))
        
        # Plot all predictions
        plt.scatter(all_fold_targets, all_fold_predictions, alpha=0.6, s=50)
        plt.plot([all_fold_targets.min(), all_fold_targets.max()], 
                [all_fold_targets.min(), all_fold_targets.max()], 'r--', lw=2)
        
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Predicted', fontsize=12)
        plt.title(f'{self.model_type.upper()} - {target_name} - Overall Performance ({stage} graph)', fontsize=14)
        
        # Calculate overall metrics
        overall_r2 = r2_score(all_fold_targets, all_fold_predictions)
        overall_rmse = np.sqrt(mean_squared_error(all_fold_targets, all_fold_predictions))
        overall_mse = mean_squared_error(all_fold_targets, all_fold_predictions)
        
        # Add metrics to plot
        metrics_text = f'Overall R² = {overall_r2:.3f}\nOverall RMSE = {overall_rmse:.3f}\nOverall MSE = {overall_mse:.3f}'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{target_name}_overall.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics CSV
        metrics_df = pd.DataFrame(all_fold_metrics)
        metrics_df.to_csv(f"{self.save_dir}/plots/{self.model_type}_{target_name}_metrics.csv", index=False)
        
        print(f"Overall plots and metrics saved for {target_name} ({stage} graph)")
        
        return fold_results

    def create_explainer_sparsified_graph(self, model, target_idx=0):
        """
        Create a sparsified graph based on GNNExplainer results
        
        Args:
            model: Trained GNN model
            target_idx: Index of the target variable to explain
            
        Returns:
            List of sparsified graph data objects
        """
        print(f"\nCreating explainer-based sparsified graph using {self.model_type.upper()}...")
        print(f"Graph mode: {self.graph_mode}")
        print(f"Number of nodes: {len(self.dataset.node_feature_names)}")
        print(f"Importance threshold: {self.importance_threshold}")
        
        # Initialize explainer
        explainer = GNNExplainerRegression(model, device)
        
        # Create a combined edge importance matrix from multiple samples
        num_explain = min(10, len(self.dataset.data_list))
        combined_edge_importance = torch.zeros((len(self.dataset.node_feature_names), len(self.dataset.node_feature_names)), device=device)
        
        for i in range(num_explain):
            data = self.dataset.data_list[i]
            edge_importance_matrix, _ = explainer.explain_graph(
                data, 
                node_names=self.dataset.node_feature_names,
                target_idx=target_idx
            )
            combined_edge_importance += edge_importance_matrix
        
        # Average the importance
        combined_edge_importance /= num_explain
        
        # Print diagnostics
        non_zero_importance = combined_edge_importance[combined_edge_importance > 0]
        print(f"Edge importance statistics:")
        print(f"  Min: {combined_edge_importance.min():.6f}")
        print(f"  Max: {combined_edge_importance.max():.6f}")
        print(f"  Mean: {combined_edge_importance.mean():.6f}")
        print(f"  Non-zero values: {len(non_zero_importance)}")
        
        # Adaptive thresholding
        if self.graph_mode == 'family':
            print("Using adaptive thresholding for family mode...")
            if len(non_zero_importance) > 0:
                percentage_to_keep = self.importance_threshold * 100
                top_percentile = max(5, min(95, 100 - percentage_to_keep))
                threshold_value = torch.quantile(non_zero_importance, top_percentile / 100.0).item()
                print(f"  Keeping top {percentage_to_keep:.0f}% of edges (using {top_percentile:.0f}th percentile)")
                print(f"  Threshold value: {threshold_value:.6f}")
            else:
                threshold_value = self.importance_threshold * 0.1
                print(f"  Using low absolute threshold: {threshold_value:.6f}")
        else:
            if len(non_zero_importance) > 0:
                max_importance = non_zero_importance.max().item()
                if max_importance < self.importance_threshold:
                    threshold_value = max_importance * 0.5
                    print(f"  Max importance ({max_importance:.6f}) < threshold, using {threshold_value:.6f}")
                else:
                    threshold_value = self.importance_threshold
            else:
                threshold_value = self.importance_threshold
        
        # Create sparsified adjacency matrix
        adj_matrix = combined_edge_importance.clone()
        adj_matrix[adj_matrix < threshold_value] = 0
        
        edges_after_threshold = (adj_matrix > 0).sum().item()
        print(f"Edges after thresholding: {edges_after_threshold}")
        
        # Convert to edge format
        num_nodes = len(self.dataset.node_feature_names)
        new_edge_index = []
        new_edge_weight = []
        new_edge_type = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    new_edge_index.append([i, j])
                    new_edge_weight.append(adj_matrix[i, j].item())
                    
                    # Determine edge type
                    corr, _ = pearsonr(self.dataset.feature_matrix[i], self.dataset.feature_matrix[j])
                    new_edge_type.append(1 if corr > 0 else 0)
        
        if len(new_edge_index) == 0:
            print("Warning: No edges meet the importance threshold. Creating empty graph.")
            new_edge_index = torch.empty((2, 0), dtype=torch.long)
            new_edge_weight = torch.empty((0,), dtype=torch.float32)
            new_edge_type = torch.empty((0,), dtype=torch.long)
            num_edges = 0
        else:
            new_edge_index = torch.tensor(new_edge_index).t().contiguous()
            new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
            new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
            num_edges = new_edge_index.shape[1]
        
        print(f"Explainer sparsified graph has {num_edges//2} undirected edges")
        
        # Store sparsified graph data
        self.dataset.explainer_sparsified_graph_data = {
            'edge_index': new_edge_index.clone(),
            'edge_weight': new_edge_weight.clone(),
            'edge_type': new_edge_type.clone()
        }
        
        # Create new data objects
        new_data_list = []
        feature_matrix_samples = self.dataset.feature_matrix.T
        
        for s in range(feature_matrix_samples.shape[0]):
            x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
            targets = torch.tensor(self.dataset.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
            
            data = Data(
                x=x,
                edge_index=new_edge_index,
                edge_weight=new_edge_weight,
                edge_attr=new_edge_weight.view(-1, 1),
                edge_type=new_edge_type,
                y=targets
            )
            new_data_list.append(data)
        
        # Visualize graphs
        self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
        return new_data_list

    def run_pipeline(self):
        """
        Run the complete single-model regression pipeline:
        1. Train specified model on KNN-sparsified graph
        2. Use that model for GNNExplainer sparsification
        3. Train same model on GNNExplainer-sparsified graph
        4. Compare results
        
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*80}")
        print(f"SINGLE-MODEL PIPELINE: {self.model_type.upper()}")
        print(f"{'='*80}")
        
        results = {}
        
        # Step 1: Train model on KNN-sparsified graph
        print(f"\n{'='*80}")
        print(f"STEP 1: Training {self.model_type.upper()} on KNN-sparsified graph")
        print(f"{'='*80}")
        
        knn_results = {}
        for target_idx, target_name in enumerate(self.target_names):
            knn_results[target_name] = self.train_model(target_idx=target_idx, stage="original")
        
        # Step 2: Create GNNExplainer-sparsified graph
        print(f"\n{'='*80}")
        print(f"STEP 2: Creating GNNExplainer-sparsified graph using {self.model_type.upper()}")
        print(f"{'='*80}")
        
        # Use the first target's first fold model for explanation
        target_name = self.target_names[0]
        model = knn_results[target_name][0]['model']
        
        explainer_sparsified_data = self.create_explainer_sparsified_graph(model, target_idx=0)
        
        # Step 3: Train model on GNNExplainer-sparsified graph
        print(f"\n{'='*80}")
        print(f"STEP 3: Training {self.model_type.upper()} on GNNExplainer-sparsified graph")
        print(f"{'='*80}")
        
        explainer_results = {}
        for target_idx, target_name in enumerate(self.target_names):
            explainer_results[target_name] = self.train_model(
                target_idx=target_idx, 
                data_list=explainer_sparsified_data, 
                stage="sparsified"
            )
        
        # Step 4: Compare results
        print(f"\n{'='*80}")
        print(f"STEP 4: Comparing results between KNN and GNNExplainer-sparsified graphs")
        print(f"{'='*80}")
        
        comparison_results = {}
        
        for target_name in self.target_names:
            # Extract metrics
            knn_metrics = []
            explainer_metrics = []
            
            for fold_result in knn_results[target_name]:
                for metric in fold_result['metrics']:
                    if metric['target_name'] == target_name:
                        knn_metrics.append(metric)
            
            for fold_result in explainer_results[target_name]:
                for metric in fold_result['metrics']:
                    if metric['target_name'] == target_name:
                        explainer_metrics.append(metric)
            
            # Calculate average metrics
            knn_mse = np.mean([m['mse'] for m in knn_metrics])
            knn_rmse = np.mean([m['rmse'] for m in knn_metrics])
            knn_r2 = np.mean([m['r2'] for m in knn_metrics])
            
            explainer_mse = np.mean([m['mse'] for m in explainer_metrics])
            explainer_rmse = np.mean([m['rmse'] for m in explainer_metrics])
            explainer_r2 = np.mean([m['r2'] for m in explainer_metrics])
            
            # Calculate improvement
            mse_improvement = (knn_mse - explainer_mse) / knn_mse * 100 if knn_mse != 0 else 0
            rmse_improvement = (knn_rmse - explainer_rmse) / knn_rmse * 100 if knn_rmse != 0 else 0
            r2_improvement = (explainer_r2 - knn_r2) / abs(knn_r2) * 100 if knn_r2 != 0 else 0
            
            comparison_results[target_name] = {
                'knn': {'mse': knn_mse, 'rmse': knn_rmse, 'r2': knn_r2},
                'explainer': {'mse': explainer_mse, 'rmse': explainer_rmse, 'r2': explainer_r2},
                'improvement': {'mse': mse_improvement, 'rmse': rmse_improvement, 'r2': r2_improvement}
            }
            
            print(f"\nResults for target: {target_name}")
            print(f"KNN Graph - MSE: {knn_mse:.4f}, RMSE: {knn_rmse:.4f}, R²: {knn_r2:.4f}")
            print(f"GNNExplainer Graph - MSE: {explainer_mse:.4f}, RMSE: {explainer_rmse:.4f}, R²: {explainer_r2:.4f}")
            print(f"Improvement - MSE: {mse_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%")
        
        # Create comparison plot
        self._plot_comparison_results(comparison_results)
        
        # Save results
        self._save_results(comparison_results)
        
        return {
            'model_type': self.model_type,
            'knn': knn_results,
            'explainer': explainer_results,
            'comparison': comparison_results
        }
    
    def _plot_comparison_results(self, comparison_results):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{self.model_type.upper()} Model: KNN vs GNNExplainer Graphs', fontsize=16)
        
        target_names = list(comparison_results.keys())
        x = np.arange(len(target_names))
        width = 0.35
        
        # Extract metrics
        knn_rmse = [comparison_results[t]['knn']['rmse'] for t in target_names]
        explainer_rmse = [comparison_results[t]['explainer']['rmse'] for t in target_names]
        knn_r2 = [comparison_results[t]['knn']['r2'] for t in target_names]
        explainer_r2 = [comparison_results[t]['explainer']['r2'] for t in target_names]
        
        # RMSE comparison
        axes[0].bar(x - width/2, knn_rmse, width, label='KNN Graph', alpha=0.8)
        axes[0].bar(x + width/2, explainer_rmse, width, label='GNNExplainer Graph', alpha=0.8)
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('RMSE Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(target_names)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1].bar(x - width/2, knn_r2, width, label='KNN Graph', alpha=0.8)
        axes[1].bar(x + width/2, explainer_r2, width, label='GNNExplainer Graph', alpha=0.8)
        axes[1].set_ylabel('R²')
        axes[1].set_title('R² Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(target_names)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save both model-specific and general comparison plots
        plt.savefig(f"{self.save_dir}/{self.model_type}_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/comparison_plot.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to:")
        print(f"  - {self.save_dir}/{self.model_type}_comparison.png")
        print(f"  - {self.save_dir}/comparison_plot.png")
        print(f"  - {self.save_dir}/comprehensive_comparison.png")
    
    def _save_results(self, comparison_results):
        """Save comparison results to CSV in the same format as multi-model pipeline"""
        comparison_data = []
        
        for target_name, results in comparison_results.items():
            comparison_data.append({
                'target': target_name,
                'model': self.model_type,
                'knn_mse': results['knn']['mse'],
                'knn_rmse': results['knn']['rmse'],
                'knn_r2': results['knn']['r2'],
                'explainer_mse': results['explainer']['mse'],
                'explainer_rmse': results['explainer']['rmse'],
                'explainer_r2': results['explainer']['r2'],
                'mse_improvement': results['improvement']['mse'],
                'rmse_improvement': results['improvement']['rmse'],
                'r2_improvement': results['improvement']['r2']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save single model results
        df.to_csv(f"{self.save_dir}/{self.model_type}_results.csv", index=False)
        
        # Also save in the same format as comprehensive comparison (for consistency)
        df.to_csv(f"{self.save_dir}/comprehensive_comparison_results.csv", index=False)
        
        # Create a simplified comparison results file (like the multi-model pipeline)
        simple_comparison = []
        for target_name, results in comparison_results.items():
            simple_comparison.append({
                'target': target_name,
                'knn_mse': results['knn']['mse'],
                'knn_rmse': results['knn']['rmse'],
                'knn_r2': results['knn']['r2'],
                'explainer_mse': results['explainer']['mse'],
                'explainer_rmse': results['explainer']['rmse'],
                'explainer_r2': results['explainer']['r2'],
                'mse_improvement': results['improvement']['mse'],
                'rmse_improvement': results['improvement']['rmse'],
                'r2_improvement': results['improvement']['r2']
            })
        
        simple_df = pd.DataFrame(simple_comparison)
        simple_df.to_csv(f"{self.save_dir}/comparison_results.csv", index=False)
        
        print(f"Results saved to:")
        print(f"  - {self.save_dir}/{self.model_type}_results.csv")
        print(f"  - {self.save_dir}/comprehensive_comparison_results.csv")
        print(f"  - {self.save_dir}/comparison_results.csv")


if __name__ == "__main__":
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Single-Model GNN Regression Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--model_type', type=str, default='gat', choices=['gcn', 'gat', 'rggc'], help='Type of GNN model to use throughout')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for KNN graph sparsification')
    parser.add_argument('--mantel_threshold', type=float, default=0.05, help='P-value threshold for Mantel test')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR, help='Directory to save results')
    parser.add_argument('--importance_threshold', type=float, default=0.3, help='Threshold for edge importance in GNNExplainer')
    parser.add_argument('--estimate_uncertainty', action='store_true', help='Estimate uncertainty in predictions')
    parser.add_argument('--use_fast_correlation', action='store_true', default=True, help='Use fast correlation-based graph construction')
    parser.add_argument('--no-use_fast_correlation', dest='use_fast_correlation', action='store_false', help='Use Mantel test-based graph construction')
    parser.add_argument('--graph_mode', type=str, default='otu', choices=['otu', 'family'], help='Graph construction mode')
    parser.add_argument('--family_filter_mode', type=str, default='relaxed', choices=['strict', 'relaxed', 'permissive'], help='Family filtering strictness')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Single-Model Pipeline parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create and run pipeline
    pipeline = SingleModelRegressionPipeline(
        data_path=args.data_path,
        model_type=args.model_type,
        k_neighbors=args.k_neighbors,
        mantel_threshold=args.mantel_threshold,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        num_folds=args.num_folds,
        save_dir=args.save_dir,
        importance_threshold=args.importance_threshold,
        estimate_uncertainty=args.estimate_uncertainty,
        use_fast_correlation=args.use_fast_correlation,
        graph_mode=args.graph_mode,
        family_filter_mode=args.family_filter_mode
    )
    
    # Run the pipeline
    results = pipeline.run_pipeline()
    
    print(f"\nSingle-model pipeline completed successfully!")
    print(f"Model used: {args.model_type.upper()}")
    print(f"Results saved to {args.save_dir}") 