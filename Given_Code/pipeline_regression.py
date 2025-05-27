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

# Import other modules
from dataset_regression import MicrobialGNNDataset
from explainer_regression import GNNExplainerRegression
from pipeline_training import train_model
from pipeline_explainer import create_explainer_sparsified_graph

# Import the models
from GNNmodelsRegression import (
    simple_GCN_res_regression,
    simple_GCN_res_plus_regression,
    simple_RGGC_regression,
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

# Constants
DEFAULT_SAVE_DIR = './regression_results'

class RegressionPipeline:
    """Complete pipeline for graph-based regression with GNN models"""
    
    def __init__(self, 
                 data_path,
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 model_type='gcn',
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
                 family_filter_mode='relaxed',
                 train_all_models=True,
                 quick_evaluation=False):
        """
        Initialize the regression pipeline
        
        Args:
            data_path: Path to the CSV file with data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
            model_type: Type of GNN model ('gcn', 'gat', 'rggc')
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
            train_all_models: Whether to train all available models (True) or just the specified model_type (False)
            quick_evaluation: If True, do quick evaluation with fewer epochs to select best model, then full training
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.model_type = model_type
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
        self.estimate_uncertainty = estimate_uncertainty
        self.use_fast_correlation = use_fast_correlation
        self.graph_mode = graph_mode
        self.family_filter_mode = family_filter_mode
        self.train_all_models = train_all_models
        self.quick_evaluation = quick_evaluation
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations", exist_ok=True)
        
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

    def create_model(self, model_type, num_targets=1):
        """Create a GNN model based on specified type"""
        if model_type == 'gcn':
            model = simple_GCN_res_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        elif model_type == 'rggc':
            model = simple_RGGC_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        elif model_type == 'gat':
            model = simple_GAT_regression(
                hidden_channels=self.hidden_dim,
                output_dim=num_targets,
                dropout_prob=self.dropout_rate,
                input_channel=1,
                num_heads=4,
                estimate_uncertainty=self.estimate_uncertainty
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def train_model(self, target_idx=None, data_list=None):
        """
        Train a GNN model for regression
        
        Args:
            target_idx: Index of the target variable to predict (if None, predict all targets)
            data_list: List of graph data objects to use (if None, use self.dataset.data_list)
            
        Returns:
            Dictionary with training results
        """
        return train_model(self, target_idx, data_list)
    
    def create_explainer_sparsified_graph(self, model, target_idx=0, importance_threshold=None):
        """
        Create a sparsified graph based on GNNExplainer results
        
        Args:
            model: Trained GNN model
            target_idx: Index of the target variable to explain
            importance_threshold: Threshold for edge importance
            
        Returns:
            List of sparsified graph data objects
        """
        if importance_threshold is None:
            importance_threshold = self.importance_threshold
            
        return create_explainer_sparsified_graph(self, model, target_idx, importance_threshold)

    def run_pipeline(self):
        """
        Run the complete regression pipeline:
        1. Train all models (GCN, RGGC, GAT) on KNN-sparsified graph
        2. Create further sparsified graph using GNNExplainer (using best performing model)
        3. Train all models on GNNExplainer-sparsified graph
        4. Compare and analyze results across all models
        
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Step 1: Train all models on KNN-sparsified graph
        print("\n" + "="*80)
        print("STEP 1: Training all models on KNN-sparsified graph")
        print("="*80)
        
        knn_results = {}
        
        if len(self.target_names) > 1:
            # Train separate models for each target
            for target_idx, target_name in enumerate(self.target_names):
                print(f"\n{'-'*60}")
                print(f"Training models for target: {target_name}")
                print(f"{'-'*60}")
                if self.train_all_models:
                    knn_results[target_name] = self.train_all_models_method(target_idx=target_idx)
                else:
                    knn_results[target_name] = {self.model_type: self.train_model(target_idx=target_idx)}
        else:
            # Only one target, train models
            target_name = self.target_names[0]
            print(f"\n{'-'*60}")
            print(f"Training models for target: {target_name}")
            print(f"{'-'*60}")
            if self.train_all_models:
                knn_results[target_name] = self.train_all_models_method(target_idx=0)
            else:
                knn_results[target_name] = {self.model_type: self.train_model(target_idx=0)}
        
        # Step 2: Create GNNExplainer-sparsified graph using best performing model
        print("\n" + "="*80)
        print("STEP 2: Creating GNNExplainer-sparsified graph")
        print("="*80)
        
        # Find the best performing model across all targets and model types
        best_model_info = self._find_best_model(knn_results)
        print(f"Using {best_model_info['model_type'].upper()} model for GNNExplainer (best R² = {best_model_info['r2']:.4f})")
        
        # Create sparsified graph data using GNNExplainer
        explainer_sparsified_data = self.create_explainer_sparsified_graph(
            best_model_info['model'], 
            target_idx=best_model_info['target_idx'], 
            importance_threshold=self.importance_threshold
        )
        
        # Visualize the graphs
        self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
        # Step 3: Train all models on GNNExplainer-sparsified graph
        print("\n" + "="*80)
        print("STEP 3: Training all models on GNNExplainer-sparsified graph")
        print("="*80)
        
        explainer_results = {}
        
        if len(self.target_names) > 1:
            # Train separate models for each target
            for target_idx, target_name in enumerate(self.target_names):
                print(f"\n{'-'*60}")
                print(f"Training models for target: {target_name} (GNNExplainer-sparsified graph)")
                print(f"{'-'*60}")
                if self.train_all_models:
                    explainer_results[target_name] = self.train_all_models_method(target_idx=target_idx, data_list=explainer_sparsified_data)
                else:
                    explainer_results[target_name] = {self.model_type: self.train_model(target_idx=target_idx, data_list=explainer_sparsified_data)}
        else:
            # Only one target, train models
            target_name = self.target_names[0]
            print(f"\n{'-'*60}")
            print(f"Training models for target: {target_name} (GNNExplainer-sparsified graph)")
            print(f"{'-'*60}")
            if self.train_all_models:
                explainer_results[target_name] = self.train_all_models_method(target_idx=0, data_list=explainer_sparsified_data)
            else:
                explainer_results[target_name] = {self.model_type: self.train_model(target_idx=0, data_list=explainer_sparsified_data)}
        
        # Step 4: Compare results across all models and graph types
        print("\n" + "="*80)
        print("STEP 4: Comparing results across all models and graph types")
        print("="*80)
        
        comparison_results = self._compare_all_results(knn_results, explainer_results)
        
        # Return consolidated results
        return {
            'knn': knn_results,
            'explainer': explainer_results,
            'comparison': comparison_results,
            'best_model_info': best_model_info
        }
    
    def _find_best_model(self, results):
        """Find the best performing model across all targets and model types"""
        best_r2 = -float('inf')
        best_model_info = None
        
        for target_idx, (target_name, target_results) in enumerate(results.items()):
            for model_type, model_results in target_results.items():
                # Calculate average R² across folds
                r2_scores = []
                for fold_result in model_results:
                    for metric in fold_result['metrics']:
                        if metric['target_name'] == target_name:
                            r2_scores.append(metric['r2'])
                
                avg_r2 = np.mean(r2_scores) if r2_scores else -float('inf')
                
                if avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_model_info = {
                        'target_name': target_name,
                        'target_idx': target_idx,
                        'model_type': model_type,
                        'r2': avg_r2,
                        'model': model_results[0]['model']  # Use first fold model
                    }
        
        return best_model_info
    
    def _compare_all_results(self, knn_results, explainer_results):
        """Compare results between KNN and GNNExplainer graphs across all models"""
        comparison_results = {}
        
        for target_name in self.target_names:
            if target_name in knn_results and target_name in explainer_results:
                comparison_results[target_name] = {}
                
                # Get all model types
                model_types = set(knn_results[target_name].keys()) | set(explainer_results[target_name].keys())
                
                for model_type in model_types:
                    if model_type in knn_results[target_name] and model_type in explainer_results[target_name]:
                        # Extract metrics for this model type
                        knn_metrics = []
                        explainer_metrics = []
                        
                        for fold_result in knn_results[target_name][model_type]:
                            for metric in fold_result['metrics']:
                                if metric['target_name'] == target_name:
                                    knn_metrics.append(metric)
                        
                        for fold_result in explainer_results[target_name][model_type]:
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
                        
                        # Store results
                        comparison_results[target_name][model_type] = {
                            'knn': {
                                'mse': knn_mse,
                                'rmse': knn_rmse,
                                'r2': knn_r2
                            },
                            'explainer': {
                                'mse': explainer_mse,
                                'rmse': explainer_rmse,
                                'r2': explainer_r2
                            },
                            'improvement': {
                                'mse': mse_improvement,
                                'rmse': rmse_improvement,
                                'r2': r2_improvement
                            }
                        }
                        
                        print(f"\nResults for target: {target_name}, Model: {model_type.upper()}")
                        print(f"KNN Graph - MSE: {knn_mse:.4f}, RMSE: {knn_rmse:.4f}, R²: {knn_r2:.4f}")
                        print(f"GNNExplainer Graph - MSE: {explainer_mse:.4f}, RMSE: {explainer_rmse:.4f}, R²: {explainer_r2:.4f}")
                        print(f"Improvement - MSE: {mse_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%")
        
        # Create comprehensive comparison plots
        self._plot_comprehensive_comparison(comparison_results)
        
        # Save comprehensive comparison results
        self._save_comprehensive_results(comparison_results)
        
        return comparison_results
    
    def _plot_comprehensive_comparison(self, comparison_results):
        """Create comprehensive comparison plots for all models and targets"""
        import matplotlib.pyplot as plt
        
        # Create a comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison: KNN vs GNNExplainer Graphs', fontsize=16)
        
        # Collect data for plotting
        targets = []
        models = []
        knn_r2 = []
        explainer_r2 = []
        knn_mse = []
        explainer_mse = []
        
        for target_name, target_results in comparison_results.items():
            for model_type, model_results in target_results.items():
                targets.append(f"{target_name}\n{model_type.upper()}")
                models.append(model_type)
                knn_r2.append(model_results['knn']['r2'])
                explainer_r2.append(model_results['explainer']['r2'])
                knn_mse.append(model_results['knn']['mse'])
                explainer_mse.append(model_results['explainer']['mse'])
        
        # R² comparison
        x = np.arange(len(targets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, knn_r2, width, label='KNN Graph', alpha=0.8)
        axes[0, 0].bar(x + width/2, explainer_r2, width, label='GNNExplainer Graph', alpha=0.8)
        axes[0, 0].set_xlabel('Target & Model')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(targets, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE comparison
        axes[0, 1].bar(x - width/2, knn_mse, width, label='KNN Graph', alpha=0.8)
        axes[0, 1].bar(x + width/2, explainer_mse, width, label='GNNExplainer Graph', alpha=0.8)
        axes[0, 1].set_xlabel('Target & Model')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('MSE Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(targets, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R² improvement
        r2_improvements = []
        for target_name, target_results in comparison_results.items():
            for model_type, model_results in target_results.items():
                r2_improvements.append(model_results['improvement']['r2'])
        
        colors = ['green' if imp > 0 else 'red' for imp in r2_improvements]
        axes[1, 0].bar(x, r2_improvements, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Target & Model')
        axes[1, 0].set_ylabel('R² Improvement (%)')
        axes[1, 0].set_title('R² Improvement (GNNExplainer vs KNN)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(targets, rotation=45, ha='right')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # MSE improvement
        mse_improvements = []
        for target_name, target_results in comparison_results.items():
            for model_type, model_results in target_results.items():
                mse_improvements.append(model_results['improvement']['mse'])
        
        colors = ['green' if imp > 0 else 'red' for imp in mse_improvements]
        axes[1, 1].bar(x, mse_improvements, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Target & Model')
        axes[1, 1].set_ylabel('MSE Improvement (%)')
        axes[1, 1].set_title('MSE Improvement (GNNExplainer vs KNN)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(targets, rotation=45, ha='right')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive comparison plot saved to {self.save_dir}/comprehensive_comparison.png")
    
    def _save_comprehensive_results(self, comparison_results):
        """Save comprehensive comparison results to CSV"""
        comparison_data = []
        
        for target_name, target_results in comparison_results.items():
            for model_type, model_results in target_results.items():
                comparison_data.append({
                    'target': target_name,
                    'model': model_type,
                    'knn_mse': model_results['knn']['mse'],
                    'knn_rmse': model_results['knn']['rmse'],
                    'knn_r2': model_results['knn']['r2'],
                    'explainer_mse': model_results['explainer']['mse'],
                    'explainer_rmse': model_results['explainer']['rmse'],
                    'explainer_r2': model_results['explainer']['r2'],
                    'mse_improvement': model_results['improvement']['mse'],
                    'rmse_improvement': model_results['improvement']['rmse'],
                    'r2_improvement': model_results['improvement']['r2']
                })
        
        # Convert to DataFrame and save
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f"{self.save_dir}/comprehensive_comparison_results.csv", index=False)
        
        print(f"Comprehensive comparison results saved to {self.save_dir}/comprehensive_comparison_results.csv")

    def train_all_models_method(self, target_idx=None, data_list=None):
        """
        Train all available GNN models for regression
        
        Args:
            target_idx: Index of the target variable to predict (if None, predict all targets)
            data_list: List of graph data objects to use (if None, use self.dataset.data_list)
            
        Returns:
            Dictionary with training results for all models
        """
        model_types = ['gcn', 'rggc', 'gat']
        all_results = {}
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*60}")
            
            # Temporarily set the model type
            original_model_type = self.model_type
            self.model_type = model_type
            
            # Train the model
            results = self.train_model(target_idx=target_idx, data_list=data_list)
            all_results[model_type] = results
            
            # Restore original model type
            self.model_type = original_model_type
        
        return all_results 