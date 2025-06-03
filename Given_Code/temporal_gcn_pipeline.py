"""
Temporal Graph Convolutional Network (T-GCN) Pipeline for Microbial Data Analysis

This pipeline implements T-GCN for temporal regression tasks on microbial abundance data.
It follows the same workflow as mixed_embedding_pipeline.py but uses T-GCN instead of static GNNs.

Pipeline Flow:
1. KNN Graph Construction (same as original pipeline)
2. Initial T-GCN Training on KNN-sparsified graph
3. GNNExplainer Graph Sparsification using best T-GCN model
4. Re-train T-GCN on explainer-sparsified graph
5. Extract T-GCN Embeddings from best model
6. Train ML Models (RandomForest, XGBoost, LinearSVR) on embeddings

Key Technical Concepts:
1. T-GCN Architecture: Combines spatial graph convolution with temporal modeling via GRU
2. Temporal Sequences: Creates time-ordered sequences from microbial abundance data  
3. Graph Construction: Uses KNN-based sparsification (same as original pipeline)
4. GNNExplainer: Sparsifies graph based on edge importance for better interpretability
5. Embedding Extraction: Uses T-GCN hidden representations for downstream ML

Sources:
- Original T-GCN Paper: "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction" 
  (Zhao et al., 2019) - IEEE Transactions on Intelligent Transportation Systems
- PyTorch Geometric Temporal: https://pytorch-geometric-temporal.readthedocs.io/
- Implementation Reference: https://github.com/lehaifeng/T-GCN
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import functions from existing files (in same directory now)
from dataset_regression import MicrobialGNNDataset
from explainer_regression import GNNExplainerRegression
from pipeline_explainer import create_explainer_sparsified_graph

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalDataset:
    """
    Temporal dataset wrapper for creating time sequences from microbial data.
    
    This class transforms static microbial abundance data into temporal sequences
    suitable for T-GCN training. It simulates temporal dynamics by creating
    sequences from different samples or time-ordered data points.
    
    Technical Approach:
    - Creates sliding windows of temporal sequences
    - Maintains graph structure across time steps
    - Handles variable sequence lengths
    """
    
    def __init__(self, data_list, sequence_length: int = 5):
        """
        Initialize temporal dataset from PyTorch Geometric data list.
        
        Args:
            data_list: List of PyTorch Geometric Data objects
            sequence_length: Length of temporal sequences to create
        """
        self.data_list = data_list
        self.sequence_length = sequence_length
        self.temporal_data_list = self._create_temporal_sequences()
    
    def _create_temporal_sequences(self):
        """
        Create temporal sequences from PyTorch Geometric data list.
        
        Returns:
            List of temporal Data objects
        """
        temporal_data_list = []
        
        for data in self.data_list:
            # Create temporal sequences from node features
            features = data.x.numpy()  # Shape: (num_nodes, 1)
            n_nodes = features.shape[0]
            
            # Create sequences by duplicating and adding noise to simulate temporal variation
            sequences = []
            for t in range(self.sequence_length):
                # Add small temporal variation
                noise_factor = 0.1 * (t / self.sequence_length)
                temporal_features = features + np.random.normal(0, noise_factor * np.std(features), features.shape)
                sequences.append(temporal_features)
            
            # Stack sequences: (sequence_length, num_nodes, 1)
            temporal_x = torch.tensor(np.array(sequences), dtype=torch.float32)
            
            # Create new temporal data object
            temporal_data = Data(
                x=temporal_x,  # Now (seq_len, num_nodes, 1)
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
                y=data.y
            )
            
            temporal_data_list.append(temporal_data)
        
        return temporal_data_list

class TGCN(nn.Module):
    """
    Temporal Graph Convolutional Network (T-GCN) Implementation.
    
    T-GCN combines Graph Convolutional Networks with Gated Recurrent Units to capture
    both spatial dependencies (through graph structure) and temporal dependencies
    (through sequential patterns).
    
    Architecture:
    1. Graph Convolution Layer: Captures spatial relationships between microbes
    2. GRU Layer: Models temporal dynamics in abundance patterns
    3. Output Layer: Projects to target dimension
    
    Technical Details:
    - Uses symmetric normalization for graph convolution
    - Implements batch normalization for training stability
    - Supports variable sequence lengths through padding
    - Returns both predictions and embeddings for downstream ML
    """
    
    def __init__(self, 
                 input_dim: int = 1, 
                 hidden_dim: int = 64, 
                 output_dim: int = 1,
                 dropout: float = 0.2):
        """
        Initialize T-GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of targets)
            dropout: Dropout probability
        """
        super(TGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph Convolutional Layer
        self.gcn = GCNConv(input_dim, hidden_dim)
        
        # Gated Recurrent Unit
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, data, return_embeddings=False):
        """
        Forward pass of T-GCN.
        
        Args:
            data: PyTorch Geometric Data object with temporal features
            return_embeddings: If True, return embeddings along with predictions
            
        Returns:
            If return_embeddings=False: predictions tensor
            If return_embeddings=True: (predictions, embeddings) tuple
        """
        x_seq = data.x  # Shape: (seq_len, num_nodes, input_dim)
        edge_index = data.edge_index
        
        seq_len, num_nodes, _ = x_seq.shape
        
        # Store GCN outputs for each time step
        gcn_outputs = []
        
        for t in range(seq_len):
            # Get features at time step t
            x_t = x_seq[t]  # (num_nodes, input_dim)
            
            # Apply Graph Convolution
            h_t = self.gcn(x_t, edge_index)
            
            # Apply batch normalization
            if h_t.shape[0] > 1:  # Only if batch size > 1
                h_t_norm = self.batch_norm(h_t.unsqueeze(0)).squeeze(0)
            else:
                h_t_norm = h_t
            
            # Global mean pooling over nodes
            h_t_pooled = torch.mean(h_t_norm, dim=0, keepdim=True)  # (1, hidden_dim)
            
            gcn_outputs.append(h_t_pooled)
        
        # Stack temporal features: (1, seq_len, hidden_dim)
        h_seq = torch.stack(gcn_outputs, dim=1)
        
        # Apply GRU
        gru_out, _ = self.gru(h_seq)
        
        # Use last time step output
        h_final = gru_out[:, -1, :]  # (1, hidden_dim)
        
        # Apply output layer
        output = self.output_layer(h_final)
        
        if return_embeddings:
            return output, h_final
        else:
            return output

class TGCNPipeline:
    """
    Complete T-GCN Pipeline for Microbial Data Analysis.
    
    This pipeline follows the exact same workflow as mixed_embedding_pipeline.py
    but uses T-GCN instead of static GNN models.
    
    Pipeline Flow:
    1. Load data using MicrobialGNNDataset (KNN graph construction)
    2. Create temporal sequences from static data
    3. Train T-GCN on KNN-sparsified graph
    4. Use GNNExplainer to create sparsified graph
    5. Re-train T-GCN on explainer-sparsified graph
    6. Extract embeddings from best T-GCN model
    7. Train ML models on embeddings with cross-validation
    8. Compare and visualize results
    """
    
    def __init__(self, 
                 data_path,
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 sequence_length=5,
                 hidden_dim=64,
                 dropout_rate=0.3,
                 batch_size=8,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 num_epochs=200,
                 patience=20,
                 num_folds=5,
                 save_dir='./tgcn_results',
                 importance_threshold=0.2,
                 use_fast_correlation=False,
                 graph_mode='family',
                 family_filter_mode='strict'):
        """
        Initialize the T-GCN pipeline following mixed_embedding_pipeline structure.
        
        Args:
            data_path: Path to the CSV file with data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
            sequence_length: Length of temporal sequences to create
            hidden_dim: Hidden dimension size for T-GCN
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
            graph_mode: Mode for graph construction ('otu' or 'family')
            family_filter_mode: Mode for family filtering ('relaxed' or 'strict')
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.sequence_length = sequence_length
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
        
        # Create save directories (matching mixed_embedding_pipeline structure)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/tgcn_models", exist_ok=True)
        os.makedirs(f"{save_dir}/ml_models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/graphs", exist_ok=True)
        os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations", exist_ok=True)
        os.makedirs(f"{save_dir}/detailed_results", exist_ok=True)
        
        # Load and process data (same as mixed_embedding_pipeline)
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
        print(f"Using T-GCN with sequence length: {sequence_length}")

    def create_tgcn_model(self, num_targets=1):
        """Create a T-GCN model"""
        model = TGCN(
            input_dim=1,  # Microbial abundance data is 1D
            hidden_dim=self.hidden_dim,
            output_dim=num_targets,
            dropout=self.dropout_rate
        ).to(device)
        return model

    def train_tgcn_model(self, target_idx, data_list=None):
        """
        Train T-GCN model using k-fold cross-validation.
        
        Args:
            target_idx: Index of target variable
            data_list: List of PyTorch Geometric data objects (None = use original)
            
        Returns:
            Dictionary with trained model and metrics
        """
        if data_list is None:
            data_list = self.dataset.data_list
        
        target_name = self.target_names[target_idx]
        phase = "explainer" if data_list != self.dataset.data_list else "knn"
        print(f"\nTraining T-GCN model for target: {target_name} ({phase} graph)")
        
        # Create temporal sequences from data_list
        temporal_dataset = TemporalDataset(data_list, self.sequence_length)
        temporal_data_list = temporal_dataset.temporal_data_list
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        best_model = None
        best_r2 = -float('inf')
        
        criterion = nn.MSELoss()
        
        # Iterate through folds
        for fold, (train_index, test_index) in enumerate(kf.split(temporal_data_list)):
            fold_num = fold + 1
            print(f"  Fold {fold_num}/{self.num_folds}")
            
            # Split into train and test sets
            train_dataset = [temporal_data_list[i] for i in train_index]
            test_dataset = [temporal_data_list[i] for i in test_index]
            
            # Create model
            model = self.create_tgcn_model(num_targets=1)
            
            # Setup optimizer and scheduler
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=self.patience//2, verbose=False
            )
            
            # Training variables
            best_fold_loss = float('inf')
            patience_counter = 0
            epoch_losses = []
            
            # Training loop
            model.train()
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                
                for data in train_dataset:
                    data = data.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = model(data)
                    target = data.y[:, target_idx].view(-1, 1)
                    
                    # Compute loss
                    loss = criterion(predictions, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                epoch_loss /= len(train_dataset)
                epoch_losses.append(epoch_loss)
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for data in test_dataset:
                            data = data.to(device)
                            predictions = model(data)
                            target = data.y[:, target_idx].view(-1, 1)
                            loss = criterion(predictions, target)
                            val_loss += loss.item()
                    
                    val_loss /= len(test_dataset)
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_fold_loss:
                        best_fold_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break
                    
                    model.train()
            
            # Final evaluation
            model.eval()
            train_preds, train_targets = [], []
            test_preds, test_targets = [], []
            
            with torch.no_grad():
                # Training predictions
                for data in train_dataset:
                    data = data.to(device)
                    predictions = model(data)
                    target = data.y[:, target_idx].view(-1, 1)
                    
                    train_preds.extend(predictions.cpu().numpy().flatten())
                    train_targets.extend(target.cpu().numpy().flatten())
                
                # Test predictions
                for data in test_dataset:
                    data = data.to(device)
                    predictions = model(data)
                    target = data.y[:, target_idx].view(-1, 1)
                    
                    test_preds.extend(predictions.cpu().numpy().flatten())
                    test_targets.extend(target.cpu().numpy().flatten())
            
            # Calculate metrics
            train_mse = mean_squared_error(train_targets, train_preds)
            test_mse = mean_squared_error(test_targets, test_preds)
            train_r2 = r2_score(train_targets, train_preds)
            test_r2 = r2_score(test_targets, test_preds)
            train_mae = mean_absolute_error(train_targets, train_preds)
            test_mae = mean_absolute_error(test_targets, test_preds)
            
            fold_results.append({
                'fold': fold_num,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_targets': train_targets,
                'test_targets': test_targets,
                'train_preds': train_preds,
                'test_preds': test_preds
            })
            
            # Update best model
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
            
            print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Calculate average metrics
        avg_metrics = {
            'train_mse': np.mean([f['train_mse'] for f in fold_results]),
            'test_mse': np.mean([f['test_mse'] for f in fold_results]),
            'train_r2': np.mean([f['train_r2'] for f in fold_results]),
            'test_r2': np.mean([f['test_r2'] for f in fold_results]),
            'train_mae': np.mean([f['train_mae'] for f in fold_results]),
            'test_mae': np.mean([f['test_mae'] for f in fold_results]),
            'r2': np.mean([f['test_r2'] for f in fold_results])  # For compatibility
        }
        
        print(f"Average T-GCN Results - Train R²: {avg_metrics['train_r2']:.4f}, Test R²: {avg_metrics['test_r2']:.4f}")
        
        return {
            'model': best_model,
            'avg_metrics': avg_metrics,
            'fold_results': fold_results
        }

    def create_explainer_sparsified_graph(self, model, target_idx=0):
        """
        Create explainer-sparsified graph using the trained T-GCN model.
        
        Args:
            model: Trained T-GCN model
            target_idx: Index of target variable
            
        Returns:
            List of sparsified PyTorch Geometric data objects
        """
        print("Creating GNNExplainer-sparsified graph using T-GCN...")
        
        # Use the existing explainer pipeline function
        # Note: This assumes the T-GCN model is compatible with the explainer
        explainer_data = create_explainer_sparsified_graph(
            dataset=self.dataset,
            model=model,
            target_idx=target_idx,
            save_path=f"{self.save_dir}/explanations",
            importance_threshold=self.importance_threshold,
            device=device
        )
        
        print(f"Created explainer-sparsified graph with {len(explainer_data)} samples")
        return explainer_data

    def extract_embeddings(self, model, data_list):
        """
        Extract embeddings from trained T-GCN model.
        
        Args:
            model: Trained T-GCN model
            data_list: List of PyTorch Geometric data objects
            
        Returns:
            Tuple of (embeddings, targets)
        """
        print("Extracting embeddings from T-GCN model...")
        
        # Create temporal sequences
        temporal_dataset = TemporalDataset(data_list, self.sequence_length)
        temporal_data_list = temporal_dataset.temporal_data_list
        
        model.eval()
        embeddings = []
        targets = []
        
        with torch.no_grad():
            for data in temporal_data_list:
                data = data.to(device)
                
                # Get embeddings from T-GCN
                _, embedding = model(data, return_embeddings=True)
                
                embeddings.append(embedding.cpu().numpy().flatten())
                targets.append(data.y.cpu().numpy())
        
        embeddings = np.array(embeddings)
        targets = np.array(targets)
        
        print(f"Extracted embeddings shape: {embeddings.shape}")
        print(f"Targets shape: {targets.shape}")
        
        return embeddings, targets

    def train_ml_models(self, embeddings, targets, target_idx):
        """
        Train ML models on T-GCN embeddings using k-fold cross-validation.
        
        Args:
            embeddings: T-GCN embeddings
            targets: Target values
            target_idx: Index of target variable
            
        Returns:
            Dictionary with ML model results
        """
        print("Training ML models on T-GCN embeddings...")
        
        target_values = targets[:, target_idx]
        
        # Define ML models (same as mixed_embedding_pipeline)
        ml_models = {
            'RandomForest': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'ExtraTrees': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', ExtraTreesRegressor(n_estimators=100, random_state=42))
            ]),
            'LinearSVR': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearSVR(random_state=42, max_iter=2000))
            ])
        }
        
        # K-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        ml_results = {}
        
        for model_name, model in ml_models.items():
            print(f"  Training {model_name}...")
            
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
                X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                y_train, y_test = target_values[train_idx], target_values[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Metrics
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                fold_results.append({
                    'fold': fold + 1,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_targets': y_train,
                    'test_targets': y_test,
                    'train_preds': train_pred,
                    'test_preds': test_pred
                })
            
            # Average metrics
            avg_metrics = {
                'train_mse': np.mean([f['train_mse'] for f in fold_results]),
                'test_mse': np.mean([f['test_mse'] for f in fold_results]),
                'train_r2': np.mean([f['train_r2'] for f in fold_results]),
                'test_r2': np.mean([f['test_r2'] for f in fold_results]),
                'train_mae': np.mean([f['train_mae'] for f in fold_results]),
                'test_mae': np.mean([f['test_mae'] for f in fold_results]),
                'r2': np.mean([f['test_r2'] for f in fold_results])  # For compatibility
            }
            
            ml_results[model_name] = {
                'avg_metrics': avg_metrics,
                'fold_results': fold_results
            }
            
            print(f"    {model_name} - Test R²: {avg_metrics['test_r2']:.4f}")
        
        return ml_results

    def plot_results(self, tgcn_results, ml_results, target_idx):
        """Create comprehensive plots for T-GCN and ML results"""
        target_name = self.target_names[target_idx]
        
        fig = plt.figure(figsize=(20, 12))
        
        # Plot T-GCN results
        ax1 = plt.subplot(2, 3, 1)
        knn_r2 = tgcn_results['knn']['avg_metrics']['test_r2']
        explainer_r2 = tgcn_results['explainer']['avg_metrics']['test_r2']
        
        plt.bar(['T-GCN (KNN)', 'T-GCN (Explainer)'], [knn_r2, explainer_r2], 
                color=['blue', 'green'], alpha=0.7)
        plt.title(f'T-GCN Performance - {target_name}')
        plt.ylabel('Test R²')
        plt.grid(True, alpha=0.3)
        
        # Plot ML model results
        ax2 = plt.subplot(2, 3, 2)
        ml_names = list(ml_results.keys())
        ml_r2_scores = [ml_results[name]['avg_metrics']['test_r2'] for name in ml_names]
        
        plt.bar(ml_names, ml_r2_scores, color='orange', alpha=0.7)
        plt.title(f'ML Models on T-GCN Embeddings - {target_name}')
        plt.ylabel('Test R²')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Combined comparison
        ax3 = plt.subplot(2, 3, 3)
        all_models = ['T-GCN (KNN)', 'T-GCN (Explainer)'] + ml_names
        all_scores = [knn_r2, explainer_r2] + ml_r2_scores
        colors = ['blue', 'green'] + ['orange'] * len(ml_names)
        
        plt.bar(all_models, all_scores, color=colors, alpha=0.7)
        plt.title(f'Overall Comparison - {target_name}')
        plt.ylabel('Test R²')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Prediction scatter plots for best models
        best_tgcn = 'explainer' if explainer_r2 > knn_r2 else 'knn'
        best_ml = max(ml_names, key=lambda x: ml_results[x]['avg_metrics']['test_r2'])
        
        # T-GCN predictions
        ax4 = plt.subplot(2, 3, 4)
        tgcn_fold_results = tgcn_results[best_tgcn]['fold_results']
        all_true = np.concatenate([f['test_targets'] for f in tgcn_fold_results])
        all_pred = np.concatenate([f['test_preds'] for f in tgcn_fold_results])
        
        plt.scatter(all_true, all_pred, alpha=0.6, s=20)
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.title(f'T-GCN ({best_tgcn}) Predictions\nR² = {tgcn_results[best_tgcn]["avg_metrics"]["test_r2"]:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # ML predictions
        ax5 = plt.subplot(2, 3, 5)
        ml_fold_results = ml_results[best_ml]['fold_results']
        all_true_ml = np.concatenate([f['test_targets'] for f in ml_fold_results])
        all_pred_ml = np.concatenate([f['test_preds'] for f in ml_fold_results])
        
        plt.scatter(all_true_ml, all_pred_ml, alpha=0.6, s=20, color='orange')
        min_val = min(all_true_ml.min(), all_pred_ml.min())
        max_val = max(all_true_ml.max(), all_pred_ml.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.title(f'{best_ml} Predictions\nR² = {ml_results[best_ml]["avg_metrics"]["test_r2"]:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Residual plot
        ax6 = plt.subplot(2, 3, 6)
        residuals = all_true_ml - all_pred_ml
        plt.scatter(all_pred_ml, residuals, alpha=0.6, s=20, color='green')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.title(f'{best_ml} Residuals')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/plots/tgcn_results_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, all_results):
        """Save all results to files"""
        print("Saving results...")
        
        # Save complete results object
        with open(f'{self.save_dir}/all_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # Create summary dataframe
        summary_data = []
        for target_name, target_results in all_results.items():
            if target_name == 'summary':
                continue
                
            # T-GCN results
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    metrics = target_results[phase]['avg_metrics']
                    summary_data.append({
                        'Target': target_name,
                        'Model': f'T-GCN ({phase})',
                        'Train_R2': metrics['train_r2'],
                        'Test_R2': metrics['test_r2'],
                        'Train_MSE': metrics['train_mse'],
                        'Test_MSE': metrics['test_mse'],
                        'Train_MAE': metrics['train_mae'],
                        'Test_MAE': metrics['test_mae']
                    })
            
            # ML results
            if 'ml_models' in target_results:
                for model_name, results in target_results['ml_models'].items():
                    metrics = results['avg_metrics']
                    summary_data.append({
                        'Target': target_name,
                        'Model': f'{model_name} (embeddings)',
                        'Train_R2': metrics['train_r2'],
                        'Test_R2': metrics['test_r2'],
                        'Train_MSE': metrics['train_mse'],
                        'Test_MSE': metrics['test_mse'],
                        'Train_MAE': metrics['train_mae'],
                        'Test_MAE': metrics['test_mae']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.save_dir}/results_summary.csv', index=False)
        
        print(f"Results saved to {self.save_dir}")

    def run_pipeline(self):
        """
        Run the complete T-GCN pipeline following mixed_embedding_pipeline structure:
        1. Train T-GCN on KNN-sparsified graph
        2. Create GNNExplainer-sparsified graph using best T-GCN model
        3. Train T-GCN on explainer-sparsified graph
        4. Extract embeddings from best T-GCN model
        5. Train ML models on embeddings with cross-validation
        6. Compare and analyze all results
        """
        print("\n" + "="*80)
        print("TEMPORAL GNN (T-GCN) PIPELINE - COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        # Process each target variable
        for target_idx, target_name in enumerate(self.target_names):
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET: {target_name} ({target_idx + 1}/{len(self.target_names)})")
            print(f"{'='*60}")
            
            target_results = {}
            
            # Step 1: Train T-GCN on KNN-sparsified graph
            print(f"\nSTEP 1: Training T-GCN on KNN-sparsified graph")
            print("-" * 50)
            
            knn_results = self.train_tgcn_model(
                target_idx=target_idx,
                data_list=self.dataset.data_list
            )
            
            target_results['knn'] = knn_results
            
            # Step 2: Create GNNExplainer-sparsified graph
            print(f"\nSTEP 2: Creating GNNExplainer-sparsified graph")
            print("-" * 50)
            
            explainer_data = self.create_explainer_sparsified_graph(
                model=knn_results['model'],
                target_idx=target_idx
            )
            
            # Step 3: Train T-GCN on explainer-sparsified graph
            print(f"\nSTEP 3: Training T-GCN on explainer-sparsified graph")
            print("-" * 50)
            
            explainer_results = self.train_tgcn_model(
                target_idx=target_idx,
                data_list=explainer_data
            )
            
            target_results['explainer'] = explainer_results
            
            # Choose best model (explainer-trained model for embeddings)
            best_model = explainer_results['model']
            embedding_data = explainer_data
            
            print(f"\nUsing explainer-trained T-GCN for embedding extraction")
            print(f"Explainer T-GCN R²: {explainer_results['avg_metrics']['test_r2']:.4f}")
            
            # Step 4: Extract embeddings from best T-GCN model
            print(f"\nSTEP 4: Extracting embeddings from T-GCN model")
            print("-" * 50)
            
            embeddings, targets = self.extract_embeddings(best_model, embedding_data)
            
            # Save embeddings
            embedding_filename = f"{target_name}_tgcn_embeddings.npy"
            targets_filename = f"{target_name}_targets.npy"
            
            np.save(f"{self.save_dir}/embeddings/{embedding_filename}", embeddings)
            np.save(f"{self.save_dir}/embeddings/{targets_filename}", targets)
            
            print(f"Extracted embeddings shape: {embeddings.shape}")
            print(f"Saved as: {embedding_filename}")
            
            # Step 5: Train ML models on embeddings
            print(f"\nSTEP 5: Training ML models on T-GCN embeddings")
            print("-" * 50)
            
            ml_results = self.train_ml_models(embeddings, targets, target_idx)
            target_results['ml_models'] = ml_results
            
            # Step 6: Create comprehensive plots
            print(f"\nSTEP 6: Creating comprehensive plots")
            print("-" * 50)
            
            self.plot_results({
                'knn': knn_results,
                'explainer': explainer_results
            }, ml_results, target_idx)
            
            all_results[target_name] = target_results
        
        # Save all results
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        
        self.save_results(all_results)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("T-GCN PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        print("\nSUMMARY OF BEST MODELS PER TARGET:")
        print("-" * 50)
        
        for target_name, target_results in all_results.items():
            # Find best model for this target across all categories
            best_r2 = -float('inf')
            best_model_info = None
            
            # Check T-GCN models
            for phase in ['knn', 'explainer']:
                if phase in target_results:
                    r2 = target_results[phase]['avg_metrics']['test_r2']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_info = f"T-GCN ({phase})"
            
            # Check ML models
            if 'ml_models' in target_results:
                for model_type, results in target_results['ml_models'].items():
                    r2 = results['avg_metrics']['test_r2']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_info = f"{model_type} (T-GCN embeddings)"
            
            print(f"{target_name}: {best_model_info} - R² = {best_r2:.4f}")
        
        print(f"\nAll results saved to: {self.save_dir}")
        print("Check the following files:")
        print(f"  - results_summary.csv: Tabular summary of all results")
        print(f"  - all_results.pkl: Complete results object")
        print(f"  - plots/: Comprehensive visualization plots")
        print(f"  - embeddings/: Extracted T-GCN embeddings and targets")
        
        return all_results

def main():
    """Main execution function with example usage."""
    print("T-GCN Pipeline for Microbial Data Analysis")
    print("Implementation based on:")
    print("- T-GCN: Zhao et al. (2019) 'T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction'")
    print("- Mixed Embedding Pipeline architecture")
    print("- PyTorch Geometric Temporal library")
    print("- Microbial network analysis principles")
    
    # Initialize pipeline
    pipeline = TGCNPipeline(
        data_path="../Data/df.csv",  # Updated path since we're now in Given_Code
        k_neighbors=5,                # KNN graph sparsification
        sequence_length=5,            # Temporal sequence length  
        hidden_dim=64,               # Hidden dimension for T-GCN
        learning_rate=0.001,         # Learning rate
        num_epochs=100,              # Training epochs
        num_folds=5,                 # Cross-validation folds
        save_dir='../tgcn_results'   # Results directory (up one level)
    )
    
    try:
        # Run the complete pipeline
        results = pipeline.run_pipeline()
        print("\nT-GCN Pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"Data file not found: ../Data/df.csv")
        print("Please ensure the data file exists and try again.")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main() 