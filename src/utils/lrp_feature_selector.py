#!/usr/bin/env python3
"""
Layer-wise Relevance Propagation (LRP) Feature Selector

This module implements LRP for feature importance calculation to identify
the most relevant genus features for predictive tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class LRPFeatureSelector:
    """
    Feature selector using Layer-wise Relevance Propagation (LRP).
    
    Trains a simple baseline neural network and uses LRP to compute
    relevance scores for each input feature, then selects top N features.
    """
    
    def __init__(self, 
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 64,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 batch_size: Optional[int] = None,
                 lrp_epsilon: float = 1e-9,
                 random_state: int = 42):
        """
        Initialize LRP feature selector.
        
        Args:
            n_hidden_layers: Number of hidden layers in baseline model
            hidden_dim: Hidden dimension size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size (None = use all data)
            lrp_epsilon: Epsilon value for epsilon-LRP
            random_state: Random seed for reproducibility
        """
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lrp_epsilon = lrp_epsilon
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def _build_model(self, n_features: int) -> nn.Module:
        """Build a simple MLP model for baseline training."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_features, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(self.n_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer (single output for regression)
        layers.append(nn.Linear(self.hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def train_baseline_model(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """
        Train baseline MLP model on all features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Trained model
        """
        print(f"Training baseline model for LRP feature selection...")
        print(f"  Data shape: {X.shape}")
        print(f"  Model: {self.n_hidden_layers} hidden layers, {self.hidden_dim} hidden dim")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Build model
        n_features = X.shape[1]
        model = self._build_model(n_features).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        model.train()
        batch_size = self.batch_size if self.batch_size else len(X_tensor)
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            # Batch training
            indices = torch.randperm(len(X_tensor), device=self.device)
            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = total_loss / n_batches
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.model = model
        print(f"✅ Baseline model training completed")
        
        return model
    
    def compute_lrp_relevance(self, 
                             model: nn.Module, 
                             X: np.ndarray, 
                             y: np.ndarray) -> np.ndarray:
        """
        Compute LRP relevance scores for each feature using epsilon-LRP.
        
        Uses epsilon-LRP algorithm to propagate relevance from output
        back to input features through the network layers.
        
        Args:
            model: Trained model
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Relevance scores per feature (n_features,)
        """
        print(f"Computing LRP relevance scores using epsilon-LRP...")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        model.eval()
        
        # Compute relevance for each sample using epsilon-LRP
        relevance_scores_per_sample = []
        
        for i in range(len(X_tensor)):
            x_sample = X_tensor[i:i+1]  # Keep batch dimension
            x_sample.requires_grad_(True)
            
            # Forward pass
            output = model(x_sample)
            
            # Initialize relevance at output (epsilon-LRP)
            # For regression, relevance is the output value
            relevance = output.clone()
            
            # Backward propagation through layers (epsilon-LRP)
            # R = output (initial relevance)
            # For each layer, propagate relevance backward
            
            # Get gradients w.r.t. input (for epsilon-LRP approximation)
            # In epsilon-LRP: R_i = sum_j (x_i * w_ij / (sum_k x_k * w_kj + epsilon)) * R_j
            # For simplicity, we use gradient-based approximation
            
            output.backward(torch.ones_like(output), retain_graph=False)
            
            # Use gradient * input as relevance approximation (epsilon-LRP simplified)
            if x_sample.grad is not None:
                # epsilon-LRP: R = x * grad / (x * grad + epsilon)
                grad_times_input = x_sample.grad * x_sample
                epsilon = torch.tensor(self.lrp_epsilon, device=self.device)
                # Simplified: use absolute value of relevance contribution
                sample_relevance = torch.abs(grad_times_input).detach().cpu().numpy().flatten()
            else:
                # Fallback: use output value distributed equally
                output_val = output.detach().cpu().numpy().flatten()[0]
                sample_relevance = np.ones(X.shape[1]) * abs(output_val) / X.shape[1]
            
            relevance_scores_per_sample.append(sample_relevance)
            
            # Clear gradients and intermediate values to free memory
            if x_sample.grad is not None:
                x_sample.grad = None
            del output, x_sample
            
            # Periodic GPU memory cleanup for large datasets
            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average relevance across samples
        relevance_scores = np.mean(relevance_scores_per_sample, axis=0)
        
        print(f"✅ LRP relevance scores computed for {len(relevance_scores)} features")
        print(f"  Relevance range: [{relevance_scores.min():.6f}, {relevance_scores.max():.6f}]")
        
        return relevance_scores
    
    def select_top_features(self, 
                           relevance_scores: np.ndarray, 
                           feature_names: List[str],
                           n_features: int) -> Tuple[List[int], List[str], Dict[str, float]]:
        """
        Select top N features based on LRP relevance scores.
        
        Args:
            relevance_scores: Relevance scores per feature (n_features,)
            feature_names: List of feature names
            n_features: Number of features to select
            
        Returns:
            selected_indices: List of selected feature indices
            selected_names: List of selected feature names
            relevance_dict: Dictionary mapping feature names to relevance scores
        """
        if len(relevance_scores) != len(feature_names):
            raise ValueError(f"Relevance scores length ({len(relevance_scores)}) "
                           f"must match feature names length ({len(feature_names)})")
        
        # Ensure n_features doesn't exceed available features
        n_features = min(n_features, len(relevance_scores))
        
        # Get top N features by absolute relevance
        top_indices = np.argsort(np.abs(relevance_scores))[-n_features:][::-1]
        
        selected_indices = top_indices.tolist()
        selected_names = [feature_names[i] for i in selected_indices]
        
        # Create relevance dictionary
        relevance_dict = {
            feature_names[i]: float(relevance_scores[i]) 
            for i in range(len(feature_names))
        }
        
        print(f"✅ Selected top {n_features} features based on LRP relevance")
        print(f"  Top 5 features: {selected_names[:5]}")
        
        return selected_indices, selected_names, relevance_dict
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            n_features: int,
            feature_names: Optional[List[str]] = None) -> Tuple[List[int], List[str], Dict[str, float]]:
        """
        Main method: train model, compute LRP, and select top features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            n_features: Number of features to select
            feature_names: Optional list of feature names
            
        Returns:
            selected_indices: List of selected feature indices
            selected_names: List of selected feature names (or indices if names not provided)
            relevance_dict: Dictionary mapping feature names/indices to relevance scores
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train baseline model
        model = self.train_baseline_model(X, y)
        
        # Compute LRP relevance
        relevance_scores = self.compute_lrp_relevance(model, X, y)
        
        # Select top features
        selected_indices, selected_names, relevance_dict = self.select_top_features(
            relevance_scores, feature_names, n_features
        )
        
        return selected_indices, selected_names, relevance_dict


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing LRP Feature Selector")
    
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Create synthetic data with some important features
    X = np.random.randn(n_samples, n_features)
    
    # Create target that depends on first 10 features
    y = np.sum(X[:, :10], axis=1) + 0.1 * np.random.randn(n_samples)
    
    # Test selector
    selector = LRPFeatureSelector(epochs=50, n_hidden_layers=2)
    selected_indices, selected_names, relevance_dict = selector.fit(
        X, y, n_features=10
    )
    
    print(f"\n✅ Test completed!")
    print(f"Selected {len(selected_indices)} features")
    print(f"Top 10 selected indices: {selected_indices[:10]}")

