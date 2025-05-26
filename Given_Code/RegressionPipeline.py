import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GNNExplainer
from torch_geometric.explain import Explainer
from torch_geometric.explain.metric import fidelity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

# Import the models from GNNmodelsRegression
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

class MicrobialGNNDataset:
    """Dataset class for GNN-based regression on microbial data"""
    
    def __init__(self, data_path, k_neighbors=5, mantel_threshold=0.05):
        """
        Initialize the dataset class
        
        Args:
            data_path: Path to the CSV file with microbial data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        
        # Load and process the data
        self.df = pd.read_csv(data_path)
        
        # Filter out rows containing 'x' values if they exist
        if self.df.isin(['x']).any().any():
            self.df = self.df[~self.df.isin(['x']).any(axis=1)]
        
        # Identify feature and target columns
        # Assuming feature columns and target columns can be distinguished
        # Adapt this section based on your specific data format
        features_pattern = None
        targets_pattern = None
        
        if 'd__' in ''.join(self.df.columns):
            # If data has taxonomy-like columns (as in microbial data)
            features_pattern = 'd__'
            self.feature_cols = [c for c in self.df.columns if features_pattern in c]
            self.target_cols = [c for c in self.df.columns if c not in self.feature_cols]
        else:
            # If no clear pattern, assume last few columns are targets
            # Adjust this logic based on your data
            num_targets = 2  # Adjust based on your data
            self.feature_cols = list(self.df.columns[:-num_targets])
            self.target_cols = list(self.df.columns[-num_targets:])
        
        print(f"Identified {len(self.feature_cols)} feature columns and {len(self.target_cols)} target columns")
        
        # Extract target data
        self.target_df = self.df[self.target_cols].copy()
        self.target_df.columns = self.target_df.columns.str.strip()
        
        # Create node features
        self.df_features, self.feature_matrix = self._create_node_features()
        
        # Create initial graph structure using Mantel test
        self.full_edge_index, self.full_edge_weight, self.full_edge_type = self._create_graph_structure()
        
        # Create KNN sparsified graph structure (always use KNN for initial graph)
        self.edge_index, self.edge_weight, self.edge_type = self._create_knn_graph(k=k_neighbors)
        
        # Create PyG data objects
        self.data_list = self._create_data_objects()
        
        # Store original graph data for visualization
        self.original_graph_data = {
            'edge_index': self.edge_index.clone(),
            'edge_weight': self.edge_weight.clone(),
            'edge_type': self.edge_type.clone()
        }
        
        # Initialize explainer-sparsified graph data as None
        self.explainer_sparsified_graph_data = None
        
        # Create directory for visualizations
        os.makedirs('graph_visualizations', exist_ok=True)
    
    def _create_node_features(self):
        """Create node features from input data"""
        # Extract feature data
        df_features = self.df[self.feature_cols].copy()
        
        # Apply variance stabilization if needed
        # For microbial data, double square root transformation is common
        # Adjust this based on your data characteristics
        if df_features.min().min() >= 0:  # Check if all values are non-negative
            df_features = df_features.apply(lambda x: np.sqrt(np.sqrt(x + 1e-10)))
        
        # Convert to numpy array with shape [num_features, num_samples]
        feature_matrix = df_features.values.T.astype(np.float32)
        
        print(f"Node feature matrix: {feature_matrix.shape} (features × samples)")
        
        # Store the feature names for later use
        self.node_feature_names = list(df_features.columns)
        
        return df_features, feature_matrix
    
    def _compute_distance_matrix(self, vec, metric='euclidean'):
        """Compute distance matrix between samples for a given feature"""
        if vec.ndim == 1:
            vec = vec.reshape(-1, 1)
        dm = squareform(pdist(vec, metric=metric))
        return np.nan_to_num(dm, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _mantel_test(self, d1, d2, permutations=999):
        """Mantel test to assess correlation between two distance matrices"""
        n = d1.shape[0]
        idx = np.triu_indices(n, k=1)
        v1, v2 = d1[idx], d2[idx]
        
        if v1.std() == 0 or v2.std() == 0:
            return 1.0, 0.0
            
        r_obs = pearsonr(v1, v2)[0]
        
        # Permutation test
        count = 0
        for _ in range(permutations):
            perm_v2 = np.random.permutation(v2)
            r_perm = abs(pearsonr(v1, perm_v2)[0])
            if r_perm >= abs(r_obs):
                count += 1
                
        p_value = (count + 1) / (permutations + 1)
        return p_value, r_obs
    
    def _create_graph_structure(self):
        """Create graph structure using either Mantel test or correlation-based approach"""
        print("Constructing graph...")
        
        # Compute distance matrices for each feature
        dist_mats = {
            feat: self._compute_distance_matrix(self.feature_matrix[i], metric='euclidean')
            for i, feat in enumerate(self.node_feature_names)
        }
        
        # Construct edges based on Mantel test
        num_features = len(self.node_feature_names)
        edge_i, edge_j, edge_weights = [], [], []
        
        # Create edges between features with significant correlation
        for i in range(num_features):
            for j in range(i+1, num_features):
                # Use Mantel test to assess correlation
                p, r = self._mantel_test(
                    dist_mats[self.node_feature_names[i]],
                    dist_mats[self.node_feature_names[j]],
                    permutations=999
                )
                
                if p < self.mantel_threshold:  # Significant correlation
                    edge_i += [i, j]
                    edge_j += [j, i]
                    edge_weights += [abs(r), abs(r)]  # Use correlation strength as edge weight
        
        # Create edge types based on correlation sign
        edge_types = []
        for i, j in zip(edge_i, edge_j):
            # Calculate correlation coefficient between the two features
            corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
            # Edge type: 0 = negative correlation, 1 = positive correlation
            edge_type = 1 if corr > 0 else 0
            edge_types.append(edge_type)
        
        print(f"Created graph with {len(edge_i)//2} undirected edges")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_i, edge_j], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_weight, edge_type
    
    def _create_data_objects(self):
        """Create PyG Data objects for each sample"""
        # Transpose feature matrix to shape [num_samples, num_features]
        feature_matrix_samples = self.feature_matrix.T
        
        # Create a list of PyG Data objects
        data_list = []
        
        for s in range(feature_matrix_samples.shape[0]):
            # Node features for this sample - feature values
            x = torch.tensor(feature_matrix_samples[s], dtype=torch.float32).view(-1, 1)
            
            # Graph-level targets
            targets = torch.tensor(self.target_df.iloc[s].values, dtype=torch.float32).view(1, -1)
            
            # Create the Data object
            data = Data(
                x=x,
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                edge_attr=self.edge_weight.view(-1, 1),
                edge_type=self.edge_type,
                y=targets
            )
            
            data_list.append(data)
        
        print(f"Created {len(data_list)} graph data objects with {len(self.node_feature_names)} nodes each")
        
        return data_list
    
    def _create_knn_graph(self, k=None):
        """Create a k-nearest neighbor sparsified version of the graph"""
        if k is None:
            k = self.k_neighbors
            
        print(f"Creating KNN graph with k={k}...")
        
        # Create adjacency matrix from full edge_index
        num_nodes = len(self.node_feature_names)
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i in range(self.full_edge_index.shape[1]):
            u, v = self.full_edge_index[0, i], self.full_edge_index[1, i]
            adj_matrix[u, v] = self.full_edge_weight[i]
        
        # KNN sparsification
        adj_matrix_np = adj_matrix.numpy()
        
        # For each node, keep only the k strongest connections
        for i in range(num_nodes):
            # Get weights of all neighbors
            neighbors = adj_matrix_np[i]
            
            # Sort neighbors by weight and keep only top k
            if np.sum(neighbors > 0) > k:
                threshold = np.sort(neighbors)[-k]
                adj_matrix_np[i, neighbors < threshold] = 0
        
        # Make matrix symmetric (undirected graph)
        adj_matrix_np = np.maximum(adj_matrix_np, adj_matrix_np.T)
        
        # Convert back to edge_index and edge_weight format
        new_edge_index = []
        new_edge_weight = []
        new_edge_type = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix_np[i, j] > 0:
                    new_edge_index.append([i, j])
                    new_edge_weight.append(adj_matrix_np[i, j])
                    
                    # Determine edge type (sign of correlation)
                    corr, _ = pearsonr(self.feature_matrix[i], self.feature_matrix[j])
                    new_edge_type.append(1 if corr > 0 else 0)
        
        new_edge_index = torch.tensor(new_edge_index).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
        
        print(f"KNN graph has {new_edge_index.shape[1]//2} undirected edges")
        
        return new_edge_index, new_edge_weight, new_edge_type

class GNNExplainerRegression:
    """GNN explainer for regression tasks"""
    
    def __init__(self, model, device):
        """
        Initialize the explainer
        
        Args:
            model: The trained GNN model
            device: Device to run the explanation on
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode
    
    def explain_graph(self, data, node_names=None, save_path=None, target_idx=0):
        """
        Generate explanation for a graph by learning an edge mask
        
        Args:
            data: PyG Data object for a graph
            node_names: Names of the nodes (features)
            save_path: Path to save the edge importance matrix
            target_idx: Index of the target variable to explain
            
        Returns:
            edge_importance_matrix: Matrix of edge importance scores
            explanation: Text explanation of the most important edges
        """
        # Copy data to device
        data = data.to(self.device)
        
        # Initialize edge mask
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.float, 
                              requires_grad=True, device=self.device)
        
        # Setup optimizer for the edge mask
        optimizer = Adam([edge_mask], lr=0.01)
        
        # Number of epochs for explanation
        num_epochs = 100
        
        # Store original edge index and edge weight
        original_edge_index = data.edge_index
        original_edge_attr = data.edge_attr
        
        # Extract target
        target = data.y[:, target_idx]
        
        # Training loop to learn edge mask
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Apply edge mask - create masked edge attributes
            masked_edge_attr = data.edge_attr * edge_mask.view(-1, 1)
            
            # Forward pass with masked edges
            out = self.model(data.x, data.edge_index, data.batch, edge_attr=masked_edge_attr)
            
            # Handle different model output formats
            if isinstance(out, tuple):
                pred = out[0]
                
                # Reshape prediction if needed
                if len(pred.shape) == 1:
                    pred = pred.view(-1, 1)
                
                # Get the specific target prediction
                if pred.shape[1] > 1:
                    pred = pred[:, target_idx]
            else:
                pred = out
                if len(pred.shape) == 1:
                    pred = pred.view(-1, 1)
                
                if pred.shape[1] > 1:
                    pred = pred[:, target_idx]
            
            # Calculate loss: model prediction should match original
            loss = F.mse_loss(pred.squeeze(), target)
            
            # Add regularization to encourage sparsity in the mask
            loss = loss + 0.01 * edge_mask.sum()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Clamp edge mask values
            with torch.no_grad():
                edge_mask.clamp_(0, 1)
        
        # Create edge importance matrix
        num_nodes = data.x.shape[0]
        edge_importance_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        # Fill importance matrix with learned edge mask values
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i], data.edge_index[1, i]
            edge_importance_matrix[u, v] = edge_mask[i]
        
        # Make importance matrix symmetric (for undirected graphs)
        edge_importance_matrix = torch.maximum(edge_importance_matrix, edge_importance_matrix.t())
        
        # Generate text explanation of most important edges
        explanation = self._generate_explanation(edge_importance_matrix, node_names)
        
        # Save edge importance matrix if path is provided
        if save_path is not None:
            if node_names is not None:
                # Create DataFrame with node names
                importance_df = pd.DataFrame(
                    edge_importance_matrix.cpu().detach().numpy(),
                    index=node_names,
                    columns=node_names
                )
                importance_df.to_csv(save_path)
            else:
                # Save as numpy array
                np.savetxt(save_path, edge_importance_matrix.cpu().detach().numpy(), delimiter=',')
        
        return edge_importance_matrix, explanation
    
    def _generate_explanation(self, edge_importance_matrix, node_names=None):
        """Generate text explanation from edge importance matrix"""
        # Convert to numpy
        importance = edge_importance_matrix.cpu().detach().numpy()
        
        # Get top edges
        n = importance.shape[0]
        top_edges = []
        
        for i in range(n):
            for j in range(i+1, n):  # Only consider upper triangle for undirected graph
                if importance[i, j] > 0.2:  # Threshold for importance
                    top_edges.append((i, j, importance[i, j]))
        
        # Sort by importance
        top_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Generate explanation text
        explanation = "Top important feature interactions:\n"
        
        for i, (u, v, imp) in enumerate(top_edges[:10]):  # Show top 10 edges
            if node_names is not None:
                u_name = node_names[u]
                v_name = node_names[v]
                explanation += f"{i+1}. {u_name} ↔ {v_name}: {imp:.3f}\n"
            else:
                explanation += f"{i+1}. Feature {u} ↔ Feature {v}: {imp:.3f}\n"
        
        return explanation

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
                 estimate_uncertainty=False):
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
            mantel_threshold=mantel_threshold
        )
        
        # Get target names for reference
        self.target_names = self.dataset.target_cols 

    def _create_model(self, num_targets=1):
        """Create a GNN model based on specified type"""
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
    
    def train_model(self, target_idx=None, data_list=None):
        """
        Train a GNN model for regression
        
        Args:
            target_idx: Index of the target variable to predict (if None, predict all targets)
            data_list: List of graph data objects to use (if None, use self.dataset.data_list)
            
        Returns:
            Dictionary with training results
        """
        if data_list is None:
            data_list = self.dataset.data_list
        
        # Determine how many targets to predict
        if target_idx is not None:
            target_name = self.target_names[target_idx]
            num_targets = 1
            print(f"\n{'='*50}")
            print(f"Training model for target: {target_name}")
            print(f"{'='*50}")
        else:
            num_targets = len(self.target_names)
            target_idx = list(range(num_targets))
            print(f"\n{'='*50}")
            print(f"Training model for all {num_targets} targets")
            print(f"{'='*50}")
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        
        # Define loss function based on uncertainty estimation
        if self.estimate_uncertainty:
            criterion = GaussianNLLLoss()
        else:
            criterion = nn.MSELoss()
        
        # Iterate through folds
        for fold, (train_index, test_index) in enumerate(kf.split(data_list)):
            fold_num = fold + 1
            print(f"Fold {fold_num}/{self.num_folds}: Train on {len(train_index)} samples, Test on {len(test_index)} samples")
            
            # Split into train and test sets
            train_dataset = [data_list[i] for i in train_index]
            test_dataset = [data_list[i] for i in test_index]
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            model = self._create_model(num_targets=num_targets)
            
            # Setup optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
            
            # Training loop
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(1, self.num_epochs+1):
                # Training step
                model.train()
                total_loss = 0
                
                for batch_data in train_loader:
                    batch_data = batch_data.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    
                    # Handle different output formats based on uncertainty estimation
                    if self.estimate_uncertainty and isinstance(out, tuple) and len(out) >= 2:
                        pred, uncertainty = out[0], out[1]
                        target = batch_data.y.view(-1, num_targets)
                        
                        # Use GaussianNLLLoss for uncertainty-aware training
                        loss = criterion(pred, target, uncertainty)
                    else:
                        # Standard MSE loss
                        if isinstance(out, tuple):
                            pred = out[0]
                        else:
                            pred = out
                            
                        target = batch_data.y.view(-1, num_targets)
                        loss = criterion(pred, target)
                    
                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()
                    
                    total_loss += loss.item() * batch_data.num_graphs
                
                avg_train_loss = total_loss / len(train_dataset)
                train_losses.append(avg_train_loss)
                
                # Evaluation step
                model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        
                        # Forward pass
                        out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        
                        # Calculate validation loss
                        if self.estimate_uncertainty and isinstance(out, tuple) and len(out) >= 2:
                            pred, uncertainty = out[0], out[1]
                            target = batch_data.y.view(-1, num_targets)
                            val_loss = criterion(pred, target, uncertainty)
                        else:
                            if isinstance(out, tuple):
                                pred = out[0]
                            else:
                                pred = out
                                
                            target = batch_data.y.view(-1, num_targets)
                            val_loss = nn.MSELoss()(pred, target)
                        
                        total_val_loss += val_loss.item() * batch_data.num_graphs
                
                avg_val_loss = total_val_loss / len(test_dataset)
                val_losses.append(avg_val_loss)
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Print progress
                if epoch % 10 == 0 or epoch == 1 or epoch == self.num_epochs:
                    print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Load best model for evaluation
            model.load_state_dict(best_model_state)
            
            # Final evaluation
            model.eval()
            all_preds = []
            all_targets = []
            all_uncertainties = [] if self.estimate_uncertainty else None
            
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    
                    # Forward pass
                    out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    
                    # Extract predictions and targets
                    if self.estimate_uncertainty and isinstance(out, tuple) and len(out) >= 2:
                        pred, uncertainty = out[0], out[1]
                        all_uncertainties.append(uncertainty.cpu().numpy())
                    else:
                        if isinstance(out, tuple):
                            pred = out[0]
                        else:
                            pred = out
                    
                    target = batch_data.y.view(-1, num_targets)
                    
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
            
            # Concatenate results
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            if self.estimate_uncertainty:
                all_uncertainties = np.vstack(all_uncertainties)
            
            # Calculate metrics for each target
            target_metrics = []
            
            for i in range(num_targets):
                # Extract predictions and targets for this specific target
                target_preds = all_preds[:, i] if all_preds.shape[1] > 1 else all_preds.flatten()
                target_true = all_targets[:, i] if all_targets.shape[1] > 1 else all_targets.flatten()
                
                # Calculate metrics
                mse = mean_squared_error(target_true, target_preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(target_true, target_preds)
                
                # Get target name
                if target_idx is not None and not isinstance(target_idx, list):
                    t_name = self.target_names[target_idx]
                else:
                    t_name = self.target_names[i]
                
                # Store metrics
                target_metrics.append({
                    'target_idx': i if isinstance(target_idx, list) else target_idx,
                    'target_name': t_name,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                })
                
                print(f"Target {t_name}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
            
            # Save model
            if target_idx is not None and not isinstance(target_idx, list):
                model_path = f"{self.save_dir}/models/{self.model_type}_{self.target_names[target_idx]}_fold{fold_num}.pt"
            else:
                model_path = f"{self.save_dir}/models/{self.model_type}_all_targets_fold{fold_num}.pt"
            
            torch.save(model.state_dict(), model_path)
            
            # Create plot for training curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Fold {fold_num} Training Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if target_idx is not None and not isinstance(target_idx, list):
                plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{self.target_names[target_idx]}_fold{fold_num}_loss.png")
            else:
                plt.savefig(f"{self.save_dir}/plots/{self.model_type}_all_targets_fold{fold_num}_loss.png")
            
            plt.close()
            
            # Store fold results
            fold_results.append({
                'fold': fold_num,
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'predictions': all_preds,
                'targets': all_targets,
                'uncertainties': all_uncertainties,
                'metrics': target_metrics,
                'test_indices': test_index
            })
            
            # Create GNN explanations
            if target_idx is not None and not isinstance(target_idx, list):
                self._generate_explanations(model, test_dataset, self.target_names[target_idx], fold_num)
        
        # Plot overall results
        self._plot_overall_results(fold_results, target_idx)
        
        return fold_results
    
    def _generate_explanations(self, model, test_dataset, target_name, fold_num):
        """Generate and save GNN explanations"""
        print(f"\nGenerating explanations for {target_name} (fold {fold_num})...")
        
        # Initialize explainer
        explainer = GNNExplainerRegression(model, device)
        
        # Create output directory
        os.makedirs(f"{self.save_dir}/explanations/{target_name}", exist_ok=True)
        
        # Generate explanations for a subset of test samples
        num_explain = min(5, len(test_dataset))  # Limit to 5 samples
        
        for i in range(num_explain):
            # Get sample data
            data = test_dataset[i]
            
            # Generate explanation
            edge_importance_matrix, explanation = explainer.explain_graph(
                data, 
                node_names=self.dataset.node_feature_names,
                save_path=f"{self.save_dir}/explanations/{target_name}/sample_{i}_fold_{fold_num}_edge_importance.csv",
                target_idx=0  # For single target models
            )
            
            # Visualize the explanation
            self._visualize_explanation(
                data, 
                edge_importance_matrix, 
                i, 
                f"{self.save_dir}/explanations/{target_name}/sample_{i}_fold_{fold_num}_explanation.png",
                target_name
            )
            
            # Save the text explanation
            with open(f"{self.save_dir}/explanations/{target_name}/sample_{i}_fold_{fold_num}_explanation.txt", 'w') as f:
                f.write(explanation)
    
    def _visualize_explanation(self, data, edge_importance_matrix, sample_id, save_path, target_name):
        """Visualize explanation as a network with edge importance"""
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(self.dataset.node_feature_names):
            G.add_node(i, name=name, value=data.x[i].item())
        
        # Add edges with importance as weights
        edge_importance = edge_importance_matrix.cpu().numpy()
        
        # Normalize edge importance to [0, 1] range
        if edge_importance.max() > 0:
            edge_importance = edge_importance / edge_importance.max()
        
        # Add edges with importance above a threshold
        threshold = 0.2  # Only keep edges with at least 20% of max importance
        for i in range(len(self.dataset.node_feature_names)):
            for j in range(i+1, len(self.dataset.node_feature_names)):
                importance = edge_importance[i, j]
                if importance > threshold:
                    G.add_edge(i, j, weight=float(importance))
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Create layout
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Get node sizes based on values
        node_size = [1000 * (0.1 + abs(data.x[i].item())) for i in range(len(self.dataset.node_feature_names))]
        
        # Get edge widths based on importance
        edge_width = []
        for u, v, d in G.edges(data=True):
            edge_width.append(d['weight'] * 5)
        
        # Draw the network
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            labels={i: self.dataset.node_feature_names[i] for i in G.nodes()},
            node_size=node_size,
            width=edge_width,
            edge_color='gray',
            font_size=8,
            font_weight='bold',
            alpha=0.8
        )
        
        plt.title(f'{target_name} Explanation - Sample {sample_id}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_results(self, fold_results, target_idx=None):
        """Plot overall results across all folds"""
        # Determine how many targets we're working with
        if target_idx is not None and not isinstance(target_idx, list):
            num_targets = 1
            target_names = [self.target_names[target_idx]]
        else:
            num_targets = len(self.target_names)
            target_names = self.target_names
        
        # Create combined plot for all targets
        plt.figure(figsize=(15, 10))
        
        # Create subplots based on number of targets
        for i, target_name in enumerate(target_names):
            plt.subplot(1, num_targets, i+1)
            
            # Gather predictions and true values for this target
            all_preds = []
            all_targets = []
            
            for fold_result in fold_results:
                # Extract predictions and targets for this specific target
                target_preds = fold_result['predictions'][:, i] if fold_result['predictions'].shape[1] > 1 else fold_result['predictions'].flatten()
                target_true = fold_result['targets'][:, i] if fold_result['targets'].shape[1] > 1 else fold_result['targets'].flatten()
                
                all_preds.extend(target_preds)
                all_targets.extend(target_true)
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Calculate metrics
            mse = mean_squared_error(all_targets, all_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(all_targets, all_preds)
            
            # Create scatter plot
            plt.scatter(all_targets, all_preds, alpha=0.6, edgecolor='k', facecolor='none')
            
            # Add diagonal line
            min_val = min(np.min(all_targets), np.min(all_preds))
            max_val = max(np.max(all_targets), np.max(all_preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'{target_name}\nMSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if target_idx is not None and not isinstance(target_idx, list):
            plt.savefig(f"{self.save_dir}/plots/{self.model_type}_{self.target_names[target_idx]}_overall.png", dpi=300)
        else:
            plt.savefig(f"{self.save_dir}/plots/{self.model_type}_all_targets_overall.png", dpi=300)
        
        plt.close()
        
        # Save metrics to CSV
        metrics_df = []
        
        for i, target_name in enumerate(target_names):
            target_metrics = []
            
            for fold_result in fold_results:
                for metric in fold_result['metrics']:
                    if metric['target_name'] == target_name:
                        target_metrics.append(metric)
            
            # Calculate average metrics across folds
            avg_mse = np.mean([m['mse'] for m in target_metrics])
            avg_rmse = np.mean([m['rmse'] for m in target_metrics])
            avg_r2 = np.mean([m['r2'] for m in target_metrics])
            
            metrics_df.append({
                'target': target_name,
                'mse': avg_mse,
                'rmse': avg_rmse,
                'r2': avg_r2
            })
        
        # Convert to DataFrame and save
        metrics_df = pd.DataFrame(metrics_df)
        
        if target_idx is not None and not isinstance(target_idx, list):
            metrics_df.to_csv(f"{self.save_dir}/plots/{self.model_type}_{self.target_names[target_idx]}_metrics.csv", index=False)
        else:
            metrics_df.to_csv(f"{self.save_dir}/plots/{self.model_type}_all_targets_metrics.csv", index=False)
            
        # Print overall results
        print("\nOverall Results:")
        print(metrics_df)

    def create_explainer_sparsified_graph(self, model, target_idx=0, importance_threshold=0.3):
        """
        Create a sparsified graph based on GNNExplainer results
        
        Args:
            model: Trained GNN model
            target_idx: Index of the target variable to explain
            importance_threshold: Threshold for edge importance
            
        Returns:
            List of sparsified graph data objects
        """
        print(f"\nCreating explainer-based sparsified graph...")
        
        # Initialize explainer
        explainer = GNNExplainerRegression(model, device)
        
        # Create a combined edge importance matrix from multiple samples
        num_explain = min(10, len(self.dataset.data_list))  # Use up to 10 samples
        combined_edge_importance = torch.zeros((len(self.dataset.node_feature_names), len(self.dataset.node_feature_names)), device=device)
        
        for i in range(num_explain):
            # Get sample data
            data = self.dataset.data_list[i]
            
            # Generate explanation
            edge_importance_matrix, _ = explainer.explain_graph(
                data, 
                node_names=self.dataset.node_feature_names,
                target_idx=target_idx
            )
            
            # Add to combined matrix
            combined_edge_importance += edge_importance_matrix
        
        # Average the importance
        combined_edge_importance /= num_explain
        
        # Create sparsified adjacency matrix by thresholding
        adj_matrix = combined_edge_importance.clone()
        adj_matrix[adj_matrix < importance_threshold] = 0
        
        # Convert adjacency matrix to edge index and edge weight format
        num_nodes = len(self.dataset.node_feature_names)
        new_edge_index = []
        new_edge_weight = []
        new_edge_type = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    new_edge_index.append([i, j])
                    new_edge_weight.append(adj_matrix[i, j].item())
                    
                    # Determine edge type (sign of correlation)
                    corr, _ = pearsonr(self.dataset.feature_matrix[i], self.dataset.feature_matrix[j])
                    new_edge_type.append(1 if corr > 0 else 0)
        
        new_edge_index = torch.tensor(new_edge_index).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weight, dtype=torch.float32)
        new_edge_type = torch.tensor(new_edge_type, dtype=torch.long)
        
        print(f"Explainer sparsified graph has {new_edge_index.shape[1]//2} undirected edges")
        
        # Store sparsified graph data for visualization
        self.dataset.explainer_sparsified_graph_data = {
            'edge_index': new_edge_index.clone(),
            'edge_weight': new_edge_weight.clone(),
            'edge_type': new_edge_type.clone()
        }
        
        # Create new data objects with sparsified graph
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
        
        # Visualize both original and sparsified graphs
        self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
        return new_data_list 

    def run_pipeline(self):
        """
        Run the complete regression pipeline:
        1. Train models on KNN-sparsified graph (created during initialization)
        2. Create further sparsified graph using GNNExplainer
        3. Train models on GNNExplainer-sparsified graph
        4. Compare and analyze results
        
        Returns:
            Dictionary with all results
        """
        results = {}
        
        # Step 1: Train models on KNN-sparsified graph (already created during initialization)
        print("\n" + "="*80)
        print("STEP 1: Training on KNN-sparsified graph")
        print("="*80)
        
        knn_results = {}
        
        if len(self.target_names) > 1:
            # Either train a single multi-target model or separate models for each target
            train_separate = True  # Set to False if you want to train a multi-target model
            
            if train_separate:
                for target_idx, target_name in enumerate(self.target_names):
                    print(f"\nTraining for target: {target_name}")
                    knn_results[target_name] = self.train_model(target_idx=target_idx)
            else:
                print("\nTraining multi-target model")
                knn_results['all_targets'] = self.train_model()
        else:
            # Only one target, train a single model
            target_name = self.target_names[0]
            print(f"\nTraining for target: {target_name}")
            knn_results[target_name] = self.train_model(target_idx=0)
        
        # Step 2: Create GNNExplainer-sparsified graph
        print("\n" + "="*80)
        print("STEP 2: Creating GNNExplainer-sparsified graph")
        print("="*80)
        
        # Choose a model from the first target's results to use for explanation
        if len(self.target_names) > 1:
            target_name = self.target_names[0]
            model = knn_results[target_name][0]['model']  # Use first fold model
            target_idx = 0
        else:
            target_name = self.target_names[0]
            model = knn_results[target_name][0]['model']
            target_idx = 0
        
        # Create sparsified graph data using GNNExplainer
        explainer_sparsified_data = self.create_explainer_sparsified_graph(
            model, 
            target_idx=target_idx, 
            importance_threshold=self.importance_threshold
        )
        
        # Visualize the graphs
        self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
        
        # Step 3: Train models on GNNExplainer-sparsified graph
        print("\n" + "="*80)
        print("STEP 3: Training on GNNExplainer-sparsified graph")
        print("="*80)
        
        explainer_results = {}
        
        if len(self.target_names) > 1:
            # Either train a single multi-target model or separate models for each target
            train_separate = True  # Set to False if you want to train a multi-target model
            
            if train_separate:
                for target_idx, target_name in enumerate(self.target_names):
                    print(f"\nTraining for target: {target_name} (GNNExplainer-sparsified graph)")
                    explainer_results[target_name] = self.train_model(target_idx=target_idx, data_list=explainer_sparsified_data)
            else:
                print("\nTraining multi-target model (GNNExplainer-sparsified graph)")
                explainer_results['all_targets'] = self.train_model(data_list=explainer_sparsified_data)
        else:
            # Only one target, train a single model
            target_name = self.target_names[0]
            print(f"\nTraining for target: {target_name} (GNNExplainer-sparsified graph)")
            explainer_results[target_name] = self.train_model(target_idx=0, data_list=explainer_sparsified_data)
        
        # Step 4: Compare results
        print("\n" + "="*80)
        print("STEP 4: Comparing results between KNN and GNNExplainer-sparsified graphs")
        print("="*80)
        
        comparison_results = {}
        
        for target_name in self.target_names:
            if target_name in knn_results and target_name in explainer_results:
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
                mse_improvement = (knn_mse - explainer_mse) / knn_mse * 100
                rmse_improvement = (knn_rmse - explainer_rmse) / knn_rmse * 100
                r2_improvement = (explainer_r2 - knn_r2) / abs(knn_r2) * 100 if knn_r2 != 0 else 0
                
                # Store results
                comparison_results[target_name] = {
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
                
                print(f"\nResults for target: {target_name}")
                print(f"KNN Graph - MSE: {knn_mse:.4f}, RMSE: {knn_rmse:.4f}, R²: {knn_r2:.4f}")
                print(f"GNNExplainer Graph - MSE: {explainer_mse:.4f}, RMSE: {explainer_rmse:.4f}, R²: {explainer_r2:.4f}")
                print(f"Improvement - MSE: {mse_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%")
        
        # Create comparison plot
        self._plot_comparison_results(comparison_results)
        
        # Save comparison results
        comparison_df = []
        
        for target_name, results in comparison_results.items():
            comparison_df.append({
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
        
        # Convert to DataFrame and save
        comparison_df = pd.DataFrame(comparison_df)
        comparison_df.to_csv(f"{self.save_dir}/comparison_results.csv", index=False)
        
        # Return consolidated results
        return {
            'knn': knn_results,
            'explainer': explainer_results,
            'comparison': comparison_results
        }
    
    def _plot_comparison_results(self, comparison_results):
        """Create a comparison bar plot for KNN vs GNNExplainer graph results"""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Setup parameters for plotting
        n_targets = len(comparison_results)
        width = 0.35
        ind = np.arange(n_targets)
        
        # Extract metrics
        target_names = list(comparison_results.keys())
        rmse_knn = [comparison_results[t]['knn']['rmse'] for t in target_names]
        rmse_explainer = [comparison_results[t]['explainer']['rmse'] for t in target_names]
        r2_knn = [comparison_results[t]['knn']['r2'] for t in target_names]
        r2_explainer = [comparison_results[t]['explainer']['r2'] for t in target_names]
        
        # Plot RMSE
        plt.subplot(2, 1, 1)
        plt.bar(ind - width/2, rmse_knn, width, label='KNN Graph', color='skyblue')
        plt.bar(ind + width/2, rmse_explainer, width, label='GNNExplainer Graph', color='salmon')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('RMSE')
        plt.title('Comparison of RMSE between KNN and GNNExplainer-sparsified Graphs')
        plt.xticks(ind, target_names)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        for i, (knn, explainer) in enumerate(zip(rmse_knn, rmse_explainer)):
            improvement = (knn - explainer) / knn * 100
            if improvement > 0:
                plt.text(i, max(knn, explainer) + 0.05, f"{improvement:.1f}%↓", ha='center')
            else:
                plt.text(i, max(knn, explainer) + 0.05, f"{-improvement:.1f}%↑", ha='center')
        
        # Plot R²
        plt.subplot(2, 1, 2)
        plt.bar(ind - width/2, r2_knn, width, label='KNN Graph', color='skyblue')
        plt.bar(ind + width/2, r2_explainer, width, label='GNNExplainer Graph', color='salmon')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('R²')
        plt.title('Comparison of R² between KNN and GNNExplainer-sparsified Graphs')
        plt.xticks(ind, target_names)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        for i, (knn, explainer) in enumerate(zip(r2_knn, r2_explainer)):
            improvement = (explainer - knn) / abs(knn) * 100 if knn != 0 else 0
            if improvement > 0:
                plt.text(i, max(knn, explainer) + 0.05, f"{improvement:.1f}%↑", ha='center')
            else:
                plt.text(i, max(knn, explainer) + 0.05, f"{-improvement:.1f}%↓", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/comparison_plot.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='GNN Regression Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for KNN graph sparsification')
    parser.add_argument('--mantel_threshold', type=float, default=0.05, help='P-value threshold for Mantel test')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'gat', 'rggc'], help='Type of GNN model')
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
    
    args = parser.parse_args()
    
    # Print arguments
    print("Pipeline parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create and run pipeline
    pipeline = RegressionPipeline(
        data_path=args.data_path,
        k_neighbors=args.k_neighbors,
        mantel_threshold=args.mantel_threshold,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        num_folds=args.num_folds,
        save_dir=args.save_dir,
        importance_threshold=args.importance_threshold,
        estimate_uncertainty=args.estimate_uncertainty
    )
    
    # Run the pipeline
    results = pipeline.run_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to {args.save_dir}") 