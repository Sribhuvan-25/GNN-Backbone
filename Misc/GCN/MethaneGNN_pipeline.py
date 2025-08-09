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

# Import custom modules
from MethaneGNN_dataset import MethaneGNNDataset
from MethaneGNN_models import MethaneGNN, MethaneGNNExplainer, simple_GCN_res, simple_RGGC, simple_GAT

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)


class MethanePipeline:
    """Complete pipeline for methane prediction with graph sparsification and GNN explanation"""
    
    def __init__(self, 
                 data_path='../Data/New_data.csv',
                 k_neighbors=5,
                 mantel_threshold=0.05,
                 model_type='gat',
                 model_architecture='default',  # Added parameter for model architecture
                 hidden_dim=64,
                 num_layers=4,
                 dropout_rate=0.3,
                 batch_size=8,
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 num_epochs=300,
                 patience=30,
                 num_folds=5,
                 save_dir='./methane_results',
                 visualize_graphs=True):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the CSV file with microbial data
            k_neighbors: Number of neighbors for KNN graph sparsification
            mantel_threshold: p-value threshold for Mantel test
            model_type: Type of GNN model ('gcn', 'gat', 'gatv2', 'gin')
            model_architecture: Architecture to use ('default', 'simple_gcn_res', 'simple_rggc', 'simple_gat')
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout_rate: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
            num_folds: Number of folds for cross-validation
            save_dir: Directory to save results
            visualize_graphs: Whether to visualize the graphs
        """
        self.data_path = data_path
        self.k_neighbors = k_neighbors
        self.mantel_threshold = mantel_threshold
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_folds = num_folds
        self.save_dir = save_dir
        self.visualize_graphs = visualize_graphs
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations/ACE-km", exist_ok=True)
        os.makedirs(f"{save_dir}/explanations/H2-km", exist_ok=True)
        
        # Target names
        self.target_names = ['ACE-km', 'H2-km']
        
        # Load and process data
        self.dataset = MethaneGNNDataset(
            data_path=data_path,
            k_neighbors=k_neighbors,
            mantel_threshold=mantel_threshold
        )
        
        # Create KNN sparsified graph
        self.sparsified_data = self.dataset.create_knn_sparsified_graph(k=k_neighbors)
        
        # Visualize the graphs if requested
        if self.visualize_graphs:
            print("\nVisualizing original and sparsified graphs...")
            self.dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
            print(f"Graph visualizations saved to {self.save_dir}/graphs/")
        
        # Store the data
        self.data_list = self.sparsified_data
        self.node_feature_names = self.dataset.node_feature_names
    
    def _create_model(self, num_targets=1):
        """Create model based on specified architecture and type"""
        if self.model_architecture == 'default':
            model = MethaneGNN(
                num_node_features=1,
                hidden_dim=self.hidden_dim,
                num_targets=num_targets,
                dropout_rate=self.dropout_rate,
                num_layers=self.num_layers,
                model_type=self.model_type
            ).to(device)
        elif self.model_architecture == 'simple_gcn_res':
            model = simple_GCN_res(
                hidden_channels=self.hidden_dim,
                n_targets=num_targets,
                dropout_prob=self.dropout_rate,
                in_channels=1
            ).to(device)
        elif self.model_architecture == 'simple_rggc':
            model = simple_RGGC(
                hidden_channels=self.hidden_dim,
                n_targets=num_targets,
                dropout_prob=self.dropout_rate,
                in_channels=1
            ).to(device)
        elif self.model_architecture == 'simple_gat':
            model = simple_GAT(
                hidden_channels=self.hidden_dim,
                n_targets=num_targets,
                dropout_prob=self.dropout_rate,
                in_channels=1,
                num_heads=4  # Increase heads for better feature learning
            ).to(device)
        else:
            raise ValueError(f"Unknown model architecture: {self.model_architecture}")
        
        return model
    
    def train_single_target(self, target_idx, target_name):
        """Train a model for a single target variable"""
        print(f"\n{'='*50}")
        print(f"Training model for {target_name}")
        print(f"{'='*50}")
        
        # Prepare data with single target
        single_target_data_list = []
        target_values = []
        
        for data in self.data_list:
            # Extract target values for normalization
            target_value = data.y[0, target_idx].item()
            target_values.append(target_value)
        
        # Compute target statistics for normalization
        target_mean = np.mean(target_values)
        target_std = np.std(target_values)
        if target_std == 0:  # Avoid division by zero
            target_std = 1.0
            
        print(f"Target normalization - mean: {target_mean:.4f}, std: {target_std:.4f}")
        
        # Create normalized data list
        for i, data in enumerate(self.data_list):
            # Create a copy with only the specific target
            target_value = data.y[0, target_idx].item()
            
            # Normalize target value
            normalized_target = (target_value - target_mean) / target_std
            
            single_target_data = data.clone()
            single_target_data.y = torch.tensor([[normalized_target]], dtype=torch.float32)
            # Store original value for later denormalization
            single_target_data.original_y = torch.tensor([[target_value]], dtype=torch.float32)
            
            single_target_data_list.append(single_target_data)
        
        # Setup k-fold cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        
        # Create loss function outside the fold loop
        criterion = nn.MSELoss()
        
        # Iterate through folds
        for fold, (train_index, test_index) in enumerate(kf.split(single_target_data_list)):
            fold_num = fold + 1
            print(f"Fold {fold_num}: Train on {len(train_index)} samples, Test on {len(test_index)} samples")
            
            # Split into train and test sets
            train_dataset = [single_target_data_list[i] for i in train_index]
            test_dataset = [single_target_data_list[i] for i in test_index]
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model using the helper method
            model = self._create_model(num_targets=1)
            
            # Setup optimizer and loss function
            optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=self.weight_decay)
            
            # Use OneCycleLR scheduler for better convergence
            scheduler = lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=0.005,
                steps_per_epoch=len(train_loader),
                epochs=self.num_epochs,
                pct_start=0.3
            )
            
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
                    out = model(batch_data)
                    
                    # Handle different return types from models
                    if isinstance(out, tuple):
                        # Get the prediction (first element of tuple)
                        pred = out[0]
                        
                        # Check if pred needs reshaping (if it's not already the right shape)
                        if len(pred.shape) == 1:  # If pred is [batch_size]
                            pred = pred.view(-1, 1)  # Convert to [batch_size, 1]
                        
                        # Check if this is a 3-tuple (pred, log_var, embedding) for uncertainty
                        if len(out) == 3 and hasattr(model, 'estimate_uncertainty') and model.estimate_uncertainty:
                            # Unpack predictions and log variance
                            pred, log_var, _ = out  # Ignore embeddings
                            
                            # Reshape target
                            target = batch_data.y.view(-1, 1)
                            
                            # Implement uncertainty-aware loss 
                            precision = torch.exp(-log_var)
                            loss = torch.mean(precision * (pred - target) ** 2 + log_var)
                        else:
                            # Standard MSE loss for (pred, embedding) tuple
                            target = batch_data.y.view(-1, 1)
                            loss = criterion(pred, target)
                    else:
                        # Standard prediction, no tuple
                        pred = out
                        target = batch_data.y.view(-1, 1)
                        loss = criterion(pred, target)
                    
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item() * batch_data.num_graphs
                
                avg_loss = total_loss / len(train_dataset)
                train_losses.append(avg_loss)
                
                # Evaluate on test set
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        out = model(batch_data)
                        
                        # Handle tuple output from model
                        if isinstance(out, tuple):
                            pred = out[0]
                            if len(pred.shape) == 1:  # If pred is [batch_size]
                                pred = pred.view(-1, 1)  # Convert to [batch_size, 1]
                        else:
                            pred = out
                            if len(pred.shape) == 1:  # If pred is [batch_size]
                                pred = pred.view(-1, 1)  # Convert to [batch_size, 1]
                        
                        # Denormalize predictions
                        pred_denorm = pred * target_std + target_mean
                        
                        # Get original targets
                        target_denorm = batch_data.original_y
                        
                        # Convert to numpy
                        pred_np = pred_denorm.cpu().numpy()
                        target_np = target_denorm.cpu().numpy()
                        
                        # For validation loss (normalized domain)
                        target = batch_data.y.view(-1, 1)
                        val_loss += criterion(pred, target).item() * batch_data.num_graphs
                    
                    val_loss /= len(test_dataset)
                    val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step()
                
                # Calculate denormalized validation metrics for early stopping
                val_mse_denorm = 0
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        out = model(batch_data)
                        
                        # Handle tuple output from model
                        if isinstance(out, tuple):
                            pred = out[0]
                            if len(pred.shape) == 1:
                                pred = pred.view(-1, 1)
                        else:
                            pred = out
                            if len(pred.shape) == 1:
                                pred = pred.view(-1, 1)
                        
                        # Denormalize predictions
                        pred_denorm = pred * target_std + target_mean
                        
                        # Get original targets
                        target_denorm = batch_data.original_y
                        
                        # Calculate MSE on denormalized values
                        val_mse_denorm += F.mse_loss(pred_denorm, target_denorm).item() * batch_data.num_graphs
                    
                    val_mse_denorm /= len(test_dataset)
                
                # Print both normalized and denormalized metrics
                if epoch % 20 == 0 or epoch == self.num_epochs:
                    print(f"Epoch {epoch:03d} - Train MSE: {avg_loss:.4f}, Val MSE: {val_loss:.4f}, Val MSE (denorm): {val_mse_denorm:.4f}")
                
                # Early stopping based on denormalized metrics
                if val_mse_denorm < best_val_loss:
                    best_val_loss = val_mse_denorm
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 20 == 0 or epoch == self.num_epochs:
                    print(f"Epoch {epoch:03d} - Train MSE: {avg_loss:.4f}, Val MSE: {val_loss:.4f}")
            
            # Load best model for this fold
            model.load_state_dict(best_model_state)
            
            # Final evaluation on test set
            model.eval()
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    out = model(batch_data)
                    
                    # Handle tuple output from model
                    if isinstance(out, tuple):
                        pred = out[0]
                        if len(pred.shape) == 1:  # If pred is [batch_size]
                            pred = pred.view(-1, 1)  # Convert to [batch_size, 1]
                    else:
                        pred = out
                        if len(pred.shape) == 1:  # If pred is [batch_size]
                            pred = pred.view(-1, 1)  # Convert to [batch_size, 1]
                    
                    # Denormalize predictions
                    pred_denorm = pred * target_std + target_mean
                    
                    # Get original targets
                    target_denorm = batch_data.original_y
                    
                    # Convert to numpy
                    pred_np = pred_denorm.cpu().numpy()
                    target_np = target_denorm.cpu().numpy()
                    
                    all_preds.append(pred_np)
                    all_trues.append(target_np)
            
            # Check for shape consistency before vstack
            if len(all_preds) > 0:
                # Get the shape of each array in the list
                shapes = [arr.shape[1] for arr in all_preds]
                
                # If shapes are not all the same, fix them
                if len(set(shapes)) > 1:
                    max_cols = max(shapes)
                    for i in range(len(all_preds)):
                        if all_preds[i].shape[1] < max_cols:
                            # Pad arrays with fewer columns
                            all_preds[i] = np.pad(all_preds[i], ((0, 0), (0, max_cols - all_preds[i].shape[1])))
            
            all_preds = np.vstack(all_preds)
            all_trues = np.vstack(all_trues)
            
            fold_mse = mean_squared_error(all_trues, all_preds)
            fold_rmse = np.sqrt(fold_mse)
            fold_r2 = r2_score(all_trues, all_preds)
            
            print(f"Fold {fold_num} results: MSE = {fold_mse:.4f}, RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")
            
            # Store fold results
            fold_results.append({
                'fold': fold_num,
                'preds': all_preds,
                'trues': all_trues,
                'mse': fold_mse,
                'rmse': fold_rmse,
                'r2': fold_r2
            })
            
            # Save the best model for this fold
            torch.save(model.state_dict(), f"{self.save_dir}/models/{target_name}_model_fold{fold_num}.pt")
            
            # Create plot for this fold
            plt.figure(figsize=(8, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title(f'{target_name} - Fold {fold_num} Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.savefig(f"{self.save_dir}/plots/{target_name}_loss_fold{fold_num}.png")
            plt.close()
            
            # Generate explanations for test samples
            self._generate_explanations(model, test_dataset, target_name, fold_num, target_idx)
        
        # Train final model on all data
        # Commenting out final model training on all data
        # self._train_final_model(single_target_data_list, target_name, target_idx)
        
        # Calculate and display overall performance
        self._evaluate_performance(fold_results, target_name)
        
        return fold_results
    
    def _train_final_model(self, data_list, target_name, target_idx):
        """Train a final model on all data"""
        # Commenting out the entire final model training method
        """
        print(f"\nTraining final {target_name} model on all data...")
        
        # Create dataloader with all data
        all_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        final_model = self._create_model(num_targets=1)
        
        # Setup optimizer and loss function
        optimizer = Adam(final_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-5
        )
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(1, self.num_epochs+1):
            final_model.train()
            total_loss = 0
            
            for batch_data in all_loader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                out = final_model(batch_data)
                target = batch_data.y.view(-1, 1)
                
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_data.num_graphs
            
            # Update learning rate
            scheduler.step()
            
            if epoch % 20 == 0 or epoch == self.num_epochs:
                print(f"Epoch {epoch:03d} - Train MSE: {total_loss/len(all_loader):.4f}")
        
        # Save the final model
        torch.save(final_model.state_dict(), f"{self.save_dir}/models/{target_name}_final_model.pt")
        print(f"Final {target_name} model saved to {self.save_dir}/models/{target_name}_final_model.pt")
        
        # Generate explanations using the final model
        self._generate_explanations(final_model, data_list[:10], target_name, 'final', target_idx)
        """
        pass
    
    def _generate_explanations(self, model, dataset, target_name, fold, target_idx):
        """Generate and save GNN explanations"""
        # Initialize explainer
        explainer = MethaneGNNExplainer(model, device)
        
        # Create output directory
        os.makedirs(f"{self.save_dir}/explanations/{target_name}/fold_{fold}", exist_ok=True)
        
        # Generate explanations for each sample
        for i, data in enumerate(dataset[:5]):  # Limit to first 5 samples to save time
            # Generate explanation
            edge_importance_matrix, explanation = explainer.explain_graph(
                data, 
                node_names=self.node_feature_names,
                save_path=f"{self.save_dir}/explanations/{target_name}/fold_{fold}/sample_{i}_edge_importance.csv",
                target_idx=0  # For single target models, target_idx is always 0
            )
            
            # Visualize the explanation as a network
            self._visualize_explanation(
                data, 
                edge_importance_matrix, 
                i, 
                f"{self.save_dir}/explanations/{target_name}/fold_{fold}/sample_{i}_explanation.png",
                target_name
            )
    
    def _visualize_explanation(self, data, edge_importance_matrix, sample_id, save_path, target_name):
        """Visualize explanation as a network with edge importance"""
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(self.node_feature_names):
            G.add_node(i, name=name, abundance=data.x[i].item())
        
        # Add edges with importance as weights - use a higher threshold to reduce disconnected components
        edge_importance = edge_importance_matrix.cpu().numpy()
        
        # Normalize edge importance to [0, 1] range
        if edge_importance.max() > 0:
            edge_importance = edge_importance / edge_importance.max()
        
        # Add edges with importance above a threshold
        threshold = 0.2  # Only keep edges with at least 20% of max importance
        for i in range(len(self.node_feature_names)):
            for j in range(i+1, len(self.node_feature_names)):
                importance = edge_importance[i, j]
                if importance > threshold:
                    G.add_edge(i, j, weight=float(importance))
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Create layout
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Get node sizes based on abundance
        node_size = [1000 * (0.1 + data.x[i].item()) for i in range(len(self.node_feature_names))]
        
        # Get edge widths based on importance
        edge_width = []
        for u, v, d in G.edges(data=True):
            edge_width.append(d['weight'] * 5)
        
        # Draw the network
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            labels={i: self.node_feature_names[i] for i in G.nodes()},
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
    
    def _evaluate_performance(self, fold_results, target_name):
        """Calculate and display overall performance metrics"""
        # Check for shape consistency across folds
        if fold_results:
            # Check shapes of prediction arrays
            pred_shapes = [fold['preds'].shape[1] for fold in fold_results]
            
            # If not all the same, fix them
            if len(set(pred_shapes)) > 1:
                max_cols = max(pred_shapes)
                for fold in fold_results:
                    if fold['preds'].shape[1] < max_cols:
                        # Pad arrays with fewer columns
                        fold['preds'] = np.pad(fold['preds'], ((0, 0), (0, max_cols - fold['preds'].shape[1])))
                        # Also pad the true values to match
                        fold['trues'] = np.pad(fold['trues'], ((0, 0), (0, max_cols - fold['trues'].shape[1])))
        
        # Combine predictions from all folds
        all_preds = np.vstack([fold['preds'] for fold in fold_results])
        all_trues = np.vstack([fold['trues'] for fold in fold_results])
        
        # Calculate metrics
        mse = mean_squared_error(all_trues, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_trues, all_preds)
        
        print(f"\n{target_name} Cross-Validation Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(all_trues, all_preds, facecolors='none', edgecolors='k', marker='o', alpha=0.6)
        
        # Set axis limits based on the target variable
        if target_name == 'ACE-km':
            plt.xlim(0, 50)
            plt.ylim(0, 50)
        elif target_name == 'H2-km':
            plt.xlim(0, 140)
            plt.ylim(0, 140)
        
        # Add 45-degree line
        plt.plot([0, plt.xlim()[1]], [0, plt.ylim()[1]], 'r--')
        
        plt.title(f'{target_name}\nMSE={mse:.4f}, R²={r2:.4f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.save_dir}/plots/{target_name}_overall_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            'Target': target_name,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }])
        metrics_df.to_csv(f"{self.save_dir}/plots/{target_name}_metrics.csv", index=False)
    
    def run_pipeline(self):
        """Run the complete pipeline for all targets"""
        results = {}
        
        # Train separate model for each target
        for target_idx, target_name in enumerate(self.target_names):
            results[target_name] = self.train_single_target(target_idx, target_name)
        
        # Create combined plot for both targets
        plt.figure(figsize=(15, 7))
        
        for i, target_name in enumerate(self.target_names):
            # Get data
            fold_results = results[target_name]
            
            # Check for shape consistency across folds
            if fold_results:
                # Check shapes of prediction arrays
                pred_shapes = [fold['preds'].shape[1] for fold in fold_results]
                
                # If not all the same, fix them
                if len(set(pred_shapes)) > 1:
                    max_cols = max(pred_shapes)
                    for fold in fold_results:
                        if fold['preds'].shape[1] < max_cols:
                            # Pad arrays with fewer columns
                            fold['preds'] = np.pad(fold['preds'], ((0, 0), (0, max_cols - fold['preds'].shape[1])))
                            # Also pad the true values to match
                            fold['trues'] = np.pad(fold['trues'], ((0, 0), (0, max_cols - fold['trues'].shape[1])))
            
            all_preds = np.vstack([fold['preds'] for fold in fold_results])
            all_trues = np.vstack([fold['trues'] for fold in fold_results])
            
            # Calculate metrics
            mse = mean_squared_error(all_trues, all_preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(all_trues, all_preds)
            
            # Create subplot
            plt.subplot(1, 2, i+1)
            plt.scatter(all_trues, all_preds, facecolors='none', edgecolors='k', marker='o', alpha=0.6)
            
            # Set axis limits based on the target variable
            if target_name == 'ACE-km':
                plt.xlim(0, 50)
                plt.ylim(0, 50)
            elif target_name == 'H2-km':
                plt.xlim(0, 140)
                plt.ylim(0, 140)
            
            # Add 45-degree line
            plt.plot([0, plt.xlim()[1]], [0, plt.ylim()[1]], 'r--')
            
            plt.title(f'{target_name}\nMSE={mse:.4f}, R²={r2:.4f}')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/plots/combined_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to {self.save_dir}")
        
        return results


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = MethanePipeline(
        data_path='../Data/New_data.csv',
        k_neighbors=12,
        mantel_threshold=0.05,
        model_type='gcn',  # Use GCN as base type 
        model_architecture='default',  # Use model with proper residual connections
        hidden_dim=128,
        num_layers=4,
        dropout_rate=0.3,  # Reduced dropout for better training
        batch_size=8,
        learning_rate=0.001,  # Not used directly now (we use fixed 0.0005 in the code)
        weight_decay=1e-4,  # Reduced weight decay
        num_epochs=300,     # Increased epochs
        patience=30,        # Increased patience 
        num_folds=5,
        save_dir='./methane_results_v2',  # New directory for new results
        visualize_graphs=True
    )
    
    results = pipeline.run_pipeline() 