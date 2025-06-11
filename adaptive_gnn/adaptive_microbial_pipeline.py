#!/usr/bin/env python3
"""
Adaptive Microbial GNN Pipeline
==============================

This pipeline follows the same structure as mixed_embedding_pipeline.py but uses
adaptive graph neural networks that learn microbial interaction networks dynamically.

Key differences from static pipeline:
- Learns graph structure from data (no KNN or fixed topology)
- Sample-specific edge weights
- Attention-based microbial interactions
- Biologically interpretable network discovery
"""

import os
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from models.adaptive_microbial_gnn import AdaptiveMicrobialGNN, BiologicalConstraints


class AdaptiveMicrobialPipeline:
    """
    Pipeline for adaptive microbial GNN analysis following mixed_embedding_pipeline structure
    """
    
    def __init__(self, 
                 data_path=None, 
                 target_columns=['ACE-km', 'H2-km'],
                 random_state=42,
                 device=None):
        """
        Initialize the adaptive pipeline
        
        Args:
            data_path: Path to microbial data CSV
            target_columns: List of target variable names
            random_state: Random seed for reproducibility
            device: PyTorch device (cuda/cpu)
        """
        self.data_path = data_path
        self.target_columns = target_columns
        self.random_state = random_state
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data containers
        self.data = None
        self.microbial_features = None
        self.targets = None
        self.microbial_families = None
        self.scaler = StandardScaler()
        
        # Model containers
        self.models = {}
        self.results = {}
        self.embeddings = {}
        self.learned_graphs = {}
        
        print(f"🧬 Adaptive Microbial GNN Pipeline initialized")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Targets: {target_columns}")
    
    def load_data(self, data_path=None):
        """
        Load microbial abundance data
        """
        if data_path:
            self.data_path = data_path
        
        print(f"\n📂 Loading data from: {self.data_path}")
        
        # Load CSV data
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Data shape: {self.data.shape}")
        
        # Extract microbial family columns (excluding targets and metadata)
        exclude_cols = self.target_columns + ['Sample_ID'] if 'Sample_ID' in self.data.columns else self.target_columns
        self.microbial_families = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"🦠 Number of microbial families: {len(self.microbial_families)}")
        
        # Extract features and targets
        self.microbial_features = self.data[self.microbial_families].values.astype(np.float32)
        self.targets = self.data[self.target_columns].values.astype(np.float32)
        
        # Normalize features
        self.microbial_features = self.scaler.fit_transform(self.microbial_features)
        
        print(f"✅ Data loaded successfully")
        print(f"   - Features: {self.microbial_features.shape}")
        print(f"   - Targets: {self.targets.shape}")
        
        # Print some statistics
        print(f"\n📈 Feature Statistics:")
        print(f"   - Mean abundance: {np.mean(self.microbial_features):.4f}")
        print(f"   - Std abundance: {np.std(self.microbial_features):.4f}")
        print(f"   - Min abundance: {np.min(self.microbial_features):.4f}")
        print(f"   - Max abundance: {np.max(self.microbial_features):.4f}")
    
    def create_data_objects(self):
        """
        Create PyTorch Geometric data objects
        """
        print(f"\n🔧 Creating adaptive GNN data objects...")
        
        data_objects = []
        num_samples, num_features = self.microbial_features.shape
        
        for i in range(num_samples):
            # Node features: each microbial family is a node with abundance as feature
            x = torch.tensor(self.microbial_features[i].reshape(-1, 1), dtype=torch.float)  # [num_families, 1]
            
            # Target values for this sample
            y = torch.tensor(self.targets[i], dtype=torch.float)  # [num_targets]
            
            # Create data object (no edge_index - will be learned by model)
            data = Data(x=x, y=y)
            data_objects.append(data)
        
        print(f"✅ Created {len(data_objects)} adaptive data objects")
        print(f"   - Node features per sample: {data_objects[0].x.shape}")
        print(f"   - Target shape per sample: {data_objects[0].y.shape}")
        
        return data_objects
    
    def train_adaptive_model(self, 
                           target_name, 
                           target_idx,
                           data_objects,
                           model_config=None):
        """
        Train adaptive microbial GNN for a specific target
        
        Args:
            target_name: Name of the target variable
            target_idx: Index of target in targets array
            data_objects: List of PyTorch Geometric data objects
            model_config: Model configuration dictionary
        """
        print(f"\n🧠 Training Adaptive GNN for {target_name}...")
        
        # Default model configuration
        default_config = {
            'hidden_dim': 64,
            'num_heads': 4,
            'dropout': 0.3,
            'sparsity_factor': 0.15,
            'num_gnn_layers': 3,
            'learning_rate': 0.001,
            'epochs': 200,
            'patience': 30,
            'batch_size': 16
        }
        
        if model_config:
            default_config.update(model_config)
        config = default_config
        
        # Create model
        num_nodes = len(self.microbial_families)
        model = AdaptiveMicrobialGNN(
            num_nodes=num_nodes,
            input_dim=1,
            hidden_dim=config['hidden_dim'],
            output_dim=1,
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            sparsity_factor=config['sparsity_factor'],
            num_gnn_layers=config['num_gnn_layers']
        ).to(self.device)
        
        # Prepare data for single target
        single_target_data = []
        for data in data_objects:
            new_data = Data(x=data.x, y=data.y[target_idx:target_idx+1])  # Single target
            single_target_data.append(new_data)
        
        # Create data loader
        train_loader = DataLoader(single_target_data, 
                                batch_size=config['batch_size'], 
                                shuffle=True)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                             patience=10, 
                                                             factor=0.5)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        
        print(f"📚 Training configuration:")
        for key, value in config.items():
            if key != 'epochs':  # Don't print epochs twice
                print(f"   - {key}: {value}")
        
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                predictions, _ = model(batch.x, batch=batch.batch)
                loss = criterion(predictions.squeeze(), batch.y.squeeze())
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"   Epoch {epoch:3d}: Loss = {avg_loss:.6f}, Best = {best_loss:.6f}")
            
            if patience_counter >= config['patience']:
                print(f"   ⏹️  Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        print(f"✅ Training completed for {target_name}")
        print(f"   - Final loss: {best_loss:.6f}")
        print(f"   - Training epochs: {len(train_losses)}")
        
        return model, train_losses, config
    
    def evaluate_model(self, model, data_objects, target_idx, cv_folds=5):
        """
        Evaluate model using cross-validation
        """
        print(f"\n📊 Evaluating model with {cv_folds}-fold cross-validation...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        fold_scores = []
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Prepare single target data
        X = np.array([data.x.cpu().numpy() for data in data_objects])
        y = np.array([data.y[target_idx].cpu().numpy() for data in data_objects])
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"   📋 Fold {fold + 1}/{cv_folds}...")
            
            # Create test data objects
            test_data = [data_objects[i] for i in test_idx]
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
            
            fold_predictions = []
            fold_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    predictions, _ = model(batch.x, batch=batch.batch)
                    
                    fold_predictions.extend(predictions.cpu().numpy().flatten())
                    fold_targets.extend(batch.y[:, 0].cpu().numpy())  # Single target
            
            # Calculate fold metrics
            fold_mse = mean_squared_error(fold_targets, fold_predictions)
            fold_r2 = r2_score(fold_targets, fold_predictions)
            fold_scores.append({'mse': fold_mse, 'r2': fold_r2})
            
            all_predictions.extend(fold_predictions)
            all_targets.extend(fold_targets)
        
        # Overall metrics
        overall_mse = mean_squared_error(all_targets, all_predictions)
        overall_r2 = r2_score(all_targets, all_predictions)
        
        # Calculate mean and std across folds
        fold_mses = [score['mse'] for score in fold_scores]
        fold_r2s = [score['r2'] for score in fold_scores]
        
        results = {
            'mse': overall_mse,
            'r2': overall_r2,
            'mse_mean': np.mean(fold_mses),
            'mse_std': np.std(fold_mses),
            'r2_mean': np.mean(fold_r2s),
            'r2_std': np.std(fold_r2s),
            'fold_scores': fold_scores,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        print(f"✅ Evaluation completed:")
        print(f"   - Overall R²: {overall_r2:.4f}")
        print(f"   - Overall MSE: {overall_mse:.4f}")
        print(f"   - CV R² (mean±std): {np.mean(fold_r2s):.4f}±{np.std(fold_r2s):.4f}")
        print(f"   - CV MSE (mean±std): {np.mean(fold_mses):.4f}±{np.std(fold_mses):.4f}")
        
        return results
    
    def extract_embeddings(self, model, data_objects, target_name):
        """
        Extract graph embeddings from trained model
        """
        print(f"\n🎯 Extracting embeddings for {target_name}...")
        
        model.eval()
        embeddings_list = []
        
        data_loader = DataLoader(data_objects, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                _, embeddings = model(batch.x, batch=batch.batch)
                embeddings_list.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings_list)
        
        print(f"✅ Extracted embeddings: {all_embeddings.shape}")
        
        return all_embeddings
    
    def analyze_learned_graphs(self, model, data_objects, target_name):
        """
        Analyze learned graph structures
        """
        print(f"\n🕸️  Analyzing learned graphs for {target_name}...")
        
        model.eval()
        learned_structures = []
        
        with torch.no_grad():
            for i, data in enumerate(data_objects[:10]):  # Analyze first 10 samples
                data = data.to(self.device)
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                
                graph_info = model.get_learned_graph(data.x, batch)
                learned_structures.append({
                    'sample_idx': i,
                    'adjacency_matrix': graph_info['adjacency_matrices'][0].cpu().numpy(),
                    'edge_index': graph_info['edge_index'].cpu().numpy(),
                    'edge_weights': graph_info['edge_weights'].cpu().numpy()
                })
        
        print(f"✅ Analyzed {len(learned_structures)} graph structures")
        
        return learned_structures
    
    def visualize_learned_network(self, learned_structures, target_name, sample_idx=0):
        """
        Create publication-quality network visualization
        """
        import networkx as nx
        
        print(f"\n🎨 Creating network visualization for {target_name} (Sample {sample_idx})...")
        
        if sample_idx >= len(learned_structures):
            sample_idx = 0
        
        structure = learned_structures[sample_idx]
        adj_matrix = structure['adjacency_matrix']
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Create high-quality visualization
        plt.figure(figsize=(15, 10))
        
        # Layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Node properties
        degrees = dict(G.degree())
        node_sizes = [max(200, degrees[node] * 50) for node in G.nodes()]
        
        # Edge properties - only strong connections
        edge_weights = [adj_matrix[u, v] for u, v in G.edges()]
        strong_edges = [(u, v) for u, v in G.edges() if adj_matrix[u, v] > np.percentile(edge_weights, 70)]
        
        # Color nodes by degree (hub detection)
        node_colors = []
        for node in G.nodes():
            degree = degrees[node]
            if degree > np.percentile(list(degrees.values()), 90):
                node_colors.append('#FF6B6B')  # Red for hubs
            elif degree > np.percentile(list(degrees.values()), 70):
                node_colors.append('#4ECDC4')  # Teal for important
            else:
                node_colors.append('#95A5A6')  # Gray for regular
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, 
                             node_size=node_sizes, 
                             node_color=node_colors,
                             alpha=0.8, 
                             edgecolors='black',
                             linewidths=0.5)
        
        # Draw only strong edges
        if strong_edges:
            nx.draw_networkx_edges(G, pos, 
                                 edgelist=strong_edges,
                                 edge_color='#2C3E50',
                                 alpha=0.6,
                                 width=1.5)
        
        # Add labels for important nodes (top 20% by degree)
        important_nodes = [node for node in G.nodes() 
                          if degrees[node] > np.percentile(list(degrees.values()), 80)]
        
        node_labels = {}
        for node in important_nodes:
            if node < len(self.microbial_families):
                family_name = self.microbial_families[node]
                # Truncate long names
                if len(family_name) > 15:
                    family_name = family_name[:12] + "..."
                node_labels[node] = family_name
        
        if node_labels:
            nx.draw_networkx_labels(G, pos, 
                                  labels=node_labels,
                                  font_size=8,
                                  font_weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor='white', 
                                          alpha=0.8,
                                          edgecolor='black'))
        
        plt.title(f'Learned Microbial Network - {target_name}\n'
                 f'Sample {sample_idx} | {len(strong_edges)} connections shown', 
                 fontsize=16, pad=20)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        plot_path = f'plots/adaptive_network_{target_name.lower()}_sample_{sample_idx}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Network visualization saved: {plot_path}")
        
        return plot_path
    
    def run_complete_pipeline(self):
        """
        Run the complete adaptive microbial GNN pipeline
        """
        print(f"\n🚀 Starting Complete Adaptive Microbial GNN Pipeline")
        print(f"=" * 60)
        
        # 1. Load data
        if not hasattr(self, 'data') or self.data is None:
            print("❌ No data loaded. Please call load_data() first.")
            return
        
        # 2. Create data objects
        data_objects = self.create_data_objects()
        
        # 3. Train models for each target
        for target_idx, target_name in enumerate(self.target_columns):
            print(f"\n🎯 Processing target: {target_name}")
            print(f"=" * 40)
            
            # Train model
            model, train_losses, config = self.train_adaptive_model(
                target_name, target_idx, data_objects
            )
            
            # Evaluate model
            results = self.evaluate_model(model, data_objects, target_idx)
            
            # Extract embeddings
            embeddings = self.extract_embeddings(model, data_objects, target_name)
            
            # Analyze learned graphs
            learned_graphs = self.analyze_learned_graphs(model, data_objects, target_name)
            
            # Visualize network
            self.visualize_learned_network(learned_graphs, target_name)
            
            # Store results
            self.models[target_name] = model
            self.results[target_name] = results
            self.embeddings[target_name] = embeddings
            self.learned_graphs[target_name] = learned_graphs
            
            print(f"✅ Completed processing for {target_name}")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Results summary:")
        for target_name in self.target_columns:
            r2 = self.results[target_name]['r2']
            print(f"   - {target_name}: R² = {r2:.4f}")
        
        return {
            'models': self.models,
            'results': self.results,
            'embeddings': self.embeddings,
            'learned_graphs': self.learned_graphs
        }


def main():
    """
    Main function to run the adaptive pipeline
    """
    # Initialize pipeline
    pipeline = AdaptiveMicrobialPipeline()
    
    # Load data (update path as needed)
    data_path="../Data/New_data.csv"
    
    if os.path.exists(data_path):
        pipeline.load_data(data_path)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print(f"\n📁 Results saved to:")
        print(f"   - Models: Stored in pipeline.models")
        print(f"   - Embeddings: Stored in pipeline.embeddings")
        print(f"   - Networks: Saved in plots/")
        
    else:
        print(f"❌ Data file not found: {data_path}")
        print(f"Please update the data_path in main() function")


if __name__ == "__main__":
    main() 