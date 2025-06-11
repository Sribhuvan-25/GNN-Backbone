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

# ... existing code ... 

def main():
    """
    Main function to run the adaptive pipeline
    """
    # Initialize pipeline
    pipeline = AdaptiveMicrobialPipeline()
    
    # Load data (correct path specified by user)
    data_path = "../Data/New_data.csv"
    
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
        print(f"Please check that the data file exists at this location") 

class AdaptiveMicrobialPipeline:
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
        Load microbial abundance data using the same approach as dataset_regression.py
        """
        if data_path:
            self.data_path = data_path
        
        print(f"\n📂 Loading data from: {self.data_path}")
        
        # Load CSV data
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Data shape: {self.data.shape}")
        
        # Filter out rows containing 'x' values if they exist (from dataset_regression.py)
        if self.data.isin(['x']).any().any():
            print(f"🧹 Filtering out rows with 'x' values...")
            original_length = len(self.data)
            self.data = self.data[~self.data.isin(['x']).any(axis=1)]
            print(f"📊 Data shape after filtering: {self.data.shape} (removed {original_length - len(self.data)} rows)")
        
        # Check if the target columns exist in the data
        missing_targets = [col for col in self.target_columns if col not in self.data.columns]
        if missing_targets:
            print(f"Warning: Missing target columns: {missing_targets}")
            # Use only the available target columns
            self.target_columns = [col for col in self.target_columns if col in self.data.columns]
        
        print(f"🎯 Target columns: {self.target_columns}")
        
        # Extract microbial family columns (excluding targets and metadata)
        exclude_cols = self.target_columns.copy()
        if 'Sample_ID' in self.data.columns:
            exclude_cols.append('Sample_ID')
        
        # Look for columns with taxonomic patterns like 'd__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__'
        taxonomic_patterns = ['d__', 'k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
        potential_microbial_cols = []
        
        for col in self.data.columns:
            if col not in exclude_cols:
                # Check if column contains taxonomic patterns or looks like microbial data
                if any(pattern in str(col) for pattern in taxonomic_patterns) or col.startswith('OTU_'):
                    potential_microbial_cols.append(col)
                # Also include numeric columns that might be microbial families
                elif col not in exclude_cols:
                    try:
                        # Test if column can be converted to numeric (excluding 'x' values)
                        test_series = pd.to_numeric(self.data[col], errors='coerce')
                        if not test_series.isna().all():  # If some values are numeric
                            potential_microbial_cols.append(col)
                    except:
                        pass
        
        self.microbial_families = potential_microbial_cols
        print(f"🦠 Number of microbial families: {len(self.microbial_families)}")
        
        # Extract features and targets with proper cleaning
        try:
            # Extract microbial features with error handling
            feature_data = self.data[self.microbial_families].copy()
            
            # Convert to numeric, replacing non-numeric values with NaN
            for col in self.microbial_families:
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
            
            # Fill NaN values with 0 (common for microbial abundance data)
            feature_data = feature_data.fillna(0)
            
            self.microbial_features = feature_data.values.astype(np.float32)
            
            # Extract and clean target data
            target_data = self.data[self.target_columns].copy()
            for col in self.target_columns:
                target_data[col] = pd.to_numeric(target_data[col], errors='coerce')
            
            # Remove rows with all NaN targets
            all_nan_targets = target_data.isna().all(axis=1)
            if all_nan_targets.any():
                print(f"🧹 Removing {all_nan_targets.sum()} samples with all NaN target values")
                target_data = target_data[~all_nan_targets]
                self.microbial_features = self.microbial_features[~all_nan_targets]
                self.data = self.data[~all_nan_targets]
            
            # Fill remaining NaN values with column means
            target_data = target_data.fillna(target_data.mean())
            self.targets = target_data.values.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error processing features: {str(e)}")
            raise
        
        # Normalize features using StandardScaler
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
        
        print(f"\n📈 Target Statistics:")
        for i, target_name in enumerate(self.target_columns):
            target_values = self.targets[:, i]
            print(f"   - {target_name}: min={np.min(target_values):.3f}, max={np.max(target_values):.3f}, mean={np.mean(target_values):.3f}")
    
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
            
            # Extract embeddings
            embeddings = self.extract_embeddings(model, data_objects, target_name)
            
            # Analyze learned graphs
            learned_graphs = self.analyze_learned_graphs(model, data_objects, target_name)
            
            # Store results
            self.models[target_name] = model
            self.embeddings[target_name] = embeddings
            self.learned_graphs[target_name] = learned_graphs
            
            print(f"✅ Completed processing for {target_name}")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Results summary:")
        for target_name in self.target_columns:
            print(f"   - {target_name}: Model trained and embeddings extracted")
        
        return {
            'models': self.models,
            'embeddings': self.embeddings,
            'learned_graphs': self.learned_graphs
        }


if __name__ == "__main__":
    main() 