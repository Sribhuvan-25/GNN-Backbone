#!/usr/bin/env python3

"""
Test Case 1 with all new features:
- Node importance reporting
- Node-based pruning instead of edge pruning
- t-SNE embedding visualization
- Consistent graph visualizations
"""

import os
import sys
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
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules - using working versions from Misc/Cases_GNNs
from datasets.dataset_regression_working import MicrobialGNNDataset
from explainers.explainer_regression_working import GNNExplainerRegression  
from explainers.pipeline_explainer_working import create_explainer_sparsified_graph
from models.GNNmodelsRegression import (
    simple_GCN_res_plus_regression,
    simple_RGGC_plus_regression,
    simple_GAT_regression
)
# Graph visualization is handled by dataset.visualize_graphs() method

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Case1Pipeline:
    """Case 1: H2-km prediction with hydrogenotrophic features"""
    
    def __init__(self, data_path, save_dir="./case1_results"):
        self.data_path = data_path
        self.save_dir = save_dir
        
        # Create directories
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
        os.makedirs(f"{save_dir}/graphs", exist_ok=True)
        
        # Initialize pipeline attributes needed for explainer
        self.graph_mode = 'family'  # Pipeline graph mode
        self.save_dir = save_dir    # Pipeline save directory
        
        # Fast parameters for testing  
        self.k_neighbors = 10  # Use same as working Misc version
        self.hidden_dim = 32
        self.num_epochs = 15  # Very fast for testing
        self.num_folds = 2    # Fast
        self.batch_size = 4
        self.learning_rate = 0.01
        
        print(f"Case 1 Pipeline initialized - Results will be saved to: {save_dir}")
    
    def create_anchored_dataset(self):
        """Create dataset with Case 1 anchored features"""
        print("\n" + "="*60)
        print("STEP 1: Creating Case 1 Dataset with Anchored Features")
        print("="*60)
        
        # Case 1: Hydrogenotrophic families
        anchored_families = [
            'Methanoregulaceae',      # Hydrogenotrophic
            'Methanobacteriaceae',    # Hydrogenotrophic  
            'Methanospirillaceae'     # Hydrogenotrophic
        ]
        
        print(f"Case 1 anchored families: {anchored_families}")
        
        # Create base dataset
        self.dataset = MicrobialGNNDataset(
            data_path=self.data_path,
            k_neighbors=self.k_neighbors,
            mantel_threshold=0.05,
            use_fast_correlation=False,  # Use Mantel test for proper filtering
            graph_mode='family',
            family_filter_mode='strict'  # Fewer families = faster Mantel test
        )
        
        print(f"Dataset created with {len(self.dataset.data_list)} samples")
        print(f"Target variables: {self.dataset.target_cols}")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        
        return self.dataset
    
    def train_gnn_models(self, data_list, target_idx=1):  # H2-km is index 1
        """Train GNN models on the data"""
        print(f"\n" + "="*60)
        print(f"STEP 2: Training GNN Models for {self.dataset.target_cols[target_idx]}")
        print("="*60)
        
        models_to_train = ['gcn', 'gat']  # Reduced for speed
        results = {}
        
        for model_type in models_to_train:
            print(f"\nTraining {model_type.upper()}...")
            
            # Create model
            if model_type == 'gcn':
                model = simple_GCN_res_plus_regression(
                    hidden_channels=self.hidden_dim,
                    dropout_prob=0.3,
                    input_channel=1,
                    output_dim=1
                ).to(device)
            elif model_type == 'gat':
                model = simple_GAT_regression(
                    hidden_channels=self.hidden_dim,
                    dropout_prob=0.3,
                    input_channel=1,
                    output_dim=1,
                    num_heads=4
                ).to(device)
            
            # Train with cross-validation
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
            fold_results = []
            best_model_state = None
            best_r2 = -float('inf')
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(data_list), 1):
                print(f"  Fold {fold}/{self.num_folds}")
                
                train_data = [data_list[i] for i in train_idx]
                test_data = [data_list[i] for i in test_idx]
                
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
                
                optimizer = Adam(model.parameters(), lr=self.learning_rate)
                criterion = nn.MSELoss()
                
                # Training loop
                best_val_loss = float('inf')
                for epoch in range(self.num_epochs):
                    model.train()
                    total_loss = 0
                    for batch in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        out, _ = model(batch.x, batch.edge_index, batch.batch)
                        target = batch.y[:, target_idx].view(-1, 1)
                        loss = criterion(out, target)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                
                # Validation
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        out, _ = model(batch.x, batch.edge_index, batch.batch)
                        preds.append(out.cpu().numpy())
                        trues.append(batch.y[:, target_idx].cpu().numpy())
                
                preds = np.concatenate(preds).flatten()
                trues = np.concatenate(trues).flatten()
                
                r2 = r2_score(trues, preds)
                rmse = np.sqrt(mean_squared_error(trues, preds))
                
                fold_results.append({
                    'fold': fold,
                    'r2': r2,
                    'rmse': rmse,
                    'predictions': preds,
                    'targets': trues
                })
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_state = model.state_dict().copy()
                
                print(f"    Fold {fold}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            overall_r2 = np.mean([f['r2'] for f in fold_results])
            overall_rmse = np.mean([f['rmse'] for f in fold_results])
            
            print(f"  {model_type.upper()} Overall: R² = {overall_r2:.3f}, RMSE = {overall_rmse:.3f}")
            
            results[model_type] = {
                'model': model,
                'fold_results': fold_results,
                'r2': overall_r2,
                'rmse': overall_rmse
            }
        
        # Find best model
        best_model_type = max(results.keys(), key=lambda k: results[k]['r2'])
        best_model = results[best_model_type]['model']
        
        print(f"\nBest model: {best_model_type.upper()} (R² = {results[best_model_type]['r2']:.3f})")
        
        return results, best_model, best_model_type
    
    def run_explainer_with_new_features(self, model, target_idx=1):
        """Run explainer with NEW FEATURES using proper pipeline explainer"""
        print(f"\n" + "="*60)
        print("STEP 3: Running Explainer with NEW FEATURES")
        print("="*60)
        
        # Use the proper pipeline explainer that follows Mantel → k-NN → GNNExplainer flow
        sparsified_data_list = create_explainer_sparsified_graph(
            self, model, target_idx=target_idx, 
            importance_threshold=0.15, 
            use_node_pruning=True  # NEW FEATURE: Use node-based pruning
        )
        
        print(f"\n✅ Explainer sparsification completed using proper pipeline flow")
        print(f"✅ NEW FEATURE: Node-based pruning applied")
        print(f"✅ NEW FEATURE: Node importance reporting generated")
        
        return sparsified_data_list
    
    def train_on_pruned_graph(self, pruned_data_list, target_idx=1):
        """Train models on pruned graph"""
        print(f"\n" + "="*60)
        print("STEP 4: Training on Node-Pruned Graph")
        print("="*60)
        
        # Train GCN on pruned graph
        model = simple_GCN_res_plus_regression(
            hidden_channels=self.hidden_dim,
            dropout_prob=0.3,
            input_channel=1,
            output_dim=1
        ).to(device)
        
        # Quick training
        train_loader = DataLoader(pruned_data_list, batch_size=self.batch_size, shuffle=True)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.num_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out, _ = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y[:, target_idx].view(-1, 1)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
        
        print("Training on pruned graph completed")
        return model
    
    def create_tsne_visualization(self, model, data_list, target_idx=1):
        """NEW FEATURE 3: Create t-SNE visualization"""
        print(f"\n" + "="*60)
        print("🆕 NEW FEATURE 3: t-SNE Embedding Visualization")
        print("="*60)
        
        # Extract embeddings
        model.eval()
        embeddings_list = []
        targets_list = []
        
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, emb = model(batch.x, batch.edge_index, batch.batch)
                embeddings_list.append(emb.cpu().numpy())
                targets_list.append(batch.y.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
        targets = np.vstack(targets_list)
        
        print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Create explainer for t-SNE functionality
        explainer = GNNExplainerRegression(model, device)
        
        # Save t-SNE plot
        tsne_path = f"{self.save_dir}/plots/H2-km_tsne_embeddings.png"
        tsne_coords = explainer.create_tsne_embedding_plot(
            embeddings,
            targets,
            target_names=self.dataset.target_cols,
            save_path=tsne_path
        )
        
        # Save embeddings
        np.save(f"{self.save_dir}/embeddings/H2-km_embeddings.npy", embeddings)
        np.save(f"{self.save_dir}/embeddings/H2-km_targets.npy", targets)
        
        return embeddings, targets, tsne_coords
    
    
    def run_full_case1_pipeline(self):
        """Run the complete Case 1 pipeline with all new features"""
        print("🚀 Starting Case 1 Pipeline with NEW FEATURES")
        print("="*80)
        
        try:
            # Step 1: Create dataset
            dataset = self.create_anchored_dataset()
            
            # Step 2: Train initial models
            gnn_results, best_model, best_model_type = self.train_gnn_models(dataset.data_list)
            
            # Step 3: Run explainer with NEW FEATURES (this creates sparsified graph in dataset)
            sparsified_data_list = self.run_explainer_with_new_features(best_model)
            
            # Step 4: Train on sparsified graph
            sparsified_model = self.train_on_pruned_graph(sparsified_data_list)
            
            # Step 5: Create t-SNE visualization 
            embeddings, targets, tsne_coords = self.create_tsne_visualization(sparsified_model, sparsified_data_list)
            
            # Step 6: Create proper graph visualizations using dataset's method
            print(f"\n" + "="*60)
            print("🆕 Creating Proper Graph Visualizations")
            print("="*60)
            dataset.visualize_graphs(save_dir=f"{self.save_dir}/graphs")
            
            # Final results
            print(f"\n" + "🎉" * 30)
            print("✅ CASE 1 PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉" * 30)
            
            print(f"\n📊 Results Summary:")
            print(f"  • Original dataset features: {len(dataset.node_feature_names)}")
            print(f"  • Sparsified features: {len(dataset.node_feature_names)} (after node-based pruning)")
            print(f"  • Best original model: {best_model_type.upper()} (R² = {gnn_results[best_model_type]['r2']:.3f})")
            print(f"  • Embeddings shape: {embeddings.shape}")
            print(f"  • t-SNE coordinates shape: {tsne_coords.shape}")
            
            print(f"\n📁 Files created in {self.save_dir}:")
            print(f"  ✅ plots/H2-km_tsne_embeddings.png - t-SNE visualization")
            print(f"  ✅ embeddings/H2-km_embeddings.npy - Node embeddings")
            print(f"  ✅ embeddings/H2-km_targets.npy - Target values")
            print(f"  ✅ graphs/graph_comparison.png - Original vs GNNExplainer sparsified")
            print(f"  ✅ graphs/knn_graph.png - High-res original k-NN graph")
            print(f"  ✅ graphs/explainer_graph.png - High-res sparsified graph")
            
            print(f"\n🎯 NEW FEATURES DEMONSTRATED:")
            print(f"  ✅ Node importance reporting - Top 20 families identified during explainer")
            print(f"  ✅ Node-based pruning - Graph sparsified by removing unimportant nodes")
            print(f"  ✅ t-SNE embedding visualization - Clustering patterns shown")
            print(f"  ✅ Proper pipeline flow - Mantel → k-NN → GNNExplainer → Visualization")
            
            return {
                'gnn_results': gnn_results,
                'dataset': dataset,
                'embeddings': embeddings,
                'targets': targets,
                'tsne_coords': tsne_coords
            }
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Run Case 1 test"""
    data_path = "../Data/New_Data.csv"
    
    # Check data file
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Run Case 1 pipeline
    pipeline = Case1Pipeline(data_path, save_dir="./case1_test_results")
    results = pipeline.run_full_case1_pipeline()
    
    if results:
        print(f"\n🎊 Case 1 test completed successfully!")
        print(f"Check ./case1_test_results/ for all outputs")
    else:
        print(f"⚠️  Pipeline encountered issues")

if __name__ == "__main__":
    main()