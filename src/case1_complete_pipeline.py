#!/usr/bin/env python3

"""
Complete Case 1 Pipeline
Combines graph visualizations (3 new features) + final ML results with 5-fold CV std metrics
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
from torch_geometric.loader import DataLoader
from torch.optim import Adam

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from datasets.dataset_regression_working import MicrobialGNNDataset
from explainers.explainer_regression_working import GNNExplainerRegression  
from explainers.pipeline_explainer_working import create_explainer_sparsified_graph
from models.GNNmodelsRegression import simple_GCN_res_plus_regression

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Case1CompletePipeline:
    def __init__(self, save_dir='./case1_complete_pipeline_results'):
        self.save_dir = save_dir
        self.dataset = None
        self.best_model = None
        self.graph_mode = 'family'
        
        # Create save directories
        os.makedirs(f"{save_dir}/graphs", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/embeddings", exist_ok=True)
        os.makedirs(f"{save_dir}/ml_results", exist_ok=True)
        
        print(f"Using device: {device}")
        print(f"Case 1 Complete Pipeline initialized - Results will be saved to: {save_dir}")
        
    def create_dataset(self):
        """Create Case 1 dataset with anchored features"""
        print("\n" + "="*80)
        print("STEP 1: Creating Case 1 Dataset with Anchored Features")
        print("="*80)
        
        case1_anchored_families = ['Methanoregulaceae', 'Methanobacteriaceae', 'Methanospirillaceae']
        print(f"Case 1 anchored families: {case1_anchored_families}")
        
        self.dataset = MicrobialGNNDataset(
            data_path='../Data/New_Data.csv',
            graph_mode='family',
            family_filter_mode='strict', 
            use_fast_correlation=False,
            k_neighbors=10
        )
        
        print(f"Dataset created: {len(self.dataset.data_list)} samples, {len(self.dataset.node_feature_names)} features")
        print(f"Target variables: ['ACE-km', 'H2-km']")
        print(f"Number of features: {len(self.dataset.node_feature_names)}")
        print(f"Original k-NN graph stored: {self.dataset.data_list[0].edge_index.shape[1]} edges")
        
        return self.dataset
    
    def train_gnn_model(self, target_idx=1):
        """Train GNN model for H2-km (Case 1)"""
        print(f"\n" + "="*80)
        print(f"STEP 2: Training GNN Models for H2-km")
        print("="*80)
        
        # Train GCN model
        print(f"\nTraining GCN...")
        model = simple_GCN_res_plus_regression(
            hidden_channels=64,
            output_dim=2,  # ACE-km, H2-km
            dropout_prob=0.3,
            input_channel=1
        ).to(device)
        
        # Simple training with 2-fold CV for speed
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        fold_results = []
        best_model_state = None
        best_r2 = -float('inf')
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset.data_list)):
            print(f"  Fold {fold+1}/2")
            
            # Create data splits
            train_data = [self.dataset.data_list[i] for i in train_idx]
            val_data = [self.dataset.data_list[i] for i in val_idx]
            
            train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
            
            # Initialize model and optimizer
            model = simple_GCN_res_plus_regression(
                hidden_channels=64, output_dim=2, dropout_prob=0.3, input_channel=1
            ).to(device)
            optimizer = Adam(model.parameters(), lr=0.01)
            
            # Training loop
            model.train()
            for epoch in range(100):  # Quick training
                total_loss = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = torch.nn.functional.mse_loss(out, batch.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch == 99:  # Final epoch
                    print(f"    Final training loss: {total_loss/len(train_loader):.6f}")
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    if isinstance(out, tuple):
                        out = out[0]
                    val_preds.append(out.cpu())
                    val_targets.append(batch.y.cpu())
            
            val_preds = torch.cat(val_preds, dim=0)[:, target_idx]  # H2-km only
            val_targets = torch.cat(val_targets, dim=0)[:, target_idx]
            
            # Calculate metrics
            r2 = r2_score(val_targets.numpy(), val_preds.numpy())
            rmse = np.sqrt(mean_squared_error(val_targets.numpy(), val_preds.numpy()))
            
            fold_results.append({'R2': r2, 'RMSE': rmse})
            print(f"    Fold {fold+1}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
            
            # Save best model
            if r2 > best_r2:
                best_r2 = r2
                best_model_state = model.state_dict()
        
        # Load best model
        model.load_state_dict(best_model_state)
        self.best_model = model
        
        # Print overall results
        avg_r2 = np.mean([r['R2'] for r in fold_results])
        avg_rmse = np.mean([r['RMSE'] for r in fold_results])
        print(f"  GCN Overall: R² = {avg_r2:.3f}, RMSE = {avg_rmse:.3f}")
        
        return model
    
    def create_graph_visualizations_with_features(self, model, target_idx=1):
        """Create all 3 new features: node importance, node-based pruning, graph visualizations"""
        print(f"\n" + "="*80)
        print("STEP 3: Creating Graph Visualizations with NEW FEATURES")
        print("="*80)
        
        # Create explainer
        explainer = GNNExplainerRegression(model, device)
        
        # Generate explanations from multiple samples
        print("Generating explanations...")
        num_samples = min(5, len(self.dataset.data_list))
        combined_importance = torch.zeros(
            (len(self.dataset.node_feature_names), len(self.dataset.node_feature_names)), 
            device=device
        )
        
        for i in range(num_samples):
            data = self.dataset.data_list[i]
            edge_importance_matrix, _ = explainer.explain_graph(
                data, 
                node_names=self.dataset.node_feature_names,
                target_idx=target_idx
            )
            combined_importance += edge_importance_matrix
        
        combined_importance /= num_samples
        
        print(f"\n🆕 NEW FEATURE 1: Node Importance Reporting")
        print("-" * 50)
        
        # NEW FEATURE 1: Node importance reporting
        node_importance, sorted_indices = explainer.get_node_importance(
            combined_importance, 
            self.dataset.node_feature_names
        )
        
        print(f"\n🆕 NEW FEATURE 2: Node-Based Pruning Workflow")
        print("-" * 50)
        
        # NEW FEATURE 2: Node-based pruning workflow
        template_data = self.dataset.data_list[0]
        pruned_data, kept_nodes, pruned_node_names = explainer.create_node_pruned_graph(
            template_data,
            combined_importance,
            self.dataset.node_feature_names,
            importance_threshold=0.2,
            min_nodes=10
        )
        
        # Extract actual edge weights for visualization
        node_based_edge_weights = []
        for i in range(pruned_data.edge_index.shape[1]):
            u_new, v_new = pruned_data.edge_index[0, i].item(), pruned_data.edge_index[1, i].item()
            u_orig, v_orig = kept_nodes[u_new], kept_nodes[v_new]
            edge_weight = abs(combined_importance[u_orig, v_orig].item())
            node_based_edge_weights.append(edge_weight)
        
        # Step 2: Edge sparsification on the pruned graph
        edge_sparsified_data, edge_weights, edge_types = explainer.create_edge_sparsified_graph(
            pruned_data,
            combined_importance,
            pruned_node_names,
            edge_threshold=0.15,
            kept_nodes=kept_nodes
        )
        
        print(f"\n🆕 NEW FEATURE 3: Comprehensive Graph Visualizations")
        print("-" * 50)
        
        # Store graph data for visualization
        original_graph_data = {
            'edge_index': self.dataset.data_list[0].edge_index,
            'edge_weight': torch.ones(self.dataset.data_list[0].edge_index.shape[1]),
            'edge_type': torch.ones(self.dataset.data_list[0].edge_index.shape[1], dtype=torch.long)
        }
        
        node_based_graph_data = {
            'edge_index': pruned_data.edge_index.clone(),
            'edge_weight': torch.tensor(node_based_edge_weights, dtype=torch.float32),
            'edge_type': torch.ones(pruned_data.edge_index.shape[1], dtype=torch.long),
            'pruning_type': 'node_based',
            'kept_nodes': kept_nodes,
            'pruned_node_names': pruned_node_names
        }
        
        edge_based_graph_data = {
            'edge_index': edge_sparsified_data.edge_index.clone(),
            'edge_weight': edge_weights,
            'edge_type': edge_types,
            'pruning_type': 'node_then_edge',
            'kept_nodes': kept_nodes,
            'pruned_node_names': pruned_node_names
        }
        
        # Create comprehensive graph comparison
        self._create_comprehensive_graph_visualization(
            original_graph_data, node_based_graph_data, edge_based_graph_data, pruned_node_names
        )
        
        return pruned_data, kept_nodes, pruned_node_names
    
    def _create_comprehensive_graph_visualization(self, original_data, node_data, edge_data, pruned_names):
        """Create the 3-panel graph comparison with node names and edge weights"""
        import networkx as nx
        
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        # Graph 1: Original k-NN
        original_edges = original_data['edge_index'].shape[1] // 2
        self._visualize_single_graph(
            original_data['edge_index'], original_data['edge_weight'], original_data['edge_type'],
            axes[0], f"Original k-NN Graph\n({len(self.dataset.node_feature_names)} nodes, {original_edges} edges)",
            node_names=self.dataset.node_feature_names
        )
        
        # Graph 2: Node-Based Pruned (with edge weights)
        node_edges = node_data['edge_index'].shape[1] // 2
        node_retention = len(pruned_names) / len(self.dataset.node_feature_names) * 100
        self._visualize_single_graph(
            node_data['edge_index'], node_data['edge_weight'], node_data['edge_type'],
            axes[1], f"Node-Based Pruned (Step 1)\n({len(pruned_names)} nodes, {node_edges} edges)\n({node_retention:.1f}% nodes retained)",
            node_names=pruned_names, show_edge_weights=True
        )
        
        # Graph 3: Edge-Based Sparsified
        edge_edges = edge_data['edge_index'].shape[1] // 2
        edge_retention = edge_edges / node_edges * 100 if node_edges > 0 else 0
        self._visualize_single_graph(
            edge_data['edge_index'], edge_data['edge_weight'], edge_data['edge_type'],
            axes[2], f"Edge-Based Sparsified (Step 2)\n({len(pruned_names)} nodes, {edge_edges} edges)\n({edge_retention:.1f}% edges retained from Step 1)",
            node_names=pruned_names
        )
        
        plt.tight_layout()
        graph_path = f"{self.save_dir}/graphs/comprehensive_graph_comparison.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive graph comparison saved: {graph_path}")
    
    def _visualize_single_graph(self, edge_index, edge_weight, edge_type, ax, title, node_names=None, show_edge_weights=False):
        """Visualize a single graph with node names and optional edge weights"""
        import networkx as nx
        
        G = nx.Graph()
        num_nodes = edge_index.max().item() + 1 if edge_index.shape[1] > 0 else 0
        
        # Add nodes with names
        if node_names is not None and len(node_names) >= num_nodes:
            for i in range(num_nodes):
                G.add_node(i, name=node_names[i])
        else:
            for i in range(num_nodes):
                G.add_node(i, name=f"Node_{i}")
        
        # Add edges
        edge_list = []
        weights = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u != v:
                weight = edge_weight[i].item() if len(edge_weight) > i else 1.0
                edge_list.append((u, v))
                weights.append(weight)
                G.add_edge(u, v, weight=weight)
        
        if len(edge_list) == 0:
            ax.text(0.5, 0.5, "No edges to display", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            ax.axis('off')
            return
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        # Draw nodes
        node_size = max(10, 300 - len(G.nodes()) // 10)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.7, ax=ax)
        
        # Draw edges
        if weights:
            min_w, max_w = min(weights), max(weights)
            if max_w > min_w:
                norm_weights = [(w - min_w) / (max_w - min_w) * 3 + 0.5 for w in weights]
            else:
                norm_weights = [1.0] * len(weights)
        else:
            norm_weights = [1.0]
        
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=norm_weights, alpha=0.6, edge_color='gray', ax=ax)
        
        # Draw node labels
        if node_names is not None:
            labels = {i: node_names[i][:15] + '...' if len(node_names[i]) > 15 else node_names[i] for i in range(num_nodes)}
        else:
            labels = {i: f"Node_{i}" for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
        
        # Draw edge weights if requested
        if show_edge_weights and len(edge_list) > 0 and len(weights) > 0:
            edge_labels = {(u, v): f"{abs(w):.3f}" for (u, v), w in zip(edge_list, weights)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def create_tsne_visualization(self, model, target_idx=1):
        """NEW FEATURE 3: Create t-SNE visualization"""
        print(f"\n" + "="*80)
        print("🆕 NEW FEATURE 3: t-SNE Embedding Visualization")
        print("="*80)
        
        # Extract embeddings
        model.eval()
        embeddings = []
        targets = []
        
        with torch.no_grad():
            for data in self.dataset.data_list:
                data = data.to(device)
                batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
                out = model(data.x, data.edge_index, batch)
                if isinstance(out, tuple):
                    embedding = out[0].squeeze()
                else:
                    embedding = out.squeeze()
                embeddings.append(embedding.cpu().numpy())
                targets.append(data.y.cpu().numpy().flatten())
        
        embeddings = np.array(embeddings)
        targets = np.array(targets)
        print(f"Extracted embeddings shape: {embeddings.shape}")
        
        # Create t-SNE plot
        from sklearn.manifold import TSNE
        
        print("\nCreating t-SNE plot for embeddings...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_coords = tsne.fit_transform(embeddings)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # ACE-km plot
        scatter1 = ax1.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=targets[:, 0], cmap='viridis', alpha=0.7)
        ax1.set_title('t-SNE Embedding Visualization - ACE-km\nClustering/Data Distribution', fontsize=14)
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter1, ax=ax1, label='ACE-km Value')
        
        # H2-km plot
        scatter2 = ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=targets[:, 1], cmap='viridis', alpha=0.7)
        ax2.set_title('t-SNE Embedding Visualization - H2-km\nClustering/Data Distribution', fontsize=14)
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter2, ax=ax2, label='H2-km Value')
        
        plt.tight_layout()
        tsne_path = f"{self.save_dir}/plots/H2-km_tsne_embeddings.png"
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE plot saved to: {tsne_path}")
        
        return embeddings, targets, tsne_coords
    
    def evaluate_ml_models_with_cv(self, embeddings, targets, num_folds=5):
        """Evaluate ML models with 5-fold cross-validation"""
        print(f"\n" + "="*80)
        print(f"STEP 4: EVALUATING ML MODELS WITH {num_folds}-FOLD CROSS-VALIDATION")
        print("="*80)
        
        # Initialize models
        models = {
            'LinearSVR': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', LinearSVR(random_state=42, max_iter=2000))
            ]),
            'ExtraTreesRegressor': Pipeline([
                ('scaler', StandardScaler()),
                ('etr', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            ])
        }
        
        target_names = ['ACE-km', 'H2-km']
        results = {}
        
        # Evaluate each target
        for target_idx, target_name in enumerate(target_names):
            print(f"\n{'-'*50}")
            print(f"TARGET: {target_name}")
            print(f"{'-'*50}")
            
            y = targets[:, target_idx]
            results[target_name] = {}
            
            # 5-fold cross-validation
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            
            for model_name, model in models.items():
                print(f"\nEvaluating {model_name}...")
                
                fold_r2 = []
                fold_rmse = []
                fold_mae = []
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
                    # Split data
                    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    fold_r2.append(r2)
                    fold_rmse.append(rmse)
                    fold_mae.append(mae)
                    
                    print(f"  Fold {fold+1}: R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
                
                # Calculate mean and std
                r2_mean, r2_std = np.mean(fold_r2), np.std(fold_r2)
                rmse_mean, rmse_std = np.mean(fold_rmse), np.std(fold_rmse)
                mae_mean, mae_std = np.mean(fold_mae), np.std(fold_mae)
                
                # Store results
                results[target_name][model_name] = {
                    'R2_mean': r2_mean, 'R2_std': r2_std,
                    'RMSE_mean': rmse_mean, 'RMSE_std': rmse_std,
                    'MAE_mean': mae_mean, 'MAE_std': mae_std,
                    'fold_results': {'R2': fold_r2, 'RMSE': fold_rmse, 'MAE': fold_mae}
                }
                
                print(f"  {model_name} FINAL RESULTS:")
                print(f"    R² = {r2_mean:.3f} ± {r2_std:.3f}")
                print(f"    RMSE = {rmse_mean:.3f} ± {rmse_std:.3f}")
                print(f"    MAE = {mae_mean:.3f} ± {mae_std:.3f}")
        
        return results
    
    def save_all_results(self, results, embeddings, targets, tsne_coords):
        """Save all results to files"""
        # Save ML results
        with open(f"{self.save_dir}/ml_results/detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save embeddings and data
        np.save(f"{self.save_dir}/embeddings/embeddings.npy", embeddings)
        np.save(f"{self.save_dir}/embeddings/targets.npy", targets)
        np.save(f"{self.save_dir}/embeddings/tsne_coords.npy", tsne_coords)
        
        # Save summary
        summary = {
            'pipeline_info': {
                'description': 'Case 1: Complete pipeline with 3 new features + ML results',
                'features': [
                    'Node importance reporting (top 20 families)',
                    'Node-based pruning workflow (2-step: nodes then edges)',
                    't-SNE embedding visualization',
                    'Comprehensive graph visualizations with edge weights',
                    '5-fold cross-validation ML results with std metrics'
                ]
            },
            'dataset_info': {
                'num_samples': len(self.dataset.data_list),
                'num_features': len(self.dataset.node_feature_names),
                'target_columns': ['ACE-km', 'H2-km'],
                'case1_anchored_families': ['Methanoregulaceae', 'Methanobacteriaceae', 'Methanospirillaceae']
            },
            'ml_results': results
        }
        
        with open(f"{self.save_dir}/case1_complete_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ All results saved to: {self.save_dir}/")
        return self.save_dir
    
    def print_final_summary(self, results):
        """Print final comprehensive summary"""
        print(f"\n" + "="*120)
        print("🎉 CASE 1 COMPLETE PIPELINE RESULTS - GRAPHS + ML WITH 5-FOLD CV")
        print("="*120)
        
        print(f"✅ FEATURES IMPLEMENTED:")
        print(f"  🆕 Node importance reporting (top 20 microbial families)")  
        print(f"  🆕 Node-based pruning workflow (2-step: nodes → edges)")
        print(f"  🆕 t-SNE embedding visualization (clustering patterns)")
        print(f"  🆕 Comprehensive graph visualizations (3 panels with edge weights)")
        print(f"  📊 5-fold cross-validation ML results with std metrics")
        
        print(f"\n📊 ML MODEL RESULTS:")
        print(f"{'Target':<10} {'Model':<20} {'R² (mean±std)':<20} {'RMSE (mean±std)':<20} {'MAE (mean±std)':<20}")
        print("-" * 120)
        
        for target_name, target_results in results.items():
            for i, (model_name, metrics) in enumerate(target_results.items()):
                target_display = target_name if i == 0 else ""
                print(f"{target_display:<10} {model_name:<20} "
                      f"{metrics['R2_mean']:.3f}±{metrics['R2_std']:.3f}    "
                      f"{metrics['RMSE_mean']:.3f}±{metrics['RMSE_std']:.3f}     "
                      f"{metrics['MAE_mean']:.3f}±{metrics['MAE_std']:.3f}")
            if len(target_results) > 1:
                print("-" * 120)
        
        print(f"\n📁 FILES CREATED:")
        print(f"  ✅ graphs/comprehensive_graph_comparison.png - 3-panel graph comparison")
        print(f"  ✅ plots/H2-km_tsne_embeddings.png - t-SNE clustering visualization")  
        print(f"  ✅ ml_results/detailed_results.json - Complete ML results with all folds")
        print(f"  ✅ embeddings/embeddings.npy - GNN embeddings")
        print(f"  ✅ embeddings/targets.npy - Target values")
        print(f"  ✅ case1_complete_summary.json - Complete pipeline summary")
    
    def run_complete_pipeline(self):
        """Run the complete Case 1 pipeline"""
        print("🚀 Starting Case 1 COMPLETE Pipeline")
        print("="*80)
        
        # Step 1: Create dataset
        self.create_dataset()
        
        # Step 2: Train GNN model
        model = self.train_gnn_model(target_idx=1)
        
        # Step 3: Create graph visualizations with all 3 new features
        self.create_graph_visualizations_with_features(model, target_idx=1)
        
        # Step 4: Create t-SNE visualization
        embeddings, targets, tsne_coords = self.create_tsne_visualization(model, target_idx=1)
        
        # Step 5: Evaluate ML models with 5-fold CV
        results = self.evaluate_ml_models_with_cv(embeddings, targets, num_folds=5)
        
        # Step 6: Save all results
        self.save_all_results(results, embeddings, targets, tsne_coords)
        
        # Step 7: Print final summary
        self.print_final_summary(results)
        
        print(f"\n🎊 Case 1 COMPLETE PIPELINE completed successfully!")
        print(f"📁 All results available in: {self.save_dir}")

def main():
    pipeline = Case1CompletePipeline()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()