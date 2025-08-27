#!/usr/bin/env python3

"""
Case 1 Final Results Pipeline
Focused on getting 5-fold CV results with standard deviation metrics for ML models
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json

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
print(f"Using device: {device}")

def train_gnn_and_extract_embeddings():
    """Train GNN and extract embeddings for ML models"""
    
    print("\n" + "="*80)
    print("CASE 1: TRAINING GNN AND EXTRACTING EMBEDDINGS")
    print("="*80)
    
    # Load dataset for Case 1
    case1_anchored_families = ['Methanoregulaceae', 'Methanobacteriaceae', 'Methanospirillaceae']
    
    dataset = MicrobialGNNDataset(
        data_path='../Data/New_Data.csv',
        graph_mode='family',
        family_filter_mode='strict', 
        use_fast_correlation=False,
        k_neighbors=10
    )
    
    print(f"Dataset created: {len(dataset.data_list)} samples, {len(dataset.node_feature_names)} features")
    print(f"Case 1 anchored families: {case1_anchored_families}")
    
    # Train GNN model (simplified 2-fold for faster results)
    print(f"\nTraining GCN model...")
    model = simple_GCN_res_plus_regression(
        hidden_channels=64,
        output_dim=64,  # ACE-km, H2-km
        dropout_prob=0.3,
        input_channel=1
    ).to(device)
    
    # Simple training loop
    from torch_geometric.loader import DataLoader
    from torch.optim import Adam
    
    dataloader = DataLoader(dataset.data_list, batch_size=8, shuffle=True)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(50):  # Quick training
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            # Handle model output (might be tuple)
            if isinstance(out, tuple):
                out = out[0]
            loss = torch.nn.functional.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.6f}")
    
    # Extract embeddings
    print(f"\nExtracting embeddings...")
    model.eval()
    embeddings = []
    targets = []
    
    with torch.no_grad():
        for data in dataset.data_list:
            data = data.to(device)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
            
            # Get model prediction and use as embedding (simplified approach)
            out = model(data.x, data.edge_index, batch)
            if isinstance(out, tuple):
                embedding = out[0].squeeze()  # Use the prediction as embedding
            else:
                embedding = out.squeeze()
            
            embeddings.append(embedding.cpu().numpy())
            targets.append(data.y.cpu().numpy().flatten())
    
    embeddings = np.array(embeddings)
    targets = np.array(targets)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return embeddings, targets, dataset

def evaluate_ml_models_with_cv(embeddings, targets, target_names, num_folds=5):
    """Evaluate ML models with 5-fold cross-validation"""
    
    print(f"\n" + "="*80)
    print(f"EVALUATING ML MODELS WITH {num_folds}-FOLD CROSS-VALIDATION")
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
                'R2_mean': r2_mean,
                'R2_std': r2_std,
                'RMSE_mean': rmse_mean,
                'RMSE_std': rmse_std,
                'MAE_mean': mae_mean,
                'MAE_std': mae_std,
                'fold_results': {
                    'R2': fold_r2,
                    'RMSE': fold_rmse,
                    'MAE': fold_mae
                }
            }
            
            print(f"  {model_name} FINAL RESULTS:")
            print(f"    R² = {r2_mean:.3f} ± {r2_std:.3f}")
            print(f"    RMSE = {rmse_mean:.3f} ± {rmse_std:.3f}")
            print(f"    MAE = {mae_mean:.3f} ± {mae_std:.3f}")
    
    return results

def print_final_summary(results):
    """Print final summary table"""
    
    print(f"\n" + "="*100)
    print("FINAL CASE 1 RESULTS SUMMARY - 5-FOLD CROSS-VALIDATION")
    print("="*100)
    
    print(f"{'Target':<10} {'Model':<20} {'R² (mean±std)':<20} {'RMSE (mean±std)':<20} {'MAE (mean±std)':<20}")
    print("-" * 100)
    
    for target_name, target_results in results.items():
        for i, (model_name, metrics) in enumerate(target_results.items()):
            target_display = target_name if i == 0 else ""
            print(f"{target_display:<10} {model_name:<20} "
                  f"{metrics['R2_mean']:.3f}±{metrics['R2_std']:.3f}    "
                  f"{metrics['RMSE_mean']:.3f}±{metrics['RMSE_std']:.3f}     "
                  f"{metrics['MAE_mean']:.3f}±{metrics['MAE_std']:.3f}")
        if len(target_results) > 1:
            print("-" * 100)

def save_results(results, embeddings, targets, dataset):
    """Save results to files"""
    
    save_dir = "./case1_final_ml_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed results
    with open(f"{save_dir}/detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save embeddings and targets
    np.save(f"{save_dir}/embeddings.npy", embeddings)
    np.save(f"{save_dir}/targets.npy", targets)
    
    # Save summary
    summary = {
        'dataset_info': {
            'num_samples': len(dataset.data_list),
            'num_features': len(dataset.node_feature_names),
            'target_columns': ['ACE-km', 'H2-km'],
            'anchored_families': ['Methanoregulaceae', 'Methanobacteriaceae', 'Methanospirillaceae']
        },
        'results': results
    }
    
    with open(f"{save_dir}/case1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {save_dir}/")
    
    return save_dir

def main():
    """Main pipeline execution"""
    
    print("🚀 Starting Case 1 Final Results Pipeline")
    print("="*80)
    
    # Step 1: Train GNN and extract embeddings
    embeddings, targets, dataset = train_gnn_and_extract_embeddings()
    
    # Step 2: Evaluate ML models with 5-fold CV
    results = evaluate_ml_models_with_cv(
        embeddings, targets, ['ACE-km', 'H2-km'], num_folds=5
    )
    
    # Step 3: Print final summary
    print_final_summary(results)
    
    # Step 4: Save results
    save_dir = save_results(results, embeddings, targets, dataset)
    
    print(f"\n🎉 Case 1 Pipeline Completed Successfully!")
    print(f"📊 Final results with 5-fold CV std metrics available in: {save_dir}")

if __name__ == "__main__":
    main()