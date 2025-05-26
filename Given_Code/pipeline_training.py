import torch
import numpy as np
import os
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from utils_regression import plot_training_curves, plot_prediction_scatter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(pipeline, target_idx=None, data_list=None):
    """
    Train a GNN model for regression with comprehensive diagnostics
    
    Args:
        pipeline: RegressionPipeline instance
        target_idx: Index of the target variable to predict (if None, predict all targets)
        data_list: List of graph data objects to use (if None, use pipeline.dataset.data_list)
        
    Returns:
        Dictionary with training results
    """
    if data_list is None:
        data_list = pipeline.dataset.data_list
    
    # Determine how many targets to predict
    if target_idx is not None:
        target_name = pipeline.target_names[target_idx]
        num_targets = 1
        print(f"\n{'='*50}")
        print(f"Training model for target: {target_name}")
        print(f"{'='*50}")
    else:
        num_targets = len(pipeline.target_names)
        target_idx = list(range(num_targets))
        print(f"\n{'='*50}")
        print(f"Training model for all {num_targets} targets")
        print(f"{'='*50}")
    
    # DIAGNOSTIC: Analyze target distribution
    print(f"\n📊 TARGET ANALYSIS:")
    all_targets = []
    for data in data_list:
        if target_idx is not None and not isinstance(target_idx, list):
            target_val = data.y[:, target_idx].item()
        else:
            target_val = data.y.numpy().flatten()
        all_targets.append(target_val)
    
    all_targets = np.array(all_targets)
    if all_targets.ndim == 1:
        print(f"Target range: {all_targets.min():.3f} to {all_targets.max():.3f}")
        print(f"Target mean: {all_targets.mean():.3f}, std: {all_targets.std():.3f}")
        print(f"Target variance: {all_targets.var():.3f}")
    
    # Setup k-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=pipeline.num_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # Define loss function based on uncertainty estimation
    if pipeline.estimate_uncertainty:
        criterion = pipeline.GaussianNLLLoss()
    else:
        criterion = nn.MSELoss()
    
    # Iterate through folds
    for fold, (train_index, test_index) in enumerate(kf.split(data_list)):
        fold_num = fold + 1
        print(f"\n🔄 Fold {fold_num}/{pipeline.num_folds}: Train on {len(train_index)} samples, Test on {len(test_index)} samples")
        
        # Split into train and test sets
        train_dataset = [data_list[i] for i in train_index]
        test_dataset = [data_list[i] for i in test_index]
        
        # DIAGNOSTIC: Analyze train/test split
        train_targets = []
        test_targets = []
        for i in train_index:
            if target_idx is not None and not isinstance(target_idx, list):
                train_targets.append(data_list[i].y[:, target_idx].item())
            else:
                train_targets.append(data_list[i].y.numpy().flatten())
        for i in test_index:
            if target_idx is not None and not isinstance(target_idx, list):
                test_targets.append(data_list[i].y[:, target_idx].item())
            else:
                test_targets.append(data_list[i].y.numpy().flatten())
        
        train_targets = np.array(train_targets)
        test_targets = np.array(test_targets)
        
        if train_targets.ndim == 1:
            print(f"📈 Train targets - mean: {train_targets.mean():.3f}, std: {train_targets.std():.3f}")
            print(f"📉 Test targets - mean: {test_targets.mean():.3f}, std: {test_targets.std():.3f}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=pipeline.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=pipeline.batch_size, shuffle=False)
        
        # Initialize model
        model = pipeline.create_model(pipeline.model_type, num_targets=num_targets)
        
        # DIAGNOSTIC: Model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🏗️  Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Setup optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=pipeline.learning_rate, weight_decay=pipeline.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)
        
        # Training loop with diagnostics
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        # Diagnostic tracking
        gradient_norms = []
        prediction_variances = []
        learning_rates = []
        
        for epoch in range(1, pipeline.num_epochs+1):
            # Training step
            model.train()
            total_loss = 0
            epoch_predictions = []
            epoch_targets = []
            
            for batch_data in train_loader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                
                # Handle different output formats based on uncertainty estimation
                if pipeline.estimate_uncertainty and isinstance(out, tuple) and len(out) >= 2:
                    pred, uncertainty = out[0], out[1]
                    # Extract specific target if training for single target
                    if target_idx is not None and not isinstance(target_idx, list):
                        target = batch_data.y[:, target_idx].view(-1, 1)
                    else:
                        target = batch_data.y.view(-1, num_targets)
                    
                    # Use GaussianNLLLoss for uncertainty-aware training
                    loss = criterion(pred, target, uncertainty)
                else:
                    # Standard MSE loss
                    if isinstance(out, tuple):
                        pred = out[0]
                    else:
                        pred = out
                    
                    # Extract specific target if training for single target
                    if target_idx is not None and not isinstance(target_idx, list):
                        target = batch_data.y[:, target_idx].view(-1, 1)
                    else:
                        target = batch_data.y.view(-1, num_targets)
                    loss = criterion(pred, target)
                
                # Store predictions and targets for variance analysis
                epoch_predictions.append(pred.detach().cpu().numpy())
                epoch_targets.append(target.detach().cpu().numpy())
                
                # Backward pass and optimization
                loss.backward()
                
                # DIAGNOSTIC: Calculate gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                total_loss += loss.item() * batch_data.num_graphs
            
            avg_train_loss = total_loss / len(train_dataset)
            train_losses.append(avg_train_loss)
            
            # DIAGNOSTIC: Calculate prediction variance
            epoch_predictions = np.vstack(epoch_predictions)
            epoch_targets = np.vstack(epoch_targets)
            pred_variance = np.var(epoch_predictions)
            target_variance = np.var(epoch_targets)
            prediction_variances.append(pred_variance)
            
            # Store current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Evaluation step
            model.eval()
            total_val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    batch_data = batch_data.to(device)
                    
                    # Forward pass
                    out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    
                    # Calculate validation loss
                    if pipeline.estimate_uncertainty and isinstance(out, tuple) and len(out) >= 2:
                        pred, uncertainty = out[0], out[1]
                        # Extract specific target if training for single target
                        if target_idx is not None and not isinstance(target_idx, list):
                            target = batch_data.y[:, target_idx].view(-1, 1)
                        else:
                            target = batch_data.y.view(-1, num_targets)
                        val_loss = criterion(pred, target, uncertainty)
                    else:
                        if isinstance(out, tuple):
                            pred = out[0]
                        else:
                            pred = out
                        
                        # Extract specific target if training for single target
                        if target_idx is not None and not isinstance(target_idx, list):
                            target = batch_data.y[:, target_idx].view(-1, 1)
                        else:
                            target = batch_data.y.view(-1, num_targets)
                        val_loss = nn.MSELoss()(pred, target)
                    
                    val_predictions.append(pred.cpu().numpy())
                    val_targets.append(target.cpu().numpy())
                    total_val_loss += val_loss.item() * batch_data.num_graphs
            
            avg_val_loss = total_val_loss / len(test_dataset)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # DIAGNOSTIC: Print detailed progress
            if epoch % 10 == 0 or epoch == 1 or epoch == pipeline.num_epochs:
                val_predictions = np.vstack(val_predictions)
                val_targets = np.vstack(val_targets)
                val_pred_variance = np.var(val_predictions)
                val_target_variance = np.var(val_targets)
                
                print(f"📊 Epoch {epoch:03d}:")
                print(f"   Loss: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
                print(f"   Pred Variance: Train={pred_variance:.6f}, Val={val_pred_variance:.6f}")
                print(f"   Target Variance: Train={target_variance:.6f}, Val={val_target_variance:.6f}")
                print(f"   Grad Norm: {total_norm:.6f}, LR: {current_lr:.6f}")
                
                # Check for constant predictions
                if pred_variance < 1e-6:
                    print(f"⚠️  WARNING: Predictions have very low variance ({pred_variance:.8f}) - model may be stuck!")
                if total_norm < 1e-6:
                    print(f"⚠️  WARNING: Gradients are very small ({total_norm:.8f}) - learning may have stopped!")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= pipeline.patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
        
        # Load best model for evaluation
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_targets = []
        all_uncertainties = [] if pipeline.estimate_uncertainty else None
        
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                
                # Forward pass
                out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                
                # Extract predictions and targets
                if pipeline.estimate_uncertainty and isinstance(out, tuple) and len(out) >= 2:
                    pred, uncertainty = out[0], out[1]
                    all_uncertainties.append(uncertainty.cpu().numpy())
                else:
                    if isinstance(out, tuple):
                        pred = out[0]
                    else:
                        pred = out
                
                # Extract specific target if training for single target
                if target_idx is not None and not isinstance(target_idx, list):
                    target = batch_data.y[:, target_idx].view(-1, 1)
                else:
                    target = batch_data.y.view(-1, num_targets)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate results
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        if pipeline.estimate_uncertainty:
            all_uncertainties = np.vstack(all_uncertainties)
        
        # Calculate metrics for each target (using original scale values)
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
                t_name = pipeline.target_names[target_idx]
            else:
                t_name = pipeline.target_names[i]
            
            # Store metrics
            metrics_dict = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            target_metrics.append({
                'target_idx': i if isinstance(target_idx, list) else target_idx,
                'target_name': t_name,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            })
            
            print(f"Target {t_name}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
            
            # Create prediction scatter plot
            if target_idx is not None and not isinstance(target_idx, list):
                plot_path = f"{pipeline.save_dir}/plots/{pipeline.model_type}_{t_name}_fold{fold_num}_pred.png"
            else:
                plot_path = f"{pipeline.save_dir}/plots/{pipeline.model_type}_target{i}_fold{fold_num}_pred.png"
                
            plot_prediction_scatter(
                target_true, 
                target_preds, 
                metrics_dict, 
                plot_path, 
                target_name=t_name
            )
        
        # Save model
        if target_idx is not None and not isinstance(target_idx, list):
            model_path = f"{pipeline.save_dir}/models/{pipeline.model_type}_{pipeline.target_names[target_idx]}_fold{fold_num}.pt"
        else:
            model_path = f"{pipeline.save_dir}/models/{pipeline.model_type}_all_targets_fold{fold_num}.pt"
        
        torch.save(model.state_dict(), model_path)
        
        # Create plot for training curves
        if target_idx is not None and not isinstance(target_idx, list):
            curves_path = f"{pipeline.save_dir}/plots/{pipeline.model_type}_{pipeline.target_names[target_idx]}_fold{fold_num}_loss.png"
        else:
            curves_path = f"{pipeline.save_dir}/plots/{pipeline.model_type}_all_targets_fold{fold_num}_loss.png"
            
        plot_training_curves(
            train_losses,
            val_losses,
            fold_num,
            curves_path
        )
        
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
        
        # Generate GNN explanations
        if target_idx is not None and not isinstance(target_idx, list):
            _generate_explanations(pipeline, model, test_dataset, pipeline.target_names[target_idx], fold_num)
    
    # Plot overall results
    _plot_overall_results(pipeline, fold_results, target_idx)
    
    return fold_results


def _generate_explanations(pipeline, model, test_dataset, target_name, fold_num):
    """Generate and save GNN explanations"""
    from explainer_regression import GNNExplainerRegression
    from utils_regression import visualize_explanation
    
    print(f"\nGenerating explanations for {target_name} (fold {fold_num})...")
    
    # Initialize explainer
    explainer = GNNExplainerRegression(model, device)
    
    # Create output directory
    os.makedirs(f"{pipeline.save_dir}/explanations/{target_name}", exist_ok=True)
    
    # Generate explanations for a subset of test samples
    num_explain = min(5, len(test_dataset))  # Limit to 5 samples
    
    for i in range(num_explain):
        # Get sample data
        data = test_dataset[i]
        
        # Generate explanation
        edge_importance_matrix, explanation = explainer.explain_graph(
            data, 
            node_names=pipeline.dataset.node_feature_names,
            save_path=f"{pipeline.save_dir}/explanations/{target_name}/sample_{i}_fold_{fold_num}_edge_importance.csv",
            target_idx=0  # For single target models
        )
        
        # Visualize the explanation
        visualize_explanation(
            data, 
            edge_importance_matrix, 
            pipeline.dataset.node_feature_names,
            i,
            f"{pipeline.save_dir}/explanations/{target_name}/sample_{i}_fold_{fold_num}_explanation.png",
            target_name
        )
        
        # Save the text explanation
        with open(f"{pipeline.save_dir}/explanations/{target_name}/sample_{i}_fold_{fold_num}_explanation.txt", 'w') as f:
            f.write(explanation)


def _plot_overall_results(pipeline, fold_results, target_idx=None):
    """Plot overall results across all folds"""
    import matplotlib.pyplot as plt
    
    # Determine how many targets we're working with
    if target_idx is not None and not isinstance(target_idx, list):
        num_targets = 1
        target_names = [pipeline.target_names[target_idx]]
    else:
        num_targets = len(pipeline.target_names)
        target_names = pipeline.target_names
    
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
        plt.savefig(f"{pipeline.save_dir}/plots/{pipeline.model_type}_{pipeline.target_names[target_idx]}_overall.png", dpi=300)
    else:
        plt.savefig(f"{pipeline.save_dir}/plots/{pipeline.model_type}_all_targets_overall.png", dpi=300)
    
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
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_df)
    
    if target_idx is not None and not isinstance(target_idx, list):
        metrics_df.to_csv(f"{pipeline.save_dir}/plots/{pipeline.model_type}_{pipeline.target_names[target_idx]}_metrics.csv", index=False)
    else:
        metrics_df.to_csv(f"{pipeline.save_dir}/plots/{pipeline.model_type}_all_targets_metrics.csv", index=False)
        
    # Print overall results
    print("\nOverall Results:")
    print(metrics_df) 