import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import networkx as nx

def plot_training_curves(train_losses, val_losses, fold_num, save_path, title=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Fold {fold_num} Training Curves')
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Training curves saved to {save_path}")

def plot_prediction_scatter(true_values, predictions, metrics_dict, save_path, target_name=None):
    """Plot scatter plot of predictions vs true values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predictions, alpha=0.6, edgecolor='k', facecolor='none')
    
    # Add diagonal line
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add metrics text
    mse = metrics_dict.get('mse', mean_squared_error(true_values, predictions))
    rmse = metrics_dict.get('rmse', np.sqrt(mse))
    r2 = metrics_dict.get('r2', r2_score(true_values, predictions))
    
    title = f'{target_name if target_name else "Target"}\nMSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}'
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Prediction scatter plot saved to {save_path}")

def plot_comparison_results(comparison_results, save_path):
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
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Comparison results plot saved to {save_path}")

def visualize_explanation(data, edge_importance_matrix, node_feature_names, sample_id, save_path, target_name, threshold=0.2):
    """Visualize explanation as a network with edge importance"""
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(node_feature_names):
        G.add_node(i, name=name, value=data.x[i].item())
    
    # Add edges with importance as weights
    edge_importance = edge_importance_matrix.detach().cpu().numpy()
    
    # Normalize edge importance to [0, 1] range
    if edge_importance.max() > 0:
        edge_importance = edge_importance / edge_importance.max()
    
    # Add edges with importance above a threshold
    for i in range(len(node_feature_names)):
        for j in range(i+1, len(node_feature_names)):
            importance = edge_importance[i, j]
            if importance > threshold:
                G.add_edge(i, j, weight=float(importance))
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Create layout
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Get node sizes based on values
    node_size = [1000 * (0.1 + abs(data.x[i].item())) for i in range(len(node_feature_names))]
    
    # Get edge widths based on importance
    edge_width = []
    for u, v, d in G.edges(data=True):
        edge_width.append(d['weight'] * 5)
    
    # Draw the network
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        labels={i: node_feature_names[i] for i in G.nodes()},
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
    
    print(f"Explanation visualization saved to {save_path}")

def save_metrics_to_csv(metrics_df, save_path):
    """Save metrics DataFrame to CSV file"""
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}") 