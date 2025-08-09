"""
Result management utilities for saving, loading, and tracking experiment results.
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def create_results_directory_structure(base_path: str, case_name: str, 
                                     subdirs: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Create standardized directory structure for results.
    
    Args:
        base_path: Base path for results
        case_name: Name of the case (e.g., 'case1', 'case2')
        subdirs: Additional subdirectories to create
        
    Returns:
        dict: Mapping of directory names to paths
    """
    if subdirs is None:
        subdirs = ['embeddings', 'models', 'plots', 'metrics', 'graphs']
    
    case_dir = os.path.join(base_path, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    dir_paths = {'base': case_dir}
    
    for subdir in subdirs:
        subdir_path = os.path.join(case_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        dir_paths[subdir] = subdir_path
    
    return dir_paths


def save_fold_results(results: Dict[str, Any], fold_idx: int, 
                     results_dir: str, prefix: str = '') -> None:
    """
    Save results for a specific fold.
    
    Args:
        results: Dictionary containing fold results
        fold_idx: Fold index
        results_dir: Directory to save results
        prefix: Optional prefix for filenames
    """
    fold_file = f"{prefix}fold_{fold_idx}_results.json" if prefix else f"fold_{fold_idx}_results.json"
    filepath = os.path.join(results_dir, fold_file)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = convert_for_json_serialization(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def convert_for_json_serialization(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_for_json_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json_serialization(item) for item in obj]
    else:
        return obj


def save_embeddings(embeddings: np.ndarray, fold_idx: int, model_name: str, 
                   embeddings_dir: str, target_name: str = '') -> None:
    """
    Save GNN embeddings for a specific fold and model.
    
    Args:
        embeddings: Embedding array
        fold_idx: Fold index
        model_name: Name of the model
        embeddings_dir: Directory to save embeddings
        target_name: Optional target name for multi-target scenarios
    """
    target_suffix = f"_{target_name}" if target_name else ""
    filename = f"{model_name}_fold_{fold_idx}{target_suffix}_embeddings.npy"
    filepath = os.path.join(embeddings_dir, filename)
    np.save(filepath, embeddings)


def save_model_checkpoint(model, fold_idx: int, model_name: str, 
                         models_dir: str, target_name: str = '') -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        fold_idx: Fold index
        model_name: Name of the model
        models_dir: Directory to save models
        target_name: Optional target name
    """
    target_suffix = f"_{target_name}" if target_name else ""
    filename = f"{model_name}_fold_{fold_idx}{target_suffix}_model.pth"
    filepath = os.path.join(models_dir, filename)
    
    import torch
    torch.save(model.state_dict(), filepath)


def aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across folds.
    
    Args:
        fold_results: List of fold result dictionaries
        
    Returns:
        dict: Aggregated results with mean and std
    """
    if not fold_results:
        return {}
    
    aggregated = {}
    
    # Get all unique keys across folds
    all_keys = set()
    for fold_result in fold_results:
        all_keys.update(fold_result.keys())
    
    for key in all_keys:
        values = []
        for fold_result in fold_results:
            if key in fold_result:
                values.append(fold_result[key])
        
        if values:
            if isinstance(values[0], (int, float, np.number)):
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': values,
                    'formatted': f"{np.mean(values):.3f} ± {np.std(values):.3f}"
                }
            else:
                # For non-numeric data, just store the values
                aggregated[key] = {'values': values}
    
    return aggregated


def save_hyperparameter_tracking(param_counts: Dict[str, int], 
                                hyperparams_dir: str, filename: str = 'hyperparameter_frequency.json') -> None:
    """
    Save hyperparameter frequency tracking.
    
    Args:
        param_counts: Dictionary with parameter combinations and their counts
        hyperparams_dir: Directory to save hyperparameter tracking
        filename: Filename for the tracking file
    """
    filepath = os.path.join(hyperparams_dir, filename)
    
    # Convert to serializable format
    serializable_counts = {}
    for param_str, count in param_counts.items():
        serializable_counts[param_str] = int(count)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_counts, f, indent=2)


def save_combined_results_summary(results: Dict[str, Any], output_dir: str, 
                                case_name: str, target_names: List[str]) -> None:
    """
    Save a comprehensive summary of combined results.
    
    Args:
        results: Combined results dictionary
        output_dir: Output directory
        case_name: Case name (e.g., 'case1')
        target_names: List of target names
    """
    # Create summary directory
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save JSON summary
    json_path = os.path.join(summary_dir, f"{case_name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(convert_for_json_serialization(results), f, indent=2)
    
    # Create CSV summary for key metrics
    csv_data = []
    for target_name in target_names:
        if target_name in results:
            target_results = results[target_name]
            for model_name in ['GCN', 'RGGC', 'GAT']:
                if model_name in target_results:
                    model_results = target_results[model_name]
                    row = {
                        'Target': target_name,
                        'Model': model_name,
                        'R2_mean': model_results.get('R2_mean', ''),
                        'R2_std': model_results.get('R2_std', ''),
                        'RMSE_mean': model_results.get('RMSE_mean', ''),
                        'RMSE_std': model_results.get('RMSE_std', ''),
                        'MAE_mean': model_results.get('MAE_mean', ''),
                        'MAE_std': model_results.get('MAE_std', '')
                    }
                    csv_data.append(row)
    
    if csv_data:
        csv_path = os.path.join(summary_dir, f"{case_name}_metrics_summary.csv")
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)


def load_fold_results(results_dir: str, fold_idx: int, prefix: str = '') -> Optional[Dict[str, Any]]:
    """
    Load results for a specific fold.
    
    Args:
        results_dir: Directory containing results
        fold_idx: Fold index
        prefix: Optional prefix for filenames
        
    Returns:
        dict or None: Loaded results or None if file doesn't exist
    """
    fold_file = f"{prefix}fold_{fold_idx}_results.json" if prefix else f"fold_{fold_idx}_results.json"
    filepath = os.path.join(results_dir, fold_file)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def save_graph_metadata(graph_info: Dict[str, Any], graphs_dir: str, 
                       graph_type: str, fold_idx: int) -> None:
    """
    Save graph metadata and statistics.
    
    Args:
        graph_info: Dictionary containing graph information
        graphs_dir: Directory to save graph metadata
        graph_type: Type of graph (e.g., 'knn', 'explainer')
        fold_idx: Fold index
    """
    filename = f"{graph_type}_graph_fold_{fold_idx}_info.json"
    filepath = os.path.join(graphs_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(convert_for_json_serialization(graph_info), f, indent=2)


def create_experiment_log(experiment_config: Dict[str, Any], 
                         output_dir: str, experiment_name: str) -> None:
    """
    Create a log file with experiment configuration.
    
    Args:
        experiment_config: Dictionary containing experiment configuration
        output_dir: Output directory
        experiment_name: Name of the experiment
    """
    log_path = os.path.join(output_dir, f"{experiment_name}_config.json")
    
    with open(log_path, 'w') as f:
        json.dump(convert_for_json_serialization(experiment_config), f, indent=2)