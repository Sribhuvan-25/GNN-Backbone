#!/usr/bin/env python3
"""
Single-Model Ablation Study Script for GNN Backbone Analysis

This script runs pure single-model pipelines for ablation studies:
- GCN → GCN explainer → GCN embeddings → ML models
- GAT → GAT explainer → GAT embeddings → ML models  
- RGGC → RGGC explainer → RGGC embeddings → ML models

Each model runs independently for clean ablation comparison.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from pipelines.embeddings_pipeline import MixedEmbeddingPipeline
from datasets.domain_expert_dataset import AnchoredMicrobialGNNDataset
from pipelines.case_implementations import CaseImplementations


class SingleModelAblationPipeline(MixedEmbeddingPipeline):
    """
    Pipeline for running pure single-model ablation studies.
    
    Unlike the mixed pipeline that selects best models at each stage,
    this pipeline uses the SAME model throughout the entire process
    for clean ablation analysis.
    """
    
    def __init__(self, model_type: str, case_type: str = 'case3', **kwargs):
        """
        Initialize single-model pipeline.
        
        Args:
            model_type: GNN model to use throughout ('gcn', 'gat', 'rggc')
            case_type: Domain expert case type ('case1', 'case2', 'case3')
            **kwargs: Additional arguments for base pipeline
        """
        self.model_type = model_type.lower()
        self.case_type = case_type
        
        # Validate model type
        valid_models = ['gcn', 'gat', 'rggc']
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}, got {self.model_type}")
        
        # Initialize base pipeline with single model
        super().__init__(**kwargs)
        
        # Override to use only the specified model
        self.gnn_models_to_train = [self.model_type]
        
        # Initialize case implementations for anchored features
        self.case_impl = CaseImplementations()
        anchored_features = self.case_impl.get_case_features(case_type)
        
        # Replace dataset with anchored version if needed
        if case_type in ['case1', 'case2', 'case3']:
            print(f"Using anchored features for {case_type}")
            self.dataset = AnchoredMicrobialGNNDataset(
                data_path=kwargs.get('data_path'),
                anchored_features=anchored_features,
                case_type=case_type,
                k_neighbors=kwargs.get('k_neighbors', 15),
                mantel_threshold=kwargs.get('mantel_threshold', 0.05),
                use_fast_correlation=kwargs.get('use_fast_correlation', False),
                graph_mode=kwargs.get('graph_mode', 'family'),
                family_filter_mode=kwargs.get('family_filter_mode', 'strict')
            )
            self.target_names = self.dataset.target_cols
    
    def run_single_model_ablation(self, target_idx: int, target_name: str) -> Dict[str, Any]:
        """
        Run complete single-model pipeline for ablation study.
        
        Args:
            target_idx: Index of target variable
            target_name: Name of target variable
            
        Returns:
            Dictionary containing all results for this model
        """
        print(f"\n{'='*80}")
        print(f"SINGLE-MODEL ABLATION: {self.model_type.upper()} FOR {target_name}")
        print(f"{'='*80}")
        print(f"Case: {self.case_type}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Target: {target_name}")
        print(f"Features: {len(self.dataset.node_feature_names)}")
        
        results = {
            'model_type': self.model_type,
            'case_type': self.case_type,
            'target_name': target_name,
            'num_features': len(self.dataset.node_feature_names),
            'feature_names': self.dataset.node_feature_names.copy()
        }
        
        # Step 1: Train model on KNN-sparsified graph
        print(f"\nSTEP 1: Training {self.model_type.upper()} on KNN-sparsified graph")
        knn_results = self.train_gnn_model(
            model_type=self.model_type,
            target_idx=target_idx,
            data_list=self.dataset.data_list
        )
        results['knn_phase'] = knn_results
        
        # Step 2: Create explainer-sparsified graph using the SAME model
        print(f"\nSTEP 2: Creating explainer-sparsified graph with {self.model_type.upper()}")
        explainer_data = self.create_explainer_sparsified_graph(
            model=knn_results['model'],
            target_idx=target_idx
        )
        
        # Step 3: Train the SAME model on explainer-sparsified graph
        print(f"\nSTEP 3: Training {self.model_type.upper()} on explainer-sparsified graph")
        explainer_results = self.train_gnn_model(
            model_type=self.model_type,
            target_idx=target_idx,
            data_list=explainer_data
        )
        results['explainer_phase'] = explainer_results
        
        # Step 4: Extract embeddings using explainer-trained model
        print(f"\nSTEP 4: Extracting embeddings from {self.model_type.upper()}")
        embeddings, targets = self.extract_embeddings(explainer_results['model'], explainer_data)
        
        # Save embeddings
        os.makedirs(f"{self.save_dir}/embeddings", exist_ok=True)
        np.save(f"{self.save_dir}/embeddings/{target_name}_{self.model_type}_embeddings.npy", embeddings)
        np.save(f"{self.save_dir}/embeddings/{target_name}_{self.model_type}_targets.npy", targets)
        
        results['embeddings_shape'] = embeddings.shape
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Step 5: Train ML models on embeddings
        print(f"\nSTEP 5: Training ML models on {self.model_type.upper()} embeddings")
        ml_results = self.train_ml_models(embeddings, targets, target_idx)
        results['ml_models'] = ml_results
        
        # Step 6: Generate feature importance
        print(f"\nSTEP 6: Computing feature importance for {self.model_type.upper()}")
        feature_importance = self._compute_gnn_feature_importance(
            explainer_results['model'], explainer_data, target_idx
        )
        results['feature_importance'] = feature_importance
        
        # Step 7: Create visualizations
        print(f"\nSTEP 7: Creating visualizations")
        self.plot_results(
            gnn_results={self.model_type: explainer_results},
            ml_results=ml_results,
            target_idx=target_idx
        )
        
        # Save detailed results
        self._save_ablation_results(results, target_name)
        
        return results
    
    def _compute_gnn_feature_importance(self, model, data_list, target_idx: int) -> Dict[str, Any]:
        """
        Compute feature importance for GNN model using gradient-based methods.
        
        Args:
            model: Trained GNN model
            data_list: Graph data
            target_idx: Target variable index
            
        Returns:
            Dictionary with feature importance scores
        """
        import torch
        from torch_geometric.loader import DataLoader
        
        model.eval()
        model.zero_grad()
        
        # Enable gradient computation for input features
        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
        
        feature_importances = []
        feature_names = self.dataset.node_feature_names
        
        for batch in loader:
            batch = batch.to(self.device if hasattr(self, 'device') else 'cpu')
            
            # Enable gradient computation for node features
            batch.x.requires_grad_(True)
            
            # Forward pass
            output, _ = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y[:, target_idx].view(-1, 1)
            
            # Compute gradients w.r.t. input features
            grad_outputs = torch.ones_like(output)
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=batch.x,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Compute importance as mean absolute gradient per feature
            # batch.x shape: [total_nodes, 1], batch.batch tells us which graph each node belongs to
            importance_per_sample = []
            
            for sample_idx in range(len(data_list)):
                # Get nodes belonging to this sample
                node_mask = batch.batch == sample_idx
                sample_gradients = gradients[node_mask]  # [num_nodes_in_sample, 1]
                sample_features = batch.x[node_mask]     # [num_nodes_in_sample, 1]
                
                # Compute importance as gradient * input (similar to integrated gradients)
                sample_importance = torch.abs(sample_gradients * sample_features).squeeze()
                importance_per_sample.append(sample_importance.detach().cpu().numpy())
            
            # Average importance across samples and nodes
            avg_importance = np.mean([np.mean(imp) for imp in importance_per_sample if len(imp) > 0])
            feature_importances.append(avg_importance)
            
            break  # Only need one batch since we're using all data
        
        # Create feature importance dictionary
        if len(feature_importances) == 1:
            # Single importance value for all features (since we have 1D node features)
            importance_dict = {
                'method': 'gradient_based',
                'overall_importance': feature_importances[0],
                'feature_names': feature_names,
                'num_features': len(feature_names),
                'description': 'Average absolute gradient * input across all nodes and samples'
            }
        else:
            importance_dict = {
                'method': 'gradient_based',
                'importance_per_feature': dict(zip(feature_names, feature_importances)),
                'feature_names': feature_names,
                'num_features': len(feature_names)
            }
        
        return importance_dict
    
    def _save_ablation_results(self, results: Dict[str, Any], target_name: str):
        """Save ablation study results with detailed metrics."""
        
        # Save complete results
        results_file = f"{self.save_dir}/{self.model_type}_{target_name}_ablation_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary DataFrame
        summary_data = []
        
        # KNN phase results
        knn_metrics = results['knn_phase']['avg_metrics']
        summary_data.append({
            'model_type': self.model_type.upper(),
            'case_type': self.case_type,
            'target': target_name,
            'phase': 'KNN',
            'r2': knn_metrics['r2'],
            'mse': knn_metrics['mse'],
            'rmse': knn_metrics['rmse'],
            'mae': knn_metrics['mae'],
            'num_features': results['num_features']
        })
        
        # Explainer phase results
        exp_metrics = results['explainer_phase']['avg_metrics']
        summary_data.append({
            'model_type': self.model_type.upper(),
            'case_type': self.case_type,
            'target': target_name,
            'phase': 'Explainer',
            'r2': exp_metrics['r2'],
            'mse': exp_metrics['mse'],
            'rmse': exp_metrics['rmse'],
            'mae': exp_metrics['mae'],
            'num_features': results['num_features']
        })
        
        # ML models results
        for ml_model, ml_metrics in results['ml_models'].items():
            summary_data.append({
                'model_type': f"{self.model_type.upper()}_embeddings",
                'case_type': self.case_type,
                'target': target_name,
                'phase': ml_model.upper(),
                'r2': ml_metrics['avg_metrics']['r2'],
                'mse': ml_metrics['avg_metrics']['mse'],
                'rmse': ml_metrics['avg_metrics']['rmse'],
                'mae': ml_metrics['avg_metrics']['mae'],
                'num_features': results['embeddings_shape'][1]  # embedding dimension
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"{self.save_dir}/{self.model_type}_{target_name}_ablation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"ABLATION RESULTS SUMMARY: {self.model_type.upper()} - {target_name}")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\nFeature Importance: {results['feature_importance']['overall_importance']:.4f}")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")


def run_single_model_for_all_cases(model_type: str, data_path: str, save_base_dir: str):
    """Run single model across all cases and both targets."""
    
    print(f"\n{'='*80}")
    print(f"RUNNING {model_type.upper()} ACROSS ALL CASES")
    print(f"{'='*80}")
    
    cases = ['case1', 'case2', 'case3']
    all_results = {}
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"CASE: {case} with {model_type.upper()}")
        print(f"{'='*60}")
        
        case_results = {}
        
        try:
            # Initialize pipeline for this case
            pipeline = SingleModelAblationPipeline(
                model_type=model_type,
                case_type=case,
                data_path=data_path,
                k_neighbors=15,
                hidden_dim=512,
                num_epochs=200,
                num_folds=5,
                save_dir=f"{save_base_dir}/{model_type}_{case}",
                importance_threshold=0.2,
                use_fast_correlation=False,
                family_filter_mode='strict'
            )
            
            # Get target indices
            ace_target_idx = pipeline.case_impl.get_target_index(pipeline.target_names, 'ACE')
            h2_target_idx = pipeline.case_impl.get_target_index(pipeline.target_names, 'H2')
            
            # Run for ACE-km
            print(f"\n--- Running {model_type.upper()} for ACE-km in {case} ---")
            case_results['ace_km'] = pipeline.run_single_model_ablation(ace_target_idx, "ACE-km")
            
            # Run for H2-km  
            print(f"\n--- Running {model_type.upper()} for H2-km in {case} ---")
            case_results['h2_km'] = pipeline.run_single_model_ablation(h2_target_idx, "H2-km")
            
            all_results[case] = case_results
            
        except Exception as e:
            print(f"Error running {model_type} for {case}: {e}")
            import traceback
            traceback.print_exc()
            all_results[case] = None
    
    # Save combined results
    combined_file = f"{save_base_dir}/{model_type}_all_cases_ablation.pkl"
    with open(combined_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n{'='*80}")
    print(f"{model_type.upper()} ABLATION STUDY COMPLETED")
    print(f"{'='*80}")
    print(f"Combined results saved to: {combined_file}")
    
    return all_results


def main():
    """Main function for single-model ablation studies."""
    parser = argparse.ArgumentParser(description="Run single-model ablation study")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=['gcn', 'gat', 'rggc'],
        required=True,
        help="GNN model to use for ablation study"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../Data/New_Data.csv",
        help="Path to the input data CSV file"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./ablation_results",
        help="Directory to save ablation results"
    )
    parser.add_argument(
        "--case", 
        type=str, 
        choices=['case1', 'case2', 'case3', 'all'],
        default='all',
        help="Which case to run (default: all)"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        choices=['ace', 'h2', 'both'],
        default='both',
        help="Which target to analyze (default: both)"
    )
    
    args = parser.parse_args()
    
    # Ensure data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.case == 'all':
        # Run model across all cases
        results = run_single_model_for_all_cases(args.model, args.data_path, args.save_dir)
    else:
        # Run specific case
        print(f"Running {args.model.upper()} ablation for {args.case}...")
        
        try:
            pipeline = SingleModelAblationPipeline(
                model_type=args.model,
                case_type=args.case,
                data_path=args.data_path,
                k_neighbors=15,
                hidden_dim=512,
                num_epochs=200,
                num_folds=5,
                save_dir=f"{args.save_dir}/{args.model}_{args.case}",
                importance_threshold=0.2,
                use_fast_correlation=False,
                family_filter_mode='strict'
            )
            
            # Get target indices
            ace_target_idx = pipeline.case_impl.get_target_index(pipeline.target_names, 'ACE')
            h2_target_idx = pipeline.case_impl.get_target_index(pipeline.target_names, 'H2')
            
            results = {}
            
            if args.target in ['ace', 'both']:
                results['ace_km'] = pipeline.run_single_model_ablation(ace_target_idx, "ACE-km")
            
            if args.target in ['h2', 'both']:
                results['h2_km'] = pipeline.run_single_model_ablation(h2_target_idx, "H2-km")
            
            print(f"{args.model.upper()} ablation completed!")
            
        except Exception as e:
            print(f"Error running {args.model} ablation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()